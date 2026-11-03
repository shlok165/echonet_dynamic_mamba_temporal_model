"""EchoNet-Dynamic clip dataset with on-the-fly cv2 decoding.

Design notes
* Official FileList.csv split is used as-is (TRAIN/VAL/TEST); nothing is re-split.
* Videos are decoded inside __getitem__ (DataLoader workers). The dataset object holds only small
  numpy arrays, so it pickles cheaply to Windows spawn workers.
* Items are uint8 tensors (T, 3, 112, 112), RGB. Normalisation runs on the GPU (`Normalize`),
  which keeps worker->main transfers 4x smaller than float32.
* All 3 channels are kept (they are not identical in these AVIs).

Clip sampling (the rule is a DECISION PENDING; every option is a parameter so all models share it):
  T       clip length in sampled frames
  stride  take every `stride`-th original frame
  span    (T-1)*stride+1 original frames
  pad     what to do if a video is shorter than the span:
          "loop"   wrap around to frame 0 (introduces one phase discontinuity at the seam)
          "repeat" repeat the last frame (a frozen heart at the end)
          "fit"    lower the stride for this video until the span fits, then loop if even
                   stride 1 is too long (keeps every sampled frame real when possible)
  Each item also returns `valid` (T,) = frame is a real first-pass frame (False for looped/
  repeated padding) so metrics such as Z_t smoothness can ignore padding.

Clip placement
  mode="random": one clip per video per epoch, uniformly random start (training).
  mode="multi":  `n_clips` clips per video with evenly spaced starts (evaluation; also gives the
                 "EF std across clips of the same video" metric). Deterministic.
  mode="first":  a single clip starting at frame 0.
  mode="traced": a single clip centred between the traced ED and ES frames (ED/ES probe).

frame_order: "normal" | "shuffle" | "reverse" for the frame-order test. Shuffling is a fixed
permutation per (video, clip) derived from `order_seed`, so evaluation is reproducible.

Augmentation (training only, one transform per clip so motion is preserved):
  aug="basic":  EchoNet-style zero-pad by 12 px + random crop
  aug="strong": basic + random rotation (+-15 deg) and scale (0.9-1.1) about the image centre,
                brightness (+-0.1), contrast (0.8-1.2) and gamma (0.8-1.25) jitter, and time reversal
                with p=0.5 (EF is the same when a clip is played backwards)
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config

PAD_MODES = ("loop", "repeat", "fit")


def clip_indices(n_frames: int, T: int, stride: int, start: int = 0, pad: str = "loop"):
    """Original-frame indices of one clip, and a mask of real (non-padding) frames.

    Returns (idx, valid, eff_stride): idx int64 (T,), valid bool (T,), the stride actually used.
    For videos at least as long as the span, `start` must be in [0, n_frames - span].
    """
    if pad not in PAD_MODES:
        raise ValueError(f"pad must be one of {PAD_MODES}")
    if n_frames < 1:
        raise ValueError("empty video")
    s = stride
    if pad == "fit":
        while s > 1 and (T - 1) * s + 1 > n_frames:
            s -= 1
    span = (T - 1) * s + 1
    pos = np.arange(T, dtype=np.int64) * s
    if span <= n_frames:
        if not 0 <= start <= n_frames - span:
            raise ValueError(f"start {start} out of range for n={n_frames}, span={span}")
        return start + pos, np.ones(T, dtype=bool), s
    # short video: start at 0 and pad
    valid = pos < n_frames
    if pad == "repeat":
        idx = np.minimum(pos, n_frames - 1)
    else:  # loop (also "fit" when even stride 1 does not fit)
        idx = pos % n_frames
    return idx, valid, s


def clip_starts(n_frames: int, T: int, stride: int, n_clips: int, pad: str = "loop"):
    """Evenly spaced valid start positions (deterministic) for multi-clip evaluation."""
    s = stride
    if pad == "fit":
        while s > 1 and (T - 1) * s + 1 > n_frames:
            s -= 1
    max_start = n_frames - ((T - 1) * s + 1)
    if max_start <= 0:
        return [0] * n_clips  # short video: every clip is the same padded clip
    return [int(round(x)) for x in np.linspace(0, max_start, n_clips)]


def load_frames(path: str | Path, idx: np.ndarray) -> np.ndarray:
    """Decode the frames listed in idx (may repeat / be unordered). Returns uint8 (T, H, W, 3) RGB.

    Reads sequentially up to max(idx) (sequential decoding is faster than seeking in MJPEG/XVID AVIs).
    """
    need = int(idx.max()) + 1
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    frames = []
    try:
        for _ in range(need):
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
    finally:
        cap.release()
    if not frames:
        raise IOError(f"no frames decoded from {path}")
    if len(frames) < need:
        # header frame count over-states the stream: clamp (logged by the caller via n_decoded)
        idx = np.minimum(idx, len(frames) - 1)
    arr = np.stack([frames[i] for i in idx])  # (T, H, W, 3) BGR
    return arr[..., ::-1]  # RGB


def _stable_seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


class EchoClips(Dataset):
    def __init__(self, split: str, T: int = 32, stride: int = 2, pad: str = "loop",
                 mode: str = "random", n_clips: int = 1, frame_order: str = "normal",
                 order_seed: int = 0, augment: bool = False, pad_px: int = 12, aug: str = "basic",
                 subset_frac: float | None = None, subset_seed: int = 0,
                 require_tracings: bool = False, max_videos: int | None = None,
                 filelist: str | Path = config.FILELIST, video_dir: str | Path = config.VIDEO_DIR,
                 ed_es_csv: str | Path = config.ED_ES_CSV):
        split = split.upper()
        assert split in ("TRAIN", "VAL", "TEST")
        assert mode in ("random", "multi", "first", "traced")
        assert frame_order in ("normal", "shuffle", "reverse")
        if pad not in PAD_MODES:
            raise ValueError(pad)
        self.T, self.stride, self.pad, self.mode = T, stride, pad, mode
        self.n_clips = n_clips if mode == "multi" else 1
        self.frame_order, self.order_seed = frame_order, order_seed
        assert aug in ("basic", "strong")
        self.augment, self.pad_px, self.aug = augment, pad_px, aug
        self.video_dir = str(video_dir)

        fl = pd.read_csv(filelist)
        fl = fl[fl.Split == split].reset_index(drop=True)
        ed_es = pd.read_csv(ed_es_csv)[["FileName", "ED", "ES"]] if Path(ed_es_csv).exists() else None
        if ed_es is not None:
            fl = fl.merge(ed_es, on="FileName", how="left")
        else:
            fl["ED"] = np.nan
            fl["ES"] = np.nan
        if require_tracings:
            fl = fl[fl.ED.notna()].reset_index(drop=True)
        if subset_frac is not None and subset_frac < 1.0:
            rng = np.random.default_rng(subset_seed)
            keep = np.sort(rng.choice(len(fl), size=int(round(subset_frac * len(fl))), replace=False))
            fl = fl.iloc[keep].reset_index(drop=True)
        if max_videos is not None:
            fl = fl.iloc[:max_videos].reset_index(drop=True)

        # light-weight per-video arrays (pickled to every worker on Windows)
        self.names = fl.FileName.to_numpy().astype(str)
        self.ef = fl.EF.to_numpy(np.float32)
        self.n_frames = fl.NumberOfFrames.to_numpy(np.int64)
        self.fps = fl.FPS.to_numpy(np.float32)
        self.ed = fl.ED.fillna(-1).to_numpy(np.int64)
        self.es = fl.ES.fillna(-1).to_numpy(np.int64)
        self.split = split
        self.sample_seed = 0  # set by the training script

    def __len__(self):
        return len(self.names) * self.n_clips

    def _start(self, v: int, k: int, rng: np.random.Generator) -> int:
        n = int(self.n_frames[v])
        if self.mode == "multi":
            return clip_starts(n, self.T, self.stride, self.n_clips, self.pad)[k]
        if self.mode == "first":
            return 0
        s = self.stride
        if self.mode == "traced":
            # clip centred between the traced ED and ES frames (for the frozen ED/ES probe)
            while self.pad == "fit" and s > 1 and (self.T - 1) * s + 1 > n:
                s -= 1
            span = (self.T - 1) * s + 1
            if span > n or self.ed[v] < 0:
                return 0
            mid = (int(self.ed[v]) + int(self.es[v])) / 2
            return int(min(max(round(mid - span / 2), 0), n - span))
        if self.pad == "fit":
            while s > 1 and (self.T - 1) * s + 1 > n:
                s -= 1
        max_start = n - ((self.T - 1) * s + 1)
        return int(rng.integers(0, max_start + 1)) if max_start > 0 else 0

    def __getitem__(self, i: int):
        # Indices >= len(self) encode the epoch (see EpochSampler), so random clip starts and
        # augmentation are a deterministic function of (seed, epoch, item) even with persistent workers.
        epoch, i = divmod(int(i), len(self))
        v, k = divmod(i, self.n_clips)
        rng = (np.random.default_rng(_stable_seed("item", self.sample_seed, epoch, i))
               if self.mode == "random" else None)
        start = self._start(v, k, rng)
        idx, valid, eff_stride = clip_indices(int(self.n_frames[v]), self.T, self.stride, start, self.pad)

        order = np.arange(self.T)
        if self.frame_order == "shuffle":
            order = np.random.default_rng(_stable_seed(self.order_seed, self.names[v], k)).permutation(self.T)
        elif self.frame_order == "reverse":
            order = order[::-1].copy()
        idx, valid = idx[order], valid[order]

        frames = load_frames(f"{self.video_dir}/{self.names[v]}.avi", idx)  # (T,H,W,3)
        if self.augment:
            frames = self._pad_crop(frames, rng)
            if self.aug == "strong":
                frames = self._strong(frames, rng)
                if rng.random() < 0.5:   # time reversal: same EF, reversed motion
                    frames, idx, valid = frames[::-1], idx[::-1].copy(), valid[::-1].copy()
        x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)  # (T,3,H,W) uint8

        return {
            "video": x,
            "ef": torch.tensor(self.ef[v]),
            "frame_idx": torch.from_numpy(idx),          # original frame index of every clip frame
            "valid": torch.from_numpy(valid),
            "stride": torch.tensor(eff_stride),
            "ed": torch.tensor(self.ed[v]),               # -1 if no tracing
            "es": torch.tensor(self.es[v]),
            "n_frames": torch.tensor(self.n_frames[v]),
            "fps": torch.tensor(self.fps[v]),
            "vid": torch.tensor(v),
            "clip": torch.tensor(k),
        }

    @staticmethod
    def _strong(frames: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Random rotation/scale (same for all frames) and intensity jitter, uint8 in and out."""
        T, H, W, C = frames.shape
        Mx = cv2.getRotationMatrix2D((W / 2, H / 2), float(rng.uniform(-15, 15)), float(rng.uniform(0.9, 1.1)))
        out = np.empty_like(frames)
        for t in range(T):
            out[t] = cv2.warpAffine(frames[t], Mx, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        # contrast, brightness, gamma as one 256-entry lookup table (same result as float maths, faster)
        x = np.arange(256, dtype=np.float32) / 255.0
        x = np.clip((x - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.1, 0.1), 0.0, 1.0)
        lut = (x ** rng.uniform(0.8, 1.25) * 255.0 + 0.5).astype(np.uint8)
        return lut[out]

    def _pad_crop(self, frames: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """EchoNet-style augmentation: zero-pad by pad_px and take a random crop (same for all frames)."""
        p = self.pad_px
        T, H, W, C = frames.shape
        padded = np.zeros((T, H + 2 * p, W + 2 * p, C), dtype=frames.dtype)
        padded[:, p:p + H, p:p + W] = frames
        r, c = rng.integers(0, 2 * p + 1, size=2)
        return padded[:, r:r + H, c:c + W]


class FeatClips(EchoClips):
    """EchoClips over cached frozen per-frame features (scripts/cache_features.py) instead of frames.

    Same clip sampling, clip placement, frame orders and returned keys; "video" holds float16 (T, D)
    features. n_frames is the cache's decoded frame count. Pixel augmentations do not apply to cached
    features; with augment=True and aug="strong" only the time reversal (p=0.5) is used.
    """

    def __init__(self, split: str, feat_dir: str | Path, clip_mode: str = "even", warp: float = 1.0,
                 hr_csv: str | Path | None = None, **kw):
        """clip_mode (mode="multi" only): "even" = evenly spaced starts; "phase" = starts one whole-video
        cardiac period apart, beginning at the same phase of the cycle (period from hr_csv =
        results/hr_reference.csv, phase from the first principal component of the video's features; both
        label-free), n_clips of them spread over the video. warp: time-warp factor (evaluation only): frames
        are taken every stride * warp original frames and features linearly interpolated, so warp > 1 plays
        the heart faster, warp < 1 slower."""
        super().__init__(split, **kw)
        idx = pd.read_csv(Path(feat_dir) / "index.csv").set_index("FileName")
        self.feat_dir = str(feat_dir)
        self.n_frames = idx.loc[self.names, "n_decoded"].to_numpy(np.int64)
        assert clip_mode in ("even", "phase")
        self.clip_mode, self.warp = clip_mode, float(warp)
        self.phase_starts = None
        if clip_mode == "phase" and self.mode == "multi":
            hr = pd.read_csv(hr_csv or (config.RESULTS / "hr_reference.csv")).set_index("FileName")["period_frames"]
            self.phase_starts = [self._phase_starts(v, float(hr.get(n, np.nan))) for v, n in enumerate(self.names)]

    def _phase_starts(self, v, period):
        n = int(self.n_frames[v])
        even = clip_starts(n, self.T, self.stride, self.n_clips, self.pad)
        span = (self.T - 1) * self.stride + 1
        if not np.isfinite(period) or period < 2 or span > n:
            return even
        x = np.load(f"{self.feat_dir}/{self.names[v]}.npy").astype(np.float32)
        x = x.reshape(len(x), -1) - x.reshape(len(x), -1).mean(0)
        pc = x @ np.linalg.svd(x, full_matrices=False)[2][0]                     # first principal component
        t = np.arange(n)
        w = 2 * np.pi / period
        phi = np.arctan2((pc * np.sin(w * t)).sum(), (pc * np.cos(w * t)).sum())   # pc ~ cos(w t - phi)
        s0 = (phi / w) % period
        cand = [int(round(s0 + j * period)) for j in range(int(n / period) + 2)]
        cand = [c for c in cand if 0 <= c <= n - span]
        if not cand:
            return even
        return [cand[int(round(q))] for q in np.linspace(0, len(cand) - 1, self.n_clips)]

    def _start(self, v, k, rng):
        if self.phase_starts is not None:
            return self.phase_starts[v][k]
        return super()._start(v, k, rng)

    def _warped(self, v, start):
        """Fractional frame positions every stride * warp frames from start; linear interpolation."""
        n = int(self.n_frames[v])
        step = self.stride * self.warp
        start = min(start, max(0.0, n - 1 - (self.T - 1) * step))
        pos = start + np.arange(self.T) * step
        valid = pos <= n - 1
        pos = pos % n if self.pad == "loop" else np.minimum(pos, n - 1)
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, n - 1)
        f = np.load(f"{self.feat_dir}/{self.names[v]}.npy", mmap_mode="r")
        a = (pos - i0).astype(np.float32).reshape(-1, *([1] * (f.ndim - 1)))     # broadcast over feature dims
        feats = (1 - a) * f[i0].astype(np.float32) + a * f[i1].astype(np.float32)
        return feats.astype(np.float16), np.round(pos).astype(np.int64) % n, valid

    def __getitem__(self, i: int):
        epoch, i = divmod(int(i), len(self))
        v, k = divmod(i, self.n_clips)
        rng = (np.random.default_rng(_stable_seed("item", self.sample_seed, epoch, i))
               if self.mode == "random" else None)
        start = self._start(v, k, rng)
        order = np.arange(self.T)
        if self.frame_order == "shuffle":
            order = np.random.default_rng(_stable_seed(self.order_seed, self.names[v], k)).permutation(self.T)
        elif self.frame_order == "reverse":
            order = order[::-1].copy()
        if self.warp != 1.0:
            feats, idx, valid = self._warped(v, start)
            feats, idx, valid, eff_stride = feats[order], idx[order], valid[order], self.stride
        else:
            idx, valid, eff_stride = clip_indices(int(self.n_frames[v]), self.T, self.stride, start, self.pad)
            idx, valid = idx[order], valid[order]
            feats = np.load(f"{self.feat_dir}/{self.names[v]}.npy", mmap_mode="r")[idx]
        if self.augment and self.aug == "strong" and rng.random() < 0.5:
            feats, idx, valid = feats[::-1], idx[::-1].copy(), valid[::-1].copy()
        return {
            "video": torch.from_numpy(np.ascontiguousarray(feats)),
            "ef": torch.tensor(self.ef[v]),
            "frame_idx": torch.from_numpy(idx),
            "valid": torch.from_numpy(valid),
            "stride": torch.tensor(eff_stride),
            "ed": torch.tensor(self.ed[v]),
            "es": torch.tensor(self.es[v]),
            "n_frames": torch.tensor(self.n_frames[v]),
            "fps": torch.tensor(self.fps[v]),
            "vid": torch.tensor(v),
            "clip": torch.tensor(k),
        }


class StackClips(EchoClips):
    """EchoClips with PanEcho's input images (for fine-tuning the PanEcho encoder, cardiac/panecho.py).

    Clip frame t becomes one 224x224 image whose channel j is channel j of video frame t-1+j (clamped at
    the video ends), after PanEcho's preprocessing (resize to 256 with cv2.INTER_AREA, centre crop 224):
    exactly the images that scripts/cache_features.py --stack 3 encodes, so a frozen encoder on these
    clips reproduces the feats_panecho_s3 cache. Same clip sampling and returned keys as EchoClips;
    "video" is uint8 (T, 3, 224, 224). No pixel augmentation.
    """

    def __init__(self, split: str, **kw):
        if kw.get("augment"):
            raise ValueError("StackClips has no augmentation")
        super().__init__(split, **kw)

    def __getitem__(self, i: int):
        epoch, i = divmod(int(i), len(self))
        v, k = divmod(i, self.n_clips)
        rng = (np.random.default_rng(_stable_seed("item", self.sample_seed, epoch, i))
               if self.mode == "random" else None)
        start = self._start(v, k, rng)
        n = int(self.n_frames[v])
        idx, valid, eff_stride = clip_indices(n, self.T, self.stride, start, self.pad)
        order = np.arange(self.T)
        if self.frame_order == "shuffle":
            order = np.random.default_rng(_stable_seed(self.order_seed, self.names[v], k)).permutation(self.T)
        elif self.frame_order == "reverse":
            order = order[::-1].copy()
        idx, valid = idx[order], valid[order]
        nb = np.stack([np.maximum(idx - 1, 0), idx, np.minimum(idx + 1, n - 1)], 1)    # (T, 3)
        u, inv = np.unique(nb, return_inverse=True)
        frames = load_frames(f"{self.video_dir}/{self.names[v]}.avi", u)               # (U, H, W, 3) RGB
        big = np.stack([cv2.resize(np.ascontiguousarray(f), (256, 256), interpolation=cv2.INTER_AREA)[16:240, 16:240]
                        for f in frames])                                             # (U, 224, 224, 3)
        inv = inv.reshape(nb.shape)
        x = np.stack([big[inv[:, j], :, :, j] for j in range(3)], 1)                  # (T, 3, 224, 224)
        return {
            "video": torch.from_numpy(x),
            "ef": torch.tensor(self.ef[v]),
            "frame_idx": torch.from_numpy(idx),
            "valid": torch.from_numpy(valid),
            "stride": torch.tensor(eff_stride),
            "ed": torch.tensor(self.ed[v]),
            "es": torch.tensor(self.es[v]),
            "n_frames": torch.tensor(self.n_frames[v]),
            "fps": torch.tensor(self.fps[v]),
            "vid": torch.tensor(v),
            "clip": torch.tensor(k),
        }


class EpochSampler(torch.utils.data.Sampler):
    """Seeded shuffling that also tells the dataset the epoch (yields epoch*n + i)."""

    def __init__(self, n: int, seed: int = 0, shuffle: bool = True):
        self.n, self.seed, self.shuffle, self.epoch = n, seed, shuffle, 0

    def set_epoch(self, e: int):
        self.epoch = e

    def __len__(self):
        return self.n

    def __iter__(self):
        order = (np.random.default_rng(_stable_seed("perm", self.seed, self.epoch)).permutation(self.n)
                 if self.shuffle else np.arange(self.n))
        return iter((self.epoch * self.n + order).tolist())


class Normalize(torch.nn.Module):
    """uint8 (B,T,3,H,W) -> float normalised with TRAIN per-channel mean/std. Runs on GPU."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 1, 3, 1, 1))

    def forward(self, x):
        return (x.float() - self.mean) / self.std


def pixel_stats(n_videos: int = 500, seed: int = 0, recompute: bool = False):
    """Per-channel (RGB) mean/std over a random sample of TRAIN videos, cached to JSON."""
    if config.STATS_JSON.exists() and not recompute:
        d = json.loads(config.STATS_JSON.read_text())
        return d["mean"], d["std"]
    fl = pd.read_csv(config.FILELIST)
    fl = fl[fl.Split == "TRAIN"]
    names = np.random.default_rng(seed).choice(fl.FileName.to_numpy(), size=n_videos, replace=False)
    s = np.zeros(3)
    s2 = np.zeros(3)
    n = 0
    for nm in names:
        cap = cv2.VideoCapture(str(config.VIDEO_DIR / f"{nm}.avi"))
        while True:
            ok, f = cap.read()
            if not ok:
                break
            f = f[..., ::-1].reshape(-1, 3).astype(np.float64)
            s += f.sum(0)
            s2 += (f ** 2).sum(0)
            n += f.shape[0]
        cap.release()
    mean = s / n
    std = np.sqrt(s2 / n - mean ** 2)
    config.CACHE.mkdir(parents=True, exist_ok=True)
    config.STATS_JSON.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist(),
                                             "n_videos": n_videos, "seed": seed}))
    return mean.tolist(), std.tolist()
