"""EchoNet-Dynamic loader reproducing UVT's preprocessing (Network/dataloader.py of
https://github.com/HReynaud/UVT, MIT licence).

Modes (DTmode):
  'repeat'  ("M", mirror): the frames between ES and ED are played forward then backward until the
            sequence is longer than fixed_length; a random window of fixed_length frames is taken.
            If |ED-ES| > fixed_length//2 the video is first subsampled 1-in-2.
  'sample'  ("R", random): a window around the ES..ED segment, extended by |ED-ES|/U(2,7) on each side,
            subsampled to <= max_length frames and zero-padded to fixed_length at a random offset;
            `attention` marks the real frames.
  'full'    the whole video (testing).
All modes: intensities /255 (no mean/std normalisation), frames zero-padded by `pad` pixels per side
(112 -> 128), RGB.

Per-frame targets (SDmode):
  'reg'  ES = -1, ED = +1; 'repeat' and 'sample' fill frames between them with the original cubic
         interpolation (attenuation 3); 'full' marks only the two labelled frames.
  'cla'  0 = other, 1 = ES, 2 = ED.

ED/ES source (label_order):
  'verified' (default): cache/ed_es_frames.csv from scripts/make_ed_es.py (ED = larger method-of-disks
             volume; EF recomputed from these volumes matches FileList EF, Pearson 0.967).
  'csv':     the original rule: first frame listed in VolumeTracings.csv = ES, last = ED. In this copy of
             the dataset the first-listed frame is the LARGER one in 10,006 / 10,024 videos, so this
             rule swaps ED and ES almost everywhere here; kept only to reproduce the original code.

Deviation: in 'sample' mode the label max-pooling uses ceil_mode=True so the label length always equals
the subsampled video length (the original raises an error when they differ).
"""
from __future__ import annotations

import zlib

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config

ATTENUATION = 3


def _seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def load_video_rgb(path) -> np.ndarray:
    """All frames, uint8 (F, H, W, 3) RGB."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise IOError(f"no frames decoded from {path}")
    return np.stack(frames)[..., ::-1]


def csv_order_frames(tracings=config.TRACINGS):
    """{FileName: (first_listed_frame, last_listed_frame)} following VolumeTracings.csv row order."""
    vt = pd.read_csv(tracings, usecols=["FileName", "Frame"])
    vt["FileName"] = vt["FileName"].str.replace(".avi", "", regex=False)
    first = vt.drop_duplicates("FileName", keep="first").set_index("FileName").Frame
    order = vt.drop_duplicates(["FileName", "Frame"])
    last = order.drop_duplicates("FileName", keep="last").set_index("FileName").Frame
    return {k: (int(first[k]), int(last[k])) for k in first.index}


class UVTEchoSet(Dataset):
    def __init__(self, split, mode="repeat", SDmode="reg", fixed_length=128, max_length=128, min_spacing=10,
                 pad=8, label_order="verified", max_videos=None, seed=0):
        assert mode in ("repeat", "sample", "full")
        assert SDmode in ("reg", "cla")
        assert label_order in ("verified", "csv")
        split = split.upper()
        self.mode, self.SDmode, self.pad = mode, SDmode, pad
        self.fixed_length, self.max_length, self.min_spacing = fixed_length, max_length, min_spacing
        self.seed, self.epoch = seed, 0
        fl = pd.read_csv(config.FILELIST)
        fl = fl[fl.Split == split] if split != "ALL" else fl
        if label_order == "verified":
            ee = pd.read_csv(config.ED_ES_CSV)[["FileName", "ES", "ED"]]
            fl = fl.merge(ee, on="FileName", how="inner")
            small, large = fl.ES.to_numpy(int), fl.ED.to_numpy(int)
        else:
            co = csv_order_frames()
            fl = fl[fl.FileName.isin(co.keys())].reset_index(drop=True)
            small = np.array([co[n][0] for n in fl.FileName])  # original: frames[0] -> "small" (ES)
            large = np.array([co[n][1] for n in fl.FileName])  # original: frames[-1] -> "large" (ED)
        keep = np.abs(small - large) > min_spacing  # original rejects videos with ED/ES too close
        fl, small, large = fl[keep].reset_index(drop=True), small[keep], large[keep]
        if max_videos is not None:
            fl, small, large = fl.iloc[:max_videos], small[:max_videos], large[:max_videos]
        self.names = fl.FileName.to_numpy().astype(str)
        self.ef = fl.EF.to_numpy(np.float32)
        self.fps = fl.FPS.to_numpy(np.int64)
        self.small, self.large = small.astype(np.int64), large.astype(np.int64)

    def __len__(self):
        return len(self.names)

    def set_epoch(self, e):
        self.epoch = e

    def _pad(self, v):
        p = self.pad
        return np.pad(v, ((0, 0), (0, 0), (p, p), (p, p))) if p else v

    def __getitem__(self, i):
        rng = np.random.default_rng(_seed(self.seed, self.epoch, i))
        video = load_video_rgb(config.VIDEO_DIR / f"{self.names[i]}.avi")
        video = np.moveaxis(video, -1, 1).astype(np.float32) / 255.0  # (F, 3, H, W)
        small, large = int(self.small[i]), int(self.large[i])
        small, large = min(small, len(video) - 1), min(large, len(video) - 1)
        out = getattr(self, f"_{self.mode}")(video, small, large, rng)
        out.update(ef=torch.tensor(self.ef[i]), fps=torch.tensor(self.fps[i]), vid=torch.tensor(i))
        return out

    # ---- modes

    def _repeat(self, video, small, large, rng):
        if abs(small - large) > self.fixed_length // 2:
            video, small, large = video[::2], small // 2, large // 2
        f = video.shape[0]
        first, last = min(small, large), max(small, large)
        label = np.zeros(f)
        if self.SDmode == "cla":
            label[small], label[large] = 1, 2
        else:
            label[small], label[large] = -1, 1
            dist = abs(small - large)
            sign = -int((small - large) / dist)
            center = min(small, large) + dist // 2
            t = np.arange(small + sign, large, sign)
            label[small + sign:large:sign] = sign * np.power((t - center) / dist * 2, ATTENUATION)
        fwd = list(range(first, last + 1))
        bwd = list(range(last - 1, first, -1))
        seq = []
        while len(seq) < self.fixed_length + 1:
            seq += fwd + bwd
        seq = np.array(seq)
        start = int(rng.integers(len(seq) - self.fixed_length))
        seq = seq[start:start + self.fixed_length]
        v = self._pad(video[seq])
        return {"video": torch.from_numpy(np.ascontiguousarray(v)), "sd_label": torch.from_numpy(label[seq]).float(),
                "attention": torch.ones(self.fixed_length)}

    def _sample(self, video, small, large, rng):
        if abs(small - large) > self.fixed_length:
            video, small, large = video[::2], small // 2, large // 2
        f, c, h, w = video.shape
        first, last = min(small, large), max(small, large)
        dist = abs(small - large)
        if self.SDmode == "cla":
            label = np.zeros(f)
            label[small], label[large] = 1, 2
        else:
            label = np.ones(f) / 2
            label[small], label[large] = 0, 1
            sign = int((small - large) / dist)
            if sign > 0:  # large -> small
                a = np.arange(max(0, large - dist) + 1, large)
                label[a] = 1 - np.power(np.abs(a - large) / dist, 3)
                a = np.arange(large + 1, small)
                label[a] = np.power(np.abs(a - small) / dist, 3)
                a = np.arange(small + 1, min(f, small + dist))
                label[a] = 1 - np.power(np.abs(a - min(f, small + dist)) / dist, 3)
            else:  # small -> large
                a = np.arange(max(0, small - dist) + 1, small)
                label[a] = np.power(np.abs(a - small) / dist, 3)
                a = np.arange(small + 1, large)
                label[a] = np.power(np.abs(a - small) / dist, 1 / 3)
                a = np.arange(large + 1, min(f, large + dist))
                label[a] = np.power(np.abs(a - min(f, large + dist)) / dist, 3)
            label = label * 2 - 1
        start = int(max(0, first - dist // (rng.random() * 5 + 2)) // 2 * 2)
        end = int(min(f, last + 1 + dist // (rng.random() * 5 + 2)) // 2 * 2)
        step = int(np.ceil((end - start) / self.max_length))
        v = video[start:end:step]
        lab = torch.nn.functional.max_pool1d(torch.tensor(label[None, None, start:end]), step, ceil_mode=True)
        lab = lab.flatten().numpy()[: len(v)]
        if self.SDmode == "cla":  # max-pooling class ids would favour ED over ES; take the subsampled ids
            lab = label[start:end:step]
        ww = v.shape[0]
        miss = self.fixed_length - ww
        before = int(rng.integers(miss)) if miss > 0 else 0
        after = max(miss - before, 0)
        v = np.concatenate((np.zeros((before, c, h, w), np.float32), v, np.zeros((after, c, h, w), np.float32)))
        lab = np.concatenate((np.zeros(before), lab, np.zeros(after)))
        att = np.zeros(self.fixed_length)
        att[before:before + ww] = 1
        v = self._pad(v)
        return {"video": torch.from_numpy(np.ascontiguousarray(v)), "sd_label": torch.from_numpy(lab).float(),
                "attention": torch.from_numpy(att).float()}

    def _full(self, video, small, large, rng):
        f = video.shape[0]
        label = np.zeros(f)
        if self.SDmode == "cla":
            label[small], label[large] = 1, 2
        else:
            label[small], label[large] = -1, 1
        return {"video": torch.from_numpy(np.ascontiguousarray(self._pad(video))),
                "sd_label": torch.from_numpy(label).float(), "attention": torch.ones(f),
                "es": torch.tensor(small), "ed": torch.tensor(large)}
