r"""Cache frozen per-frame embeddings of every EchoNet-Dynamic video (all splits) from a 2D encoder.

Encoder "panecho": PanEcho ConvNeXt-T image encoder (cardiac/panecho.py), PanEcho's EchoNet
preprocessing (resize 256, centre crop 224, ImageNet normalisation), 768-d per frame, bf16 inference.
--stack 3: the image for frame t holds frames (t-1, t, t+1) as its three channels (clamped at the video
ends), which is what PanEcho's encoder saw in training (its FrameTransformer packs consecutive frames
into channels; scripts/panecho_zeroshot.py). Still one embedding per frame, in temporal order.
Output: one float16 .npy per video, (n_decoded_frames, D), in <ECHO_DATA>\cache\feats_<encoder>[_s3]\,
and index.csv (FileName, Split, n_header, n_decoded). Resumable: videos with an existing file are skipped.
The encoder is frozen; nothing here uses labels.

Usage: py.cmd scripts\cache_features.py [--encoder panecho] [--stack 1|3] [--workers 3] [--chunk 128]
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cardiac  # noqa: E402,F401
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from cardiac import config  # noqa: E402


class WholeVideos(Dataset):
    def __init__(self, names, out_dir):
        self.names, self.out_dir = names, out_dir

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        cap = cv2.VideoCapture(f"{config.VIDEO_DIR}/{name}.avi")
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            f = cv2.resize(f[..., ::-1], (256, 256), interpolation=cv2.INTER_AREA)[16:240, 16:240]  # RGB
            frames.append(f)
        cap.release()
        return name, torch.from_numpy(np.ascontiguousarray(np.stack(frames)))  # (T, 224, 224, 3) uint8


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="panecho", choices=["panecho"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--stack", type=int, default=1, choices=[1, 3])
    a = ap.parse_args()
    out_dir = config.CACHE / (f"feats_{a.encoder}" + ("_s3" if a.stack == 3 else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    fl = pd.read_csv(config.FILELIST)
    todo = [n for n in fl.FileName if not (out_dir / f"{n}.npy").exists()]
    print(f"{len(fl)} videos, {len(todo)} to do -> {out_dir}", flush=True)
    dev = torch.device("cuda")
    from cardiac.panecho import MEAN, STD, panecho_image_encoder
    enc = panecho_image_encoder().to(dev).eval().to(memory_format=torch.channels_last)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    dl = DataLoader(WholeVideos(todo, out_dir), batch_size=None, num_workers=a.workers, pin_memory=True)
    t0, n_fr = time.perf_counter(), 0
    for k, (name, fr) in enumerate(dl):
        x = fr.to(dev, non_blocking=True).permute(0, 3, 1, 2)
        if a.stack == 3:
            t = torch.arange(x.shape[0], device=dev)
            nb = torch.stack([(t - 1).clamp_min(0), t, (t + 1).clamp_max(x.shape[0] - 1)], 1)   # (T, 3)
            x = torch.stack([x[nb[:, j], j] for j in range(3)], 1)                            # channel j <- t-1+j
        feats = []
        for i in range(0, x.shape[0], a.chunk):
            xi = ((x[i:i + a.chunk].float() / 255.0 - mean) / std).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats.append(enc(xi).float())
        f = torch.cat(feats).cpu().numpy().astype(np.float16)
        tmp = out_dir / f"{name}.tmp.npy"
        np.save(tmp, f)
        os.replace(tmp, out_dir / f"{name}.npy")
        n_fr += f.shape[0]
        if (k + 1) % 200 == 0 or k + 1 == len(todo):
            dt = time.perf_counter() - t0
            print(f"{k + 1}/{len(todo)} videos, {n_fr} frames, {n_fr / dt:.0f} frames/s, "
                  f"eta {(len(todo) - k - 1) * dt / (k + 1) / 60:.0f} min", flush=True)
    rows = []
    for n, s, nh in zip(fl.FileName, fl.Split, fl.NumberOfFrames):
        p = out_dir / f"{n}.npy"
        rows.append((n, s, int(nh), int(np.load(p, mmap_mode="r").shape[0]) if p.exists() else -1))
    idx = pd.DataFrame(rows, columns=["FileName", "Split", "n_header", "n_decoded"])
    idx.to_csv(out_dir / "index.csv", index=False)
    print("index written; header/decoded mismatches:", int((idx.n_header != idx.n_decoded).sum()),
          "missing:", int((idx.n_decoded < 0).sum()), flush=True)


if __name__ == "__main__":
    main()
