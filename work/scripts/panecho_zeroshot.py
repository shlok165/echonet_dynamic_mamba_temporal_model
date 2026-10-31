r"""PanEcho as released (frozen, zero-shot, no training here): image encoder + temporal Transformer + EF head.

Rebuilt from the CarDS-Yale/PanEcho source (src/models.py FrameTransformer + MultiTaskModel, hubconf
settings: ConvNeXt-T, 4 post-norm Transformer layers, d 768, 8 heads, FFN 768, ReLU, dropout 0, fixed
sinusoidal positions for 16 frames, mean over time, EF_head = Dropout(0.25) + Linear(768, 1), EF in %).
Only tensors are loaded (torch.load weights_only=True); hubconf is not run.

Input as in PanEcho's loaders: 16 consecutive frames (native frame rate), frames in OpenCV BGR order,
resized to 256 (INTER_AREA), centre-cropped to 224, ImageNet-normalised; --n-clips evenly spaced clips
per video, EF averaged per video. --packing quirk (default) reproduces FrameTransformer.forward exactly:
it reshapes (b, c, l, h, w) -> (b*l, c, h, w) without moving time first, so each image the CNN sees
packs three consecutive frames as its channels; --packing frame feeds one frame per image instead;
--packing stack3 feeds, for every clip frame t in temporal order, the frames (t-1, t, t+1) as channels.

Writes results/panecho_zeroshot[_frame]/{split}_preds_T16.npz (same format as evaluate.py) + eval json.
Usage: py.cmd scripts\panecho_zeroshot.py --split VAL [--packing quirk|frame]
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cardiac  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from cardiac import config  # noqa: E402
from cardiac import metrics as M  # noqa: E402
from cardiac.dataset import EchoClips, load_frames  # noqa: E402
from cardiac.panecho import WEIGHTS, panecho_image_encoder, preprocess  # noqa: E402


class PanEchoEF(nn.Module):
    def __init__(self, packing="quirk", clip_len=16):
        super().__init__()
        self.packing = packing
        self.encoder = panecho_image_encoder()
        layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=768, dropout=0.0,
                                           activation="relu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=4, enable_nested_tensor=False)
        self.EF_head = nn.Sequential(nn.Dropout(0.25), nn.Linear(768, 1))
        pos = torch.arange(clip_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, 768, 2) * (-math.log(10000.0) / 768))
        pe = torch.zeros(1, clip_len, 768)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)["weights"]
        own = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.transformer.")}
        own.update({k: v for k, v in sd.items() if k.startswith("EF_head.")})
        missing, unexpected = self.load_state_dict(own, strict=False)
        assert not unexpected and all(k.startswith("encoder.") or k == "pe" for k in missing), (missing, unexpected)
        self.pe_max_abs_diff = float((sd["encoder.time_encoder.pe"] - pe).abs().max())

    def forward(self, x):
        """x: (b, l, 3, 224, 224) normalised frames -> EF in % (b,)."""
        b, l = x.shape[:2]
        if self.packing == "quirk":
            x = x.permute(0, 2, 1, 3, 4).reshape(b * l, 3, *x.shape[3:])    # PanEcho: (b,c,l,h,w).reshape
        else:
            x = x.reshape(b * l, 3, *x.shape[3:])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = self.encoder(x.contiguous(memory_format=torch.channels_last))
        e = e.float().view(b, l, 768) + self.pe[:, :l]
        return self.EF_head(self.transformer(e).mean(1)).squeeze(-1)


class Clips(Dataset):
    def __init__(self, split, n_clips, packing):
        self.ds = EchoClips(split, T=16, stride=1, pad="loop", mode="multi", n_clips=n_clips)
        self.names, self.packing = self.ds.names, packing

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        it = self.ds[i]
        if self.packing == "stack3":
            v = int(it["vid"])
            idx = it["frame_idx"].numpy()
            n = int(self.ds.n_frames[v])
            nb = np.stack([np.clip(idx - 1, 0, n - 1), idx, np.clip(idx + 1, 0, n - 1)], 1).ravel()
            fr = load_frames(f"{self.ds.video_dir}/{self.names[v]}.avi", nb)[..., ::-1]   # (3T,H,W,3) BGR
            p = preprocess(np.ascontiguousarray(fr)).view(len(idx), 3, 3, 224, 224)   # (T, neighbour, ch, H, W)
            it["video"] = torch.stack([p[:, j, j] for j in range(3)], 1)            # channel j = neighbour j
            return it
        frames = it["video"].permute(0, 2, 3, 1).numpy()[..., ::-1]          # RGB -> BGR (PanEcho used cv2 BGR)
        it["video"] = preprocess(np.ascontiguousarray(frames))
        return it


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="VAL", choices=["VAL", "TEST"])
    ap.add_argument("--packing", default="quirk", choices=["quirk", "frame", "stack3"])
    ap.add_argument("--n-clips", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    dev = torch.device("cuda")
    model = PanEchoEF(a.packing).to(dev).eval()
    print("pe |stored - computed| max:", model.pe_max_abs_diff)
    ds = Clips(a.split, a.n_clips, a.packing)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=a.workers, pin_memory=True)
    out = {k: [] for k in ("pred", "ef", "vid", "clip")}
    for b in dl:
        out["pred"].append(model(b["video"].to(dev, non_blocking=True)).float().cpu().numpy())
        for k in ("ef", "vid", "clip"):
            out[k].append(b[k].numpy())
    r = {k: np.concatenate(v) for k, v in out.items()}
    u, pm = M.video_mean(r["vid"], r["pred"])
    _, ym = M.video_mean(r["vid"], r["ef"])
    res = M.regression_metrics(ym, pm)
    res.update(M.threshold_metrics(ym, pm, 40))
    res.update(M.threshold_metrics(ym, pm, 50))
    name = "panecho_zeroshot" + ("" if a.packing == "quirk" else f"_{a.packing}")
    out_dir = config.RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{a.split.lower()}_preds_T16.npz", names=ds.names, vid=r["vid"], clip=r["clip"],
                        pred=r["pred"], ef=r["ef"])
    rec = {"split": a.split, "packing": a.packing, "n_clips": a.n_clips, "clip": "16 consecutive frames",
           "n_videos": int(len(u)), "pe_max_abs_diff": model.pe_max_abs_diff, "ef": res}
    (out_dir / f"eval_{a.split}.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("MAE", "RMSE", "R2", "Pearson")}, indent=1), "n videos", len(u))


if __name__ == "__main__":
    main()
