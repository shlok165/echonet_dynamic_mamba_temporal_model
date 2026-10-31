r"""Does the area readout's mask (learned from EF labels only) find the left ventricle?

For traced clips, the soft mask at the traced ED and ES frames (bilinearly upsampled to 112x112) is
compared with the expert LV tracing (polygon from the chord end points, as in the EchoNet code; the
tracings are never used in training). The mask is only defined up to its level (EF depends on area
ratios), so threshold-free agreement is reported: per-frame Pearson correlation and soft Dice; plus
Dice after per-frame max-normalisation at one global threshold (chosen on VAL; pass --threshold to
reuse it on TEST), and the correlation of mask-area ratios (ES/ED) with traced-area ratios.

Usage: py.cmd scripts\mask_dice.py --run sync_hr_area_T32_s0 [--split VAL] [--threshold 0.5]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cardiac  # noqa: E402,F401
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cardiac import config  # noqa: E402
from cardiac.dataset import EchoClips  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_model  # noqa: E402


def traced_masks(size=112):
    """{(video name, frame): binary LV mask} from VolumeTracings.csv (first 21-row block per frame)."""
    vt = pd.read_csv(config.TRACINGS)
    vt["FileName"] = vt["FileName"].str.replace(".avi", "", regex=False)
    out = {}
    for (fn, fr), g in vt.groupby(["FileName", "Frame"], sort=False):
        t = g[["X1", "Y1", "X2", "Y2"]].to_numpy()
        t = t[:21] if len(t) % 21 == 0 else t
        x = np.concatenate((t[1:, 0], t[1:, 2][::-1]))
        y = np.concatenate((t[1:, 1], t[1:, 3][::-1]))
        m = np.zeros((size, size), np.uint8)
        cv2.fillPoly(m, [np.stack((np.rint(x), np.rint(y)), 1).astype(np.int32)], 1)
        out[(fn, int(fr))] = m.astype(bool)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="VAL", choices=["VAL", "TEST"])
    ap.add_argument("--threshold", type=float, default=None, help="reuse a threshold chosen on VAL")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    dev = torch.device("cuda")
    model, targs, _ = load_model(a.run, dev)
    tb = model.temporal
    assert getattr(tb, "area_readout", False), "run has no area readout"
    ds = EchoClips(a.split, T=targs["T"], stride=targs["stride"], pad=targs["pad"], mode="traced",
                   require_tracings=True)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=a.workers, pin_memory=True)
    gt = traced_masks()
    soft, hard, names, kinds, frames, imgs = [], [], [], [], [], []
    for b in dl:
        v = b["video"].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(v, b["valid"].to(dev) if targs["pool_valid"] else None)
        mk = F.interpolate(tb.last_mask.flatten(0, 1)[:, None].float(), size=(112, 112), mode="bilinear",
                           align_corners=False).view(v.shape[0], v.shape[1], 112, 112).cpu().numpy()
        fi = b["frame_idx"].numpy()
        for i in range(v.shape[0]):
            name = ds.names[int(b["vid"][i])]
            for kind, tf in (("ED", int(b["ed"][i])), ("ES", int(b["es"][i]))):
                j = int(np.argmin(np.abs(fi[i] - tf)))
                if fi[i][j] != tf or (name, tf) not in gt:
                    continue                      # only frames that are exactly in the clip
                soft.append(mk[i, j])
                hard.append(gt[(name, tf)])
                names.append(name)
                kinds.append(kind)
                frames.append(tf)
                if len(imgs) < 12:
                    imgs.append(v[i, j].float().mean(0).cpu().numpy())
    S = np.stack(soft).reshape(len(soft), -1).astype(np.float64)
    G = np.stack(hard).reshape(len(hard), -1).astype(np.float64)
    Sc, Gc = S - S.mean(1, keepdims=True), G - G.mean(1, keepdims=True)
    corr = (Sc * Gc).sum(1) / np.sqrt((Sc ** 2).sum(1) * (Gc ** 2).sum(1) + 1e-12)
    soft_dice = 2 * (S * G).sum(1) / (S.sum(1) + G.sum(1) + 1e-12)
    Sn = S / S.max(1, keepdims=True).clip(1e-12)

    def dice_at(th):
        P = Sn >= th
        return 2 * (P & (G > 0)).sum(1) / (P.sum(1) + G.sum(1) + 1e-12)
    if a.threshold is None:
        ths = np.linspace(0.05, 0.95, 19)
        th = float(ths[np.argmax([dice_at(t).mean() for t in ths])])
    else:
        th = a.threshold
    dice = dice_at(th)
    # area ratio ES/ED: mask vs tracing, per video with both frames
    df = pd.DataFrame({"name": names, "kind": kinds, "mask_area": S.mean(1), "gt_area": G.mean(1)})
    piv = df.pivot_table(index="name", columns="kind", values=["mask_area", "gt_area"]).dropna()
    r_ratio = float(np.corrcoef(piv["mask_area"]["ES"] / piv["mask_area"]["ED"],
                                piv["gt_area"]["ES"] / piv["gt_area"]["ED"])[0, 1]) if len(piv) > 2 else None
    out = {"run": a.run, "split": a.split, "n_frames": int(len(S)),
           "pearson_mask_vs_tracing_mean": float(corr.mean()), "soft_dice_mean": float(soft_dice.mean()),
           "threshold_on_max_normalised_mask": th, "threshold_chosen_on": "this split" if a.threshold is None else "given",
           "dice_mean": float(dice.mean()), "dice_median": float(np.median(dice)),
           "n_videos_ed_and_es": int(len(piv)), "pearson_area_ratio_ES_over_ED": r_ratio}
    print(json.dumps(out, indent=1))
    out_dir = config.RESULTS / a.run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"mask_dice_{a.split}.json").write_text(json.dumps(out, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 6, figsize=(13, 4.6), dpi=130)
    for ax, k in zip(axes.ravel(), range(len(imgs))):
        ax.imshow(imgs[k], cmap="gray")
        ax.imshow(np.stack(soft)[k], cmap="magma", alpha=0.45)
        ax.contour(np.stack(hard)[k].astype(float), levels=[0.5], colors="#00e5ff", linewidths=0.8)
        ax.set_title(f"{kinds[k]}  Dice {dice[k]:.2f}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{a.run}: learned mask (heat) vs expert LV tracing (cyan), {a.split}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"mask_vs_tracing_{a.split}.png")
    print("wrote", out_dir / f"mask_vs_tracing_{a.split}.png")


if __name__ == "__main__":
    main()
