r"""Interpretability of the sync block (no training, model frozen): the learned log-volume curve and clock.

On traced clips (one clip centred between the traced ED and ES frames), without any fitting:
  * unsupervised ED/ES: ED = argmax, ES = argmin of the log-volume curve inside the probe window
    (metrics.probe_window, the same window as the ED/ES probe) -> frame errors vs the traced frames;
  * physical EF of the curve vs the label (MAE, Pearson);
  * clock: heart rate in bpm = omega / 2pi * fps / stride * 60;
  * figure: example curves with the traced ED/ES frames.

Usage: py.cmd scripts\volume_curves.py --run sync_hr_vol_T32_s0 [--split VAL]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cardiac  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cardiac import config  # noqa: E402
from cardiac import metrics as M  # noqa: E402
from cardiac.dataset import EchoClips  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_model, make_clips  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="VAL", choices=["VAL", "TEST"])
    ap.add_argument("--T", type=int, default=None, help="default: the training T")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    dev = torch.device("cuda")
    model, targs, _ = load_model(a.run, dev)
    tb = model.temporal
    assert getattr(tb, "vol_readout", False), "run has no volumetric readout"
    T = a.T or targs["T"]
    ds = make_clips(targs, a.split, T=T, stride=targs["stride"], pad=targs["pad"], mode="traced",
                    require_tracings=True)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=a.workers, pin_memory=True)
    keep = {k: [] for k in ("ell", "omega", "phys", "pred", "ef", "frame_idx", "valid", "ed", "es", "fps", "stride")}
    for b in dl:
        v = b["video"].to(dev)
        valid = b["valid"].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(v, valid if targs["pool_valid"] else None)
        keep["ell"].append(tb.last_ell.float().cpu().numpy())
        keep["omega"].append(tb.last_omega.float().cpu().numpy() if tb.last_omega is not None
                             else np.full(len(v), np.nan, np.float32))
        keep["phys"].append(tb.aux.float().cpu().numpy() * config.EF_STD + config.EF_MEAN)
        keep["pred"].append(o["ef"].float().cpu().numpy() * config.EF_STD + config.EF_MEAN)
        for k in ("ef", "frame_idx", "valid", "ed", "es", "fps", "stride"):
            keep[k].append(b[k].numpy())
    r = {k: np.concatenate(v) for k, v in keep.items()}
    win = M.probe_window(r["frame_idx"], r["ed"], r["es"], valid=r["valid"])
    ok = (M.in_clip(r["frame_idx"], r["valid"], r["ed"], r["stride"])
          & M.in_clip(r["frame_idx"], r["valid"], r["es"], r["stride"]))
    # EF = 1 - exp(-range) is invariant to flipping the curve, so its sign is not identified by training.
    # Label-free sign: ejection is fast and filling slow, so a volume curve's frame-to-frame changes are
    # negatively skewed; flip the curve (once, globally) if their mean skewness is positive.
    skews = []
    for c, v in zip(r["ell"], r["valid"]):
        d = np.diff(c[v].astype(np.float64))
        if d.size > 3 and d.std() > 0:
            skews.append(((d - d.mean()) ** 3).mean() / d.std() ** 3)
    sign = -1.0 if np.mean(skews) > 0 else 1.0
    ell = sign * r["ell"]
    ed_hat = M.localize(ell[ok], r["frame_idx"][ok], win[ok])
    es_hat = M.localize(-ell[ok], r["frame_idx"][ok], win[ok])
    orient = {}
    for name, s in (("as_is", 1.0), ("flipped", -1.0)):
        orient[name] = {
            "ED": M.frame_error(M.localize(s * r["ell"][ok], r["frame_idx"][ok], win[ok]), r["ed"][ok]),
            "ES": M.frame_error(M.localize(-s * r["ell"][ok], r["frame_idx"][ok], win[ok]), r["es"][ok])}
    k = ok
    w_lo = np.where(win[k], r["frame_idx"][k], np.inf).min(1)
    w_hi = np.where(win[k], r["frame_idx"][k], -np.inf).max(1)
    centre = (w_lo + w_hi) / 2
    bpm = r["omega"] / (2 * np.pi) * r["fps"] / r["stride"] * 60
    out = {"run": a.run, "split": a.split, "T": T, "n_clips": int(len(r["ef"])), "n_with_ed_es": int(ok.sum()),
           "label_free_sign": sign, "mean_derivative_skewness_raw_curve": float(np.mean(skews)),
           "unsupervised_ED_frame_error": M.frame_error(ed_hat, r["ed"][ok]),
           "unsupervised_ES_frame_error": M.frame_error(es_hat, r["es"][ok]),
           "both_orientations": orient,
           "window_centre_reference": {"ED": M.frame_error(centre, r["ed"][ok]),
                                       "ES": M.frame_error(centre, r["es"][ok])},
           "phys_EF_vs_label": {"MAE": float(np.abs(r["phys"] - r["ef"]).mean()),
                                "Pearson": float(np.corrcoef(r["phys"], r["ef"])[0, 1])},
           "head_EF_vs_label_single_clip": {"MAE": float(np.abs(r["pred"] - r["ef"]).mean())},
           "clock_bpm_quantiles_5_25_50_75_95": (np.nanpercentile(bpm, [5, 25, 50, 75, 95]).round(1).tolist()
                                                 if np.isfinite(bpm).any() else None)}
    print(json.dumps(out, indent=1))
    out_dir = config.RESULTS / a.run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"volume_curves_{a.split}.json").write_text(json.dumps(out, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = np.where(ok)[0]
    pick = idx[np.linspace(0, len(idx) - 1, 8).astype(int)]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5), dpi=130)
    for ax, i in zip(axes.ravel(), pick):
        v = r["valid"][i]
        ax.plot(r["frame_idx"][i][v], ell[i][v], "-", color="#2a6f97", lw=1.4)
        for tf, c, lab in ((r["ed"][i], "#c1121f", "ED"), (r["es"][i], "#386641", "ES")):
            ax.axvline(tf, color=c, ls="--", lw=1, label=lab)
        ax.set_title(f"EF {r['ef'][i]:.0f} (curve {r['phys'][i]:.0f}, model {r['pred'][i]:.0f})", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.supxlabel("original frame index", fontsize=9)
    fig.supylabel("learned log-volume (a.u.)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"volume_curves_{a.split}.png")
    print("wrote", out_dir / f"volume_curves_{a.split}.png")


if __name__ == "__main__":
    main()
