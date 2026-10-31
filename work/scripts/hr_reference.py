r"""Per-video heart-rate reference (label-free, model-independent) for heart-rate-stratified analyses.

For every video: the dominant period of the WHOLE video (all decoded frames at the native frame rate,
usually 3-5 beats) from the plain periodogram of its cached frozen per-frame features
(cardiac.sync.cardiac_clock, the same estimator the sync block uses on clips), searched between
0.3 s and 2.0 s (200-30 bpm), converted to bpm with the FileList frame rate. No labels are used.
Check: on --n-check VAL videos the same estimator on 28x28 grey-level pixels of all frames
(scripts/check_clock.py's reference) is compared with the feature-based period.

Output: results/hr_reference.csv (FileName, Split, FPS, n_frames, period_frames, hr_bpm, peakiness, EF)
and results/hr_reference.json (quantiles per split, pixel-vs-feature agreement).

Usage: py.cmd scripts\hr_reference.py [--features panecho] [--n-check 300]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cardiac  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cardiac import config  # noqa: E402
from cardiac.dataset import load_frames  # noqa: E402
from cardiac.sync import cardiac_clock  # noqa: E402

MIN_S, MAX_S = 0.3, 2.0


def period(x, fps, n_grid=240):
    """x: (T, C) array -> (period in frames, peakiness); NaN if the video is shorter than 1.5 x MAX_S."""
    T = x.shape[0]
    lo, hi = MIN_S * fps, min(MAX_S * fps, T / 1.5)
    if hi <= lo * 1.2:
        return np.nan, np.nan
    om, pk = cardiac_clock(torch.as_tensor(np.asarray(x, np.float32))[None], min_period=lo, max_period=hi,
                           n_grid=n_grid)
    return float(2 * np.pi / om[0]), float(pk[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="panecho")
    ap.add_argument("--n-check", type=int, default=300)
    a = ap.parse_args()
    torch.set_num_threads(4)
    feat_dir = config.CACHE / f"feats_{a.features}"
    idx = pd.read_csv(feat_dir / "index.csv").set_index("FileName")
    fl = pd.read_csv(config.FILELIST).set_index("FileName")
    fl = fl.loc[fl.index.intersection(idx.index)]
    rows = []
    for i, (name, r) in enumerate(fl.iterrows()):
        x = np.load(feat_dir / f"{name}.npy")
        p, pk = period(x, float(r.FPS))
        rows.append(dict(FileName=name, Split=r.Split, FPS=float(r.FPS), n_frames=int(x.shape[0]),
                         period_frames=p, hr_bpm=60.0 * float(r.FPS) / p if np.isfinite(p) else np.nan,
                         peakiness=pk, EF=float(r.EF)))
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{len(fl)}", flush=True)
    df = pd.DataFrame(rows)
    out = config.RESULTS / "hr_reference.csv"
    df.to_csv(out, index=False)
    rep = {"features": a.features, "search_s": [MIN_S, MAX_S], "n_videos": len(df),
           "n_nan": int(df.hr_bpm.isna().sum()), "hr_bpm_quantiles_5_25_50_75_95": {}}
    for s, g in df.groupby("Split"):
        rep["hr_bpm_quantiles_5_25_50_75_95"][s] = np.nanpercentile(g.hr_bpm, [5, 25, 50, 75, 95]).round(1).tolist()

    # agreement with the pixel periodogram on the first n_check VAL videos (whole videos, all frames)
    val = df[df.Split == "VAL"].head(a.n_check)
    pix, feat = [], []
    for _, r in val.iterrows():
        fr = load_frames(config.VIDEO_DIR / f"{r.FileName}.avi", np.arange(int(r.n_frames)))
        g = torch.from_numpy(np.ascontiguousarray(fr)).float().mean(-1)[:, None] / 255.0     # (T, 1, H, W)
        g = F.adaptive_avg_pool2d(g, 28).flatten(1).numpy()
        pp, _ = period(g, r.FPS)
        pix.append(pp)
        feat.append(r.period_frames)
    pix, feat = np.array(pix), np.array(feat)
    ok = np.isfinite(pix) & np.isfinite(feat)
    ratio = feat[ok] / pix[ok]
    rep["pixel_check"] = {"n": int(ok.sum()), "median_abs_rel_diff": float(np.median(np.abs(ratio - 1))),
                          "within_10pct": float(np.mean(np.abs(ratio - 1) < 0.1)),
                          "within_20pct": float(np.mean(np.abs(ratio - 1) < 0.2)),
                          "octave_(ratio<0.6_or_>1.7)": float(np.mean((ratio < 0.6) | (ratio > 1.7)))}
    (config.RESULTS / "hr_reference.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
