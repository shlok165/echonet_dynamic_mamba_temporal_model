r"""EF error stratified by heart rate (label-free whole-video reference, scripts/hr_reference.py).

A model = one run or several seeds (comma-separated; per-video absolute errors averaged over seeds, as
in stats.py). Bins: clinical (<60, 60-100, >100 bpm) and quintiles of the split's heart rate. For every
model the MAE per bin; for every model against --ref the per-bin MAE difference with a 95% paired
bootstrap CI over the bin's videos.

Usage: py.cmd scripts\hr_strata.py --models a_s0,a_s1 b_s0,b_s1 --ref b_s0,b_s1 [--split val] [--T 32]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cardiac import config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import per_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--ref", default=None, help="model (as in --models) the others are compared with")
    ap.add_argument("--split", default="val", choices=["test", "val"])
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    hr = pd.read_csv(config.RESULTS / "hr_reference.csv").set_index("FileName").hr_bpm
    err, truth = {}, None
    for m in a.models:
        ps = []
        for r in m.split(","):
            p, y = per_video(r, a.split, a.T)
            truth = y if truth is None else truth
            ps.append(p)
        common = set.intersection(*(set(p) for p in ps))
        err[m] = {v: float(np.mean([abs(p[v] - truth[v]) for p in ps])) for v in common}
    vids = sorted(set.intersection(*(set(e) for e in err.values())) & set(hr.dropna().index))
    h = hr.loc[vids].to_numpy()
    E = {m: np.array([err[m][v] for v in vids]) for m in a.models}
    q = np.percentile(h, [20, 40, 60, 80])
    schemes = {"clinical": [("<60", h < 60), ("60-100", (h >= 60) & (h <= 100)), (">100", h > 100)],
               "quintile": [(f"Q{i + 1} [{lo:.0f}-{hi:.0f}]", (h >= lo) & (h < hi) if i < 4 else (h >= lo))
                            for i, (lo, hi) in enumerate(zip([-np.inf, *q], [*q, np.inf]))]}
    rng = np.random.default_rng(0)
    out = {"split": a.split, "T": a.T, "n_videos": len(vids), "ref": a.ref, "bins": {}}
    for scheme, bins in schemes.items():
        out["bins"][scheme] = {}
        for label, mask in bins:
            rec = {"n": int(mask.sum()), "hr_median": float(np.median(h[mask])) if mask.any() else None,
                   "MAE": {m: float(E[m][mask].mean()) for m in a.models}}
            if a.ref and mask.sum() > 1:
                idx = rng.integers(0, mask.sum(), size=(a.n_boot, mask.sum()))
                rec["diff_vs_ref"] = {}
                for m in a.models:
                    if m == a.ref:
                        continue
                    d = E[m][mask] - E[a.ref][mask]
                    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
                    rec["diff_vs_ref"][m] = {"diff": float(d.mean()), "ci95": [float(lo), float(hi)]}
            out["bins"][scheme][label] = rec
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
