r"""Paired significance of EF differences between models, on per-video predictions.

A model = one run or several seeds (comma-separated runs: their per-video errors are averaged, so the
test is over videos, not seeds); run@T reads that run's predictions at clip length T instead of --T. Reads results/<run>/<split>_preds_T<T>.npz written by evaluate.py.
Reports per model: MAE (mean over seeds of each seed's MAE, and its SD); per pair: MAE difference with
a 95% paired bootstrap CI over videos, the bootstrap two-sided p-value, and the Wilcoxon signed-rank p
on per-video absolute errors.

Usage: py.cmd scripts\stats.py --models bigru_T32_s0,bigru_T32_s1 sync_T32_s0,sync_T32_s1 [--split test] [--T 32]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from scipy import stats as st  # noqa: E402

from cardiac import config  # noqa: E402
from cardiac import metrics as M  # noqa: E402


def per_video(run, split, T):
    if "@" in run:
        run, T = run.split("@")
    d = np.load(config.RESULTS / run / f"{split}_preds_T{T}.npz", allow_pickle=True)
    names = np.asarray(d["names"])
    u, pm = M.video_mean(d["vid"], d["pred"])
    _, ym = M.video_mean(d["vid"], d["ef"])
    return dict(zip(names[u], pm)), dict(zip(names[u], ym))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="each: run[,run,...] (seeds of one model)")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    models = {m: m.split(",") for m in a.models}
    preds, truth, seed_mae = {}, None, {}
    for m, runs in models.items():
        ps = []
        for r in runs:
            p, y = per_video(r, a.split, a.T)
            truth = y if truth is None else truth
            ps.append(p)
        common = sorted(set.intersection(*(set(p) for p in ps)))
        preds[m] = {v: [p[v] for p in ps] for v in common}
        seed_mae[m] = [float(np.mean([abs(p[v] - truth[v]) for v in common])) for p in ps]
    vids = sorted(set.intersection(*(set(p) for p in preds.values())))
    y = np.array([truth[v] for v in vids])
    err = {m: np.array([np.mean([abs(pp - truth[v]) for pp in preds[m][v]]) for v in vids]) for m in models}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(vids), size=(a.n_boot, len(vids)))
    out = {"split": a.split, "T": a.T, "n_videos": len(vids), "models": {}, "pairs": {}}
    for m in models:
        out["models"][m] = {"seeds": len(models[m]), "MAE_mean_over_seeds": float(np.mean(seed_mae[m])),
                            "MAE_sd_over_seeds": float(np.std(seed_mae[m], ddof=1)) if len(seed_mae[m]) > 1 else None,
                            "MAE_seed_averaged_errors": float(err[m].mean())}
    names = list(models)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = err[names[i]] - err[names[j]]                       # < 0: first model better
            boot = d[idx].mean(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            p_boot = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
            p_w = float(st.wilcoxon(err[names[i]], err[names[j]]).pvalue)
            out["pairs"][f"{names[i]} vs {names[j]}"] = {"MAE_diff": float(d.mean()), "ci95": [float(lo), float(hi)],
                                                         "p_bootstrap": p_boot, "p_wilcoxon": p_w}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
