r"""Single comparison table from several runs' results/<run>/eval.json (markdown + CSV).

Usage: py.cmd scripts\compare.py --runs bigru_T32_s0 mamba_T32_s0 [--T 32] [--split VAL] [--out ...]
       seeds pooled per column: --runs BiGRU=bigru_T32_s0,bigru_T32_s1 Sync=sync_T32_s0,sync_T32_s1
Arrows: (low) lower is better, (high) higher is better. Best value per row is marked with *.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardiac import config  # noqa: E402

ROWS = [  # (section, label, path into by_T[T] or top-level, better)
    ("EF prediction", "MAE", ("ef", "MAE"), "low"),
    ("EF prediction", "MSE", ("ef", "MSE"), "low"),
    ("EF prediction", "RMSE", ("ef", "RMSE"), "low"),
    ("EF prediction", "R2", ("ef", "R2"), "high"),
    ("EF prediction", "MAPE (%)", ("ef", "MAPE"), "low"),
    ("EF prediction", "Pearson r", ("ef", "Pearson"), "high"),
    ("EF prediction", "Spearman rho", ("ef", "Spearman"), "high"),
    ("EF prediction", "CCC (Lin)", ("ef", "CCC"), "high"),
    ("EF prediction", "within +-5 EF", ("ef", "within5"), "high"),
    ("EF prediction", "AUROC EF<50", ("ef", "AUROC_lt50"), "high"),
    ("EF prediction", "AUROC EF<40", ("ef", "AUROC_lt40"), "high"),
    ("Bland-Altman", "bias", ("ef", "bland_altman", "bias"), "abs"),
    ("Bland-Altman", "LoA width (2x1.96 SD)", ("ef", "bland_altman", "sd"), "low"),
    ("Bland-Altman", "proportional-bias slope", ("ef", "bland_altman", "prop_bias_slope"), "abs"),
    ("Frame-order test", "MAE, shuffled frames", ("shuffled_ef", "MAE"), None),
    ("Frame-order test", "MAE increase when shuffled", ("shuffle_change", "MAE"), "high"),
    ("Temporal consistency", "cos(Z_t, Z_t+1)", ("cosine_smoothness",), "high"),
    ("Temporal consistency", "EF SD across clips (all videos)", ("ef_std_across_clips", "mean_std"), "low"),
    ("Temporal consistency", "EF SD across clips (distinct clips)", ("ef_std_across_distinct_clips", "mean_std"), "low"),
    ("Periodicity", "autocorr. peak of Z_t", ("periodicity", "autocorr_peak_mean"), "high"),
    ("Efficiency", "inference ms / video (model only)", ("infer_model_ms_per_video",), "low"),
    ("Efficiency", "inference peak GiB", ("infer_peak_gib",), "low"),
]


def get(d, path):
    for k in path:
        if d is None or k not in d:
            return None
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="table columns; a column can pool seeds: name=run_s0,run_s1,run_s2")
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--split", default="TEST", choices=["TEST", "VAL"])
    ap.add_argument("--out", default=str(config.RESULTS / "comparison"))
    a = ap.parse_args()
    cols = {}
    for spec in a.runs:
        name, _, runs = spec.rpartition("=")
        runs = runs.split(",")
        cols[name or runs[0]] = runs
    suffix = "" if a.split == "TEST" else f"_{a.split}"
    ev = {r: json.loads((config.RESULTS / r / f"eval{suffix}.json").read_text()) for rs in cols.values() for r in rs}

    def per_col(fn):
        """fn(run) -> number or None; per column (mean, sd or None, n seeds) or None."""
        out = []
        for rs in cols.values():
            v = [x for x in (fn(r) for r in rs) if x is not None]
            out.append(None if not v else (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else None, len(v)))
        return out

    table = []
    for sec, label, path, better in ROWS:
        scale = 2 * 1.96 if label == "LoA width (2x1.96 SD)" else 1.0
        vals = per_col(lambda r: (lambda v: None if v is None else scale * v)(get(ev[r]["by_T"].get(str(a.T)), path)))
        table.append((sec, label, vals, better))
    # probe (at the training T) and size / training cost
    for feat, name in (("Z", "Z_t"), ("F", "F_t (encoder, reference)")):
        for ph in ("ED", "ES"):
            vals = per_col(lambda r: get(ev[r].get("probe", {}).get(str(a.T)), (feat, "test", ph, "mean")))
            table.append(("ED/ES probe (frozen)", f"{ph} error, frames, on {name}", vals, "low" if feat == "Z" else None))
    table.append(("Efficiency", "parameters, temporal block", per_col(lambda r: ev[r]["params_temporal"]), None))
    table.append(("Efficiency", "parameters, total", per_col(lambda r: ev[r]["params_total"]), None))

    def train_h(r):
        log = config.RUNS / r / "log.jsonl"
        L = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
        return sum(x["train_time_s"] for x in L) / 3600 if L else None
    table.append(("Efficiency", "training time, h (train loop only)", per_col(train_h), "low"))
    # clip-length sweep
    Ts = sorted({int(t) for r in ev for t in ev[r]["by_T"]})
    for metric in ("MAE", "R2"):
        for T in Ts:
            table.append(("Clip-length sweep", f"{metric} at T={T}",
                          per_col(lambda r: get(ev[r]["by_T"].get(str(T)), ("ef", metric))),
                          "low" if metric == "MAE" else "high"))

    def fmt1(v):
        if isinstance(v, int) or (isinstance(v, float) and v.is_integer() and abs(v) > 1000):
            return f"{int(v):,}"
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"

    def fmt(c):
        if c is None:
            return "-"
        m, sd, n = c
        return fmt1(m) + ("" if sd is None else f" ± {fmt1(sd)}")

    seeds = {name: len(rs) for name, rs in cols.items()}
    lines = [f"{a.split} set, trained at T={a.T} unless stated; mean ± SD over seeds {seeds}. * = best mean.\n",
             "| Section | Metric | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for sec, label, vals, better in table:
        cells = [fmt(c) for c in vals]
        means = [None if c is None else c[0] for c in vals]
        nums = [m for m in means if m is not None]
        if better and len(nums) == len(means) and len(set(nums)) > 1:
            key = {"low": min, "high": max}.get(better)
            target = key(means) if key else min(means, key=abs)
            cells = [c + (" *" if m == target else "") for c, m in zip(cells, means)]
        lines.append(f"| {sec} | {label} | " + " | ".join(cells) + " |")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "metric"] + [f"{c}_{k}" for c in cols for k in ("mean", "sd", "n")])
        for sec, label, vals, _ in table:
            w.writerow([sec, label] + [x for c in vals for x in ((None, None, None) if c is None else c)])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
