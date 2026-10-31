r"""Shared learning rate from the frozen-feature sweep (run_feat_lr.ps1), decided on VAL only.

Rule (fixed before looking): for every lr, the mean over models of each run's best VAL MAE (the epoch
the checkpoint is selected at); the lr with the lowest mean wins. Prints a table, writes
results/lr_sweep_<features>.json, and prints the chosen lr on the last line.

Usage: py.cmd scripts\pick_lr.py [--prefix pe] [--models bigru sync] [--lrs 1e-4 3e-4 1e-3]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardiac import config  # noqa: E402


def best(run):
    rows = [json.loads(l) for l in open(config.RUNS / run / "log.jsonl")]
    b = min(rows, key=lambda r: r["val_MAE"])
    return b["val_MAE"], b["epoch"], len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="pe")
    ap.add_argument("--models", nargs="+", default=["bigru", "sync"])
    ap.add_argument("--lrs", nargs="+", default=["1e-4", "3e-4", "1e-3"])
    a = ap.parse_args()
    table = {lr: {m: best(f"{a.prefix}_{m}_lr{lr}_s0") for m in a.models} for lr in a.lrs}
    mean = {lr: sum(v[0] for v in t.values()) / len(t) for lr, t in table.items()}
    for lr in a.lrs:
        print(lr, "  ".join(f"{m} {table[lr][m][0]:.4f} (ep {table[lr][m][1]}/{table[lr][m][2]})" for m in a.models),
              f"| mean {mean[lr]:.4f}")
    pick = min(mean, key=mean.get)
    (config.RESULTS / f"lr_sweep_{a.prefix}.json").write_text(json.dumps(
        {"rule": "argmin over lr of mean best VAL MAE", "table": table, "mean": mean, "chosen": pick}, indent=1))
    print(pick)


if __name__ == "__main__":
    main()
