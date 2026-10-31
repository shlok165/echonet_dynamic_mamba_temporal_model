r"""One line per training run from runs/<run>/log.jsonl (VAL, single clip per video, as logged in training).

best = the epoch the checkpoint is selected at (lowest val MAE); last5 = mean val MAE of the last 5
epochs (less sensitive to a lucky epoch); params from info.json when present.

Usage: py.cmd scripts\summarize_runs.py run1 run2 ...   (shell globs are expanded against runs/)
"""
import fnmatch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardiac import config  # noqa: E402


def main():
    names = sorted({r.name for p in sys.argv[1:] for r in config.RUNS.iterdir() if fnmatch.fnmatch(r.name, p)})
    print(f"{'run':34s} {'ep':>5s} {'best':>7s} {'RMSE':>6s} {'R2':>6s} {'last5':>7s} {'params':>10s}")
    for n in names:
        f = config.RUNS / n / "log.jsonl"
        if not f.exists():
            print(f"{n:34s} (no log)")
            continue
        rows = [json.loads(l) for l in open(f)]
        b = min(rows, key=lambda r: r["val_MAE"])
        last5 = sum(r["val_MAE"] for r in rows[-5:]) / min(5, len(rows))
        info = config.RUNS / n / "info.json"
        params = json.loads(info.read_text()).get("params_total", "") if info.exists() else ""
        print(f"{n:34s} {len(rows):>5d} {b['val_MAE']:7.4f} {b['val_RMSE']:6.3f} {b['val_R2']:6.3f} {last5:7.4f} "
              f"{params!s:>10s}")


if __name__ == "__main__":
    main()
