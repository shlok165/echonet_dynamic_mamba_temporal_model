r"""Steady-state DataLoader throughput (clips/s) vs number of workers, excluding worker start-up."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cardiac  # noqa: E402,F401
from torch.utils.data import DataLoader  # noqa: E402

from cardiac.dataset import EchoClips, EpochSampler  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--batches", type=int, default=40)
    a = ap.parse_args()
    ds = EchoClips("TRAIN", T=a.T, stride=2, pad="loop", mode="random", augment=True)
    for w in a.workers:
        dl = DataLoader(ds, batch_size=16, sampler=EpochSampler(len(ds)), num_workers=w,
                        persistent_workers=True, prefetch_factor=4)
        it = iter(dl)
        t_start = time.perf_counter()
        for _ in range(2 * w):  # wait until every worker has produced
            next(it)
        t0 = time.perf_counter()
        for _ in range(a.batches):
            next(it)
        dt = time.perf_counter() - t0
        print(f"workers={w}: startup {t0 - t_start:.1f}s, steady {a.batches * 16 / dt:.0f} clips/s", flush=True)
        del it, dl
