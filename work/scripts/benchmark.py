r"""Compute benchmark: ResNet-18 + {BiGRU, Mamba} on random 112x112 uint8 clips.

For each temporal model and T in --Ts:
  * largest batch whose full training step (fwd + bwd + AdamW, bf16 autocast, channels_last)
    fits under the allocator cap (doubling, then bisection);
  * training step time and peak memory (torch.cuda.max_memory_allocated) at that batch and at --ref-batch;
  * inference latency per clip (batch 1, no_grad) and its peak memory;
  * the temporal block alone (fwd + bwd on random (B, T, 512) features), to isolate its scaling
    from the encoder, which dominates the full model.

Windows note: the NVIDIA driver can silently spill VRAM into shared system RAM (sysmem fallback),
so we enforce a hard cap with torch.cuda.set_per_process_memory_fraction; exceeding it raises OOM.

Usage:  py.cmd scripts\benchmark.py [--Ts 8 16 32 64 128] [--cap-gib 5.5] [--grad-ckpt]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cardiac  # noqa: E402,F401
import torch  # noqa: E402
from cardiac import config  # noqa: E402
from cardiac.models import EFModel, count_params, make_temporal  # noqa: E402

GiB = 1024 ** 3


def sync():
    torch.cuda.synchronize()


def train_step(model, opt, B, T, dev):
    x = torch.randint(0, 256, (B, T, 3, 112, 112), dtype=torch.uint8, device=dev)
    y = torch.randn(B, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x)
        loss = torch.nn.functional.mse_loss(out["ef"].float(), y)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def try_batch(model, opt, B, T, dev):
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):  # several steps: the first may fit and a later one fragment/OOM
            train_step(model, opt, B, T, dev)
        sync()
        return True, torch.cuda.max_memory_allocated()
    except torch.OutOfMemoryError:
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False, None


def max_batch(model, opt, T, dev, limit):
    lo, hi = 0, None
    b = 1
    while b <= limit:
        ok, _ = try_batch(model, opt, b, T, dev)
        if not ok:
            hi = b
            break
        lo = b
        b *= 2
    if hi is None:
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, _ = try_batch(model, opt, mid, T, dev)
        lo, hi = (mid, hi) if ok else (lo, mid)
    return lo


def time_train(model, opt, B, T, dev, iters):
    torch.cuda.empty_cache()
    train_step(model, opt, B, T, dev)  # warm-up (cuDNN autotune, allocator)
    sync()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        train_step(model, opt, B, T, dev)
    sync()
    return (time.perf_counter() - t0) / iters, torch.cuda.max_memory_allocated()


@torch.no_grad()
def time_infer(model, T, dev, iters):
    model.eval()
    x = torch.randint(0, 256, (1, T, 3, 112, 112), dtype=torch.uint8, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(x)
        sync()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        sync()
    model.train()
    return (time.perf_counter() - t0) / iters, torch.cuda.max_memory_allocated()


def time_temporal(kind, B, T, dev, iters, d_model, n_layers):
    m = make_temporal(kind, 512, d_model, n_layers).to(dev)
    f = torch.randn(B, T, 512, device=dev, requires_grad=True)
    for i in range(iters + 1):
        if i == 1:
            sync()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m(f).float().square().mean().backward()
    sync()
    return (time.perf_counter() - t0) / iters, torch.cuda.max_memory_allocated(), count_params(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["bigru", "mamba"])
    ap.add_argument("--Ts", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--cap-gib", type=float, default=5.5, help="hard allocator cap")
    ap.add_argument("--ref-batch", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=4096, help="search limit on B*T")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--out", default=str(config.RESULTS / "benchmark.json"))
    args = ap.parse_args()

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    cap = min(args.cap_gib * GiB, free - 256 * 1024 ** 2)
    torch.cuda.set_per_process_memory_fraction(cap / total)
    info = {"gpu": torch.cuda.get_device_name(0), "total_gib": total / GiB, "free_at_start_gib": free / GiB,
            "cap_gib": cap / GiB, "torch": torch.__version__, "grad_ckpt": args.grad_ckpt,
            "d_model": args.d_model, "n_layers": args.n_layers}
    print(json.dumps(info))
    rows = []
    for kind in args.kinds:
        for T in args.Ts:
            torch.manual_seed(0)
            model = EFModel(kind, [30.0] * 3, [50.0] * 3, d_model=args.d_model, n_layers=args.n_layers,
                            pretrained=False, grad_ckpt=args.grad_ckpt).to(dev)
            model.encoder.to(memory_format=torch.channels_last)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            r = {"kind": kind, "T": T, "params_total": count_params(model),
                 "params_temporal": count_params(model.temporal)}
            bmax = max_batch(model, opt, T, dev, max(1, args.max_frames // T))
            r["max_batch"] = bmax
            if bmax >= 1:
                for tag, B in (("ref", min(args.ref_batch, bmax)), ("max", bmax)):
                    while True:
                        try:
                            st, mem = time_train(model, opt, B, T, dev, args.iters)
                            break
                        except torch.OutOfMemoryError:
                            opt.zero_grad(set_to_none=True)
                            torch.cuda.empty_cache()
                            B = max(1, int(B * 0.9)) if B > 1 else 0
                            if B == 0:
                                raise
                    if tag == "max":
                        r["max_batch"] = B  # largest batch that also survived the timed run
                    r[f"train_{tag}_B"] = B
                    r[f"train_{tag}_step_s"] = st
                    r[f"train_{tag}_clips_per_s"] = B / st
                    r[f"train_{tag}_peak_gib"] = mem / GiB
            it, im = time_infer(model, T, dev, args.iters * 4)
            r["infer_clip_ms"] = 1000 * it
            r["infer_peak_gib"] = im / GiB
            del model, opt
            torch.cuda.empty_cache()
            tt, tm, _ = time_temporal(kind, 8, T, dev, args.iters * 4, args.d_model, args.n_layers)
            r["temporal_only_B8_fwdbwd_ms"] = 1000 * tt
            r["temporal_only_B8_peak_gib"] = tm / GiB
            torch.cuda.empty_cache()
            print(json.dumps(r), flush=True)
            rows.append(r)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"info": info, "rows": rows}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
