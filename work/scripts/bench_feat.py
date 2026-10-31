r"""Efficiency of the frozen-feature models (temporal block + pool + head; the shared frozen encoder costs
the same for every model and is excluded): parameters, forward FLOPs, inference latency (batch 1 and
--batch), training step time (fwd + bwd + AdamW) and peak memory, for every T in --Ts.

Random (B, T, d_in) inputs, bf16 autocast as in training/evaluation. FLOPs from
torch.utils.flop_counter for matmul-type ops; cuDNN GRU kernels are not counted there, so the GRU's
FLOPs are added analytically (6 * (d_in + h + 1) * h * 2 per step per direction per layer, x2 for
multiply-add). Run on an otherwise idle GPU.

Usage: py.cmd scripts\bench_feat.py [--Ts 8 16 32 64 128 256] [--batch 16]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cardiac  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from cardiac import config  # noqa: E402
from cardiac.models import FeatModel, count_params  # noqa: E402

GiB = 1024 ** 3
VOL = dict(vol_readout=True, aux_weight=1.0, conv_pad="replicate", vol_trim=2)
CONFIGS = {
    "identity": ("identity", 2, {}),
    "bigru": ("bigru", 3, {}),
    "mamba": ("mamba", 2, {"expand": 1}),
    "uvt": ("uvt", 2, {}),
    "sync": ("sync", 2, {"clock": "hr", **VOL}),
    "sync_fixed": ("sync", 2, {"clock": "fixed", "conv_pad": "replicate"}),
    "sync_hr": ("sync", 2, {"clock": "hr", "conv_pad": "replicate"}),
    "sync_harm_range": ("sync", 2, {"clock": "hr", "conv_pad": "replicate", "harm_k": 16, "range_k": 16}),
    "bigru_harm_range": ("bigru", 3, {"harm_k": 16, "range_k": 16}),
    "panecho_tf": ("panecho_tf", 4, {}),
}


class PanEchoTF(torch.nn.Module):
    """PanEcho's temporal model on frame features (scripts/panecho_probe.py --stage tune): 4 post-norm
    Transformer layers (d 768, 8 heads, FFN 768), sinusoidal positions, mean over valid frames, EF head.
    Random weights (timing only)."""

    def __init__(self, max_T=1024):
        super().__init__()
        layer = torch.nn.TransformerEncoderLayer(768, 8, 768, 0.0, "relu", batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
        self.EF_head = torch.nn.Sequential(torch.nn.Dropout(0.25), torch.nn.Linear(768, 1))
        self.register_buffer("pe", torch.randn(1, max_T, 768))
        self.temporal = self.transformer

    def forward(self, x, valid=None):
        z = self.transformer(x.float() + self.pe[:, :x.shape[1]])
        return {"ef": self.EF_head(z.mean(1)).squeeze(-1)}


def gru_flops(model, B, T):
    f = 0
    for m in model.modules():
        if isinstance(m, torch.nn.GRU):
            dirs = 2 if m.bidirectional else 1
            for layer in range(m.num_layers):
                d_in = m.input_size if layer == 0 else m.hidden_size * dirs
                f += 2 * 3 * (d_in + m.hidden_size) * m.hidden_size * T * B * dirs
    return f


def timed(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def graph_ms(model, x, iters):
    """Inference latency with the whole forward captured as one CUDA graph (removes Python and
    kernel-launch overhead, identically for every model); None if the model cannot be captured."""
    try:
        static_x = x.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            for _ in range(3):
                model(static_x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            with torch.cuda.graph(g):
                model(static_x)
        return 1000 * timed(g.replay, iters)
    except Exception as e:  # noqa: BLE001 - report, do not abort the benchmark
        print("  graph capture failed:", type(e).__name__, str(e)[:120], flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ts", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--d-in", type=int, default=768)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--kinds", nargs="+", default=list(CONFIGS))
    ap.add_argument("--out", default=str(config.RESULTS / "bench_feat.json"))
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    res = {"batch": a.batch, "d_in": a.d_in, "gpu": torch.cuda.get_device_name(), "models": {}}
    for name in a.kinds:
        kind, nl, kw = CONFIGS[name]
        torch.manual_seed(0)
        model = (PanEchoTF() if kind == "panecho_tf" else FeatModel(kind, a.d_in, 256, nl, kw)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        rec = {"params_total": count_params(model), "params_temporal": count_params(model.temporal), "by_T": {}}
        for T in a.Ts:
            r = {}
            for B, key in ((1, "b1"), (a.batch, f"b{a.batch}")):
                x = torch.randn(B, T, a.d_in, device=dev).half()
                model.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    torch.cuda.reset_peak_memory_stats()
                    r[f"infer_ms_per_clip_{key}"] = 1000 * timed(lambda: model(x), a.iters) / B
                    r[f"infer_peak_gib_{key}"] = torch.cuda.max_memory_allocated() / GiB
                    if B == 1:
                        with FlopCounterMode(display=False) as fc:
                            model(x)
                        r["fwd_gflops_per_clip"] = (fc.get_total_flops() + gru_flops(model, 1, T)) / 1e9
                ms = graph_ms(model, x, a.iters)
                r[f"infer_graph_ms_per_clip_{key}"] = ms / B if ms is not None else None
            x = torch.randn(a.batch, T, a.d_in, device=dev).half()
            y = torch.randn(a.batch, device=dev)
            model.train()

            def step():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = torch.nn.functional.mse_loss(model(x)["ef"].float(), y)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)

            torch.cuda.reset_peak_memory_stats()
            r[f"train_ms_per_step_b{a.batch}"] = 1000 * timed(step, a.iters // 2)
            r[f"train_peak_gib_b{a.batch}"] = torch.cuda.max_memory_allocated() / GiB
            rec["by_T"][T] = r
            print(name, T, {k: (round(v, 4) if v is not None else None) for k, v in r.items()}, flush=True)
        res["models"][name] = rec
        del model, opt
        torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
