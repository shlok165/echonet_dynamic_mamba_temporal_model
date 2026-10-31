r"""Train an EF model. Identical protocol for every temporal block; only --temporal changes.

Resumable: re-running the same command with the same --run continues from runs/<run>/last.pt.
Each epoch logs to runs/<run>/log.jsonl and saves last.pt (atomic) and best.pt (lowest VAL MAE).

Example (smoke):
  py.cmd scripts\train.py --run smoke_bigru --temporal bigru --T 16 --subset-frac 0.02 --epochs 1
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cardiac  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cardiac import config  # noqa: E402
from cardiac import metrics as M  # noqa: E402
from cardiac.dataset import EchoClips, EpochSampler, FeatClips, StackClips, pixel_stats  # noqa: E402
from cardiac.models import EFModel, FeatModel, count_params  # noqa: E402
from cardiac.panecho import PanEchoEFModel  # noqa: E402

GiB = 1024 ** 3


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--temporal", required=True, choices=["bigru", "mamba", "mamba_uni", "uvt", "identity", "sync"])
    # clip sampling (shared by all models; see cardiac/dataset.py)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--pad", default="loop", choices=["loop", "repeat", "fit"])
    ap.add_argument("--pool-valid", action="store_true", help="mean-pool only real (non-padding) frames")
    ap.add_argument("--augment", action="store_true", help="random pad-and-crop (12 px)")
    ap.add_argument("--aug", default="basic", choices=["basic", "strong"],
                    help="with --augment: basic = pad-crop; strong = + rotation/scale, intensity, time reversal")
    ap.add_argument("--subset-frac", type=float, default=None)
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--val-clips", type=int, default=1)
    # model
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--temporal-kw", default="{}",
                    help='JSON, e.g. {"expand": 1}, or key=value pairs, e.g. clock=hr,aux_weight=1.0')
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--features", default=None,
                    help="train on cached frozen per-frame features <CACHE>/feats_<name> (no encoder), e.g. panecho")
    ap.add_argument("--panecho-tune", type=int, default=0,
                    help="fine-tune the PanEcho encoder (cardiac/panecho.py) on 3-frame-stack clips with its last N "
                         "ConvNeXt blocks trainable (3 = stage 4)")
    ap.add_argument("--init-from", default=None,
                    help="warm-start the temporal block and head from runs/<run>/best.pt (a frozen-feature run "
                         "with the same temporal configuration)")
    ap.add_argument("--enc-lr", type=float, default=None, help="lr of the trainable encoder layers (default: --lr)")
    # optimisation
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--loss", default="mse", choices=["mse", "l1", "huber"],
                    help="EF regression loss on standardised EF (huber: --huber-delta, in EF SDs)")
    ap.add_argument("--huber-delta", type=float, default=0.25)
    ap.add_argument("--ema", type=float, default=0.0,
                    help="EMA decay of the weights per optimiser step (0 = off); validation, best.pt and so "
                         "evaluate.py then use the EMA weights")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-workers", type=int, default=4)
    ap.add_argument("--cap-gib", type=float, default=5.5)
    ap.add_argument("--max-train-batches", type=int, default=None, help="debug: stop epochs early")
    ap.add_argument("--max-videos", type=int, default=None, help="debug: first N videos of each split")
    a = ap.parse_args(argv)
    a.temporal_kw = json.dumps(parse_kw(a.temporal_kw), sort_keys=True)  # saved/compared as JSON
    return a


def parse_kw(s):
    """JSON object, or comma-separated key=value pairs (values via literal_eval, else strings);
    the second form survives Windows shell quoting."""
    s = s.strip()
    if not s or s.startswith("{"):
        return json.loads(s or "{}")
    import ast
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=", 1)
        try:
            out[k.strip()] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            out[k.strip()] = v.strip()
    return out


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def to_dev(batch, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, dev, pool_valid):
    model.eval()
    preds, efs, vids, phys = [], [], [], []
    for b in loader:
        b = to_dev(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b["video"], b["valid"] if pool_valid else None)
        preds.append(out["ef"].float().cpu() * config.EF_STD + config.EF_MEAN)
        efs.append(b["ef"].cpu())
        vids.append(b["vid"].cpu())
        if getattr(model.temporal, "aux", None) is not None:   # physical EF of the volume readout
            phys.append(model.temporal.aux.float().cpu() * config.EF_STD + config.EF_MEAN)
    model.train()
    p, y, v = torch.cat(preds).numpy(), torch.cat(efs).numpy(), torch.cat(vids).numpy()
    u, pm = M.video_mean(v, p)
    _, ym = M.video_mean(v, y)
    r = M.regression_metrics(ym, pm)
    res = {k: r[k] for k in ("MAE", "RMSE", "R2", "Pearson")}
    if phys:
        _, qm = M.video_mean(v, torch.cat(phys).numpy())
        res["phys_MAE"] = M.regression_metrics(ym, qm)["MAE"]
    return res


def ef_loss(pred, y, args):
    if args.loss == "l1":
        return torch.nn.functional.l1_loss(pred, y)
    if args.loss == "huber":
        return torch.nn.functional.huber_loss(pred, y, delta=args.huber_delta)
    return torch.nn.functional.mse_loss(pred, y)


def lr_lambda(total, warmup):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return f


def save_atomic(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main(argv=None):
    args = get_args(argv)
    run_dir = config.RUNS / args.run
    if (run_dir / "SKIP").exists():  # queue control: a queued run can be deferred by creating this file
        print(f"{run_dir / 'SKIP'} exists; not training {args.run}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    cap = min(args.cap_gib * GiB, free - 256 * 1024 ** 2)
    torch.cuda.set_per_process_memory_fraction(cap / total)  # no silent spill into shared RAM

    common = dict(T=args.T, stride=args.stride, pad=args.pad)
    if args.features:
        feat_dir = config.CACHE / f"feats_{args.features}"
        Clips = lambda split, **kw: FeatClips(split, feat_dir, **kw)  # noqa: E731
    elif args.panecho_tune:
        Clips = StackClips
    else:
        Clips = EchoClips
    train_ds = Clips("TRAIN", mode="random", augment=args.augment, aug=args.aug, subset_frac=args.subset_frac,
                     subset_seed=args.subset_seed, max_videos=args.max_videos, **common)
    train_ds.sample_seed = args.seed
    val_ds = Clips("VAL", mode="multi", n_clips=args.val_clips, max_videos=args.max_videos, **common)
    sampler = EpochSampler(len(train_ds), seed=args.seed)
    lkw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
               prefetch_factor=2 if args.workers > 0 else None)
    train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, drop_last=True, **lkw)
    # separate, smaller persistent worker pool for validation (re-spawning 8 Windows workers each epoch
    # cost ~40 s); speed only, results are unaffected
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.val_workers,
                        pin_memory=True, persistent_workers=args.val_workers > 0)

    if args.features:
        d_in = int(train_ds[0]["video"].shape[-1])
        model = FeatModel(args.temporal, d_in, d_model=args.d_model, n_layers=args.n_layers,
                          temporal_kw=json.loads(args.temporal_kw)).to(dev)
    elif args.panecho_tune:
        model = PanEchoEFModel(args.temporal, d_model=args.d_model, n_layers=args.n_layers,
                               temporal_kw=json.loads(args.temporal_kw), n_tune=args.panecho_tune).to(dev)
    else:
        mean, std = pixel_stats()
        model = EFModel(args.temporal, mean, std, d_model=args.d_model, n_layers=args.n_layers,
                        grad_ckpt=args.grad_ckpt, temporal_kw=json.loads(args.temporal_kw)).to(dev)
        model.encoder.to(memory_format=torch.channels_last)
    if args.init_from and not (run_dir / "last.pt").exists():
        src = torch.load(config.RUNS / args.init_from / "best.pt", map_location=dev, weights_only=False)
        for k in ("temporal", "d_model", "n_layers", "temporal_kw"):
            if src["args"][k] != getattr(args, k):
                raise ValueError(f"--init-from {args.init_from}: {k} {src['args'][k]!r} != {getattr(args, k)!r}")
        r = model.load_state_dict(src["model"], strict=False)
        if r.unexpected_keys or any(not k.startswith("tuned.") for k in r.missing_keys):
            raise ValueError(f"--init-from {args.init_from}: unexpected {r.unexpected_keys}, missing {r.missing_keys}")
        print(f"warm start from {args.init_from} (epoch {src['epoch']}, val MAE {src['val']['MAE']:.3f})")
    enc_params = list(model.tuned.parameters()) if args.panecho_tune else []
    enc_ids = {id(p) for p in enc_params}
    groups = [{"params": [p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids]}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.enc_lr or args.lr})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd)
    ema =(torch.optim.swa_utils.AveragedModel(model, use_buffers=True,
                                               multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema))
           if args.ema else None)
    for m in (ema.modules() if ema is not None else []):
        if isinstance(m, torch.nn.RNNBase):
            m.flatten_parameters()    # deepcopy loses cuDNN's single weight buffer (speed only)
    n_batches = len(train_dl) if args.max_train_batches is None else min(len(train_dl), args.max_train_batches)
    steps_per_epoch = math.ceil(n_batches / args.accum)
    total_steps = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total_steps, int(args.warmup_frac * total_steps)))

    start_epoch, best = 0, float("inf")
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        if ck["args"] != vars(args):
            diff = {k: (ck["args"].get(k), v) for k, v in vars(args).items() if ck["args"].get(k) != v}
            print("WARNING: resuming with changed args:", diff)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if ema is not None:
            ema.load_state_dict(ck["ema"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {ck['epoch']}")
    else:
        (run_dir / "args.json").write_text(json.dumps(vars(args), indent=1))
        info = {"params_total": count_params(model), "params_temporal": count_params(model.temporal),
                "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "n_train": len(train_ds), "n_val_videos": len(val_ds.names), "cap_gib": cap / GiB}
        (run_dir / "info.json").write_text(json.dumps(info, indent=1))
        print(json.dumps(info))

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        losses = []
        data_wait = 0.0
        t_data = time.perf_counter()
        for bi, b in enumerate(train_dl):
            if bi >= n_batches:
                break
            data_wait += time.perf_counter() - t_data
            b = to_dev(b, dev)
            y = (b["ef"] - config.EF_MEAN) / config.EF_STD
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(b["video"], b["valid"] if args.pool_valid else None)
            loss = ef_loss(out["ef"].float(), y, args)
            aux_w = getattr(model.temporal, "aux_weight", 0.0)
            if aux_w:  # EF-consistency of the block's physical readout (EF labels only, no ED/ES)
                loss = loss + aux_w * torch.nn.functional.mse_loss(model.temporal.aux.float(), y)
            (loss / args.accum).backward()
            if (bi + 1) % args.accum == 0 or bi + 1 == n_batches:
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                if ema is not None:
                    ema.update_parameters(model)
            losses.append(loss.item())
            t_data = time.perf_counter()
        train_time = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / GiB
        t1 = time.perf_counter()
        eval_model = ema.module if ema is not None else model
        val = evaluate(eval_model, val_dl, dev, args.pool_valid)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": sched.get_last_lr()[0],
               "train_time_s": train_time, "data_wait_s": data_wait, "val_time_s": time.perf_counter() - t1,
               "peak_gib": peak, **{f"val_{k}": v for k, v in val.items()}}
        is_best = val["MAE"] < best
        best = min(best, val["MAE"])
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "epoch": epoch, "best": best, "args": vars(args)}
        if ema is not None:
            ck["ema"] = ema.state_dict()
        save_atomic(ck, last)
        if is_best:
            save_atomic({"model": eval_model.state_dict(), "epoch": epoch, "val": val, "args": vars(args)},
                        run_dir / "best.pt")
        with open(run_dir / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
