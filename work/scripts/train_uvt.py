r"""Train UVT (Reynaud et al., MICCAI 2021) with the original recipe (Network/train.py + main.py of
https://github.com/HReynaud/UVT): AdamW lr 1e-5, StepLR(step=3, gamma=0.1), batch 2, 128-frame clips,
loss = SD loss + 5 * MSE(EF/100), DTmode 'repeat' (mirror) or 'sample' (random).

NOTE on size: the paper configuration (emb 1024, 16 layers, FFN 8192, 16 heads) has ~336M parameters
in the BERT encoder; AdamW training needs ~16 bytes/parameter (~5.4 GB) before activations, which does
not fit the ~4.8 GB usable on the RTX 3050 6GB. Use --preset small (or set the sizes) to train here;
the paper preset is kept for completeness and for machines with more memory.

NOTE on ED/ES supervision: UVT is trained WITH ED/ES labels (its SD branch). It is therefore an external
reference model, not one of the protocol-matched EF-only models, and its Z_t must not be used for the
frozen ED/ES probe comparison. --rm-branch SD trains the EF branch only.

Resumable: re-run the same command to continue from runs/<run>/last.pt.
Example: py.cmd scripts\train_uvt.py --run uvt_small_M_reg --preset small --mode repeat
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cardiac  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cardiac import config  # noqa: E402
from cardiac.uvt import PAPER_CONFIG, count_params, get_uvt, uvt_loss  # noqa: E402
from cardiac.uvt_data import UVTEchoSet  # noqa: E402

GiB = 1024 ** 3
PRESETS = {
    "paper": PAPER_CONFIG,
    "small": dict(emb_dim=256, num_hidden_layers=4, intermediate_size=1024, attention_heads=4),
}


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--preset", choices=list(PRESETS), default="small")
    ap.add_argument("--emb-dim", type=int)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--ffn", type=int)
    ap.add_argument("--heads", type=int)
    ap.add_argument("--sdmode", choices=["reg", "cla"], default="reg")
    ap.add_argument("--mode", choices=["repeat", "sample"], default="repeat", help="'repeat' = M, 'sample' = R")
    ap.add_argument("--rm-branch", choices=["SD", "EF"], default=None)
    ap.add_argument("--pos-emb", action="store_true", help="add learned positions (NOT in the original)")
    ap.add_argument("--label-order", choices=["verified", "csv"], default="verified")
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-step", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="disable encoder gradient checkpointing (needed to fit batch 2 x 128 frames in ~4.8 GB)")
    ap.add_argument("--cap-gib", type=float, default=5.5)
    ap.add_argument("--max-batches", type=int, default=None, help="debug: limit batches per epoch")
    return ap.parse_args(argv)


def model_cfg(a):
    cfg = dict(PRESETS[a.preset])
    for k, v in (("emb_dim", a.emb_dim), ("num_hidden_layers", a.layers), ("intermediate_size", a.ffn),
                 ("attention_heads", a.heads)):
        if v is not None:
            cfg[k] = v
    return cfg


def run_epoch(model, dl, train, opt, dev, a, max_batches=None):
    model.train(train)
    tot, n = 0.0, 0
    preds, efs = [], []
    ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp == "bf16")
    with torch.set_grad_enabled(train):
        for bi, b in enumerate(dl):
            if max_batches is not None and bi >= max_batches:
                break
            v = b["video"].to(dev, non_blocking=True)
            y, att, ef = b["sd_label"].to(dev), b["attention"].to(dev), b["ef"].to(dev)
            with ctx:
                out = model(v)
            loss, lsd, lef = uvt_loss(out, y, ef, a.sdmode, att if a.mode == "sample" else None)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            tot += loss.item()
            n += 1
            if out["ef"] is not None:
                preds.append(out["ef"].float().detach().cpu() * 100)
                efs.append(ef.cpu())
    res = {"loss": tot / max(n, 1)}
    if preds:
        p, y = torch.cat(preds).numpy(), torch.cat(efs).numpy()
        res["ef_MAE_clip"] = float(np.abs(p - y).mean())
    return res


def main(argv=None):
    a = get_args(argv)
    run_dir = config.RUNS / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(a.cap_gib * GiB, free - 256 * 1024 ** 2) / total)

    cfg = model_cfg(a)
    model = get_uvt(SDmode=a.sdmode, rm_branch=a.rm_branch, pos_emb=a.pos_emb,
                    grad_ckpt=not a.no_grad_ckpt, **cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, a.lr_step)  # gamma 0.1, as the original
    kw = dict(mode=a.mode, SDmode=a.sdmode, fixed_length=a.length, max_length=a.length,
              label_order=a.label_order, seed=a.seed)
    tr, va = UVTEchoSet("TRAIN", **kw), UVTEchoSet("VAL", **kw)
    lk = dict(batch_size=a.batch, num_workers=a.workers, pin_memory=True, drop_last=True,
              persistent_workers=a.workers > 0)
    tr_dl = DataLoader(tr, shuffle=True, generator=torch.Generator().manual_seed(a.seed), **lk)
    va_dl = DataLoader(va, shuffle=False, **lk)

    start, best = 0, math.inf
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best = ck["epoch"] + 1, ck["best"]
        print("resumed from epoch", ck["epoch"])
    else:
        info = {"args": vars(a), "model_cfg": cfg, "params": count_params(model),
                "params_bert": count_params(model.T), "params_ae": count_params(model.AE),
                "n_train": len(tr), "n_val": len(va)}
        (run_dir / "info.json").write_text(json.dumps(info, indent=1))
        print(json.dumps(info))

    for epoch in range(start, a.epochs):
        tr.set_epoch(epoch)
        va.set_epoch(0)  # fixed validation windows
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        r_tr = run_epoch(model, tr_dl, True, opt, dev, a, a.max_batches)
        t1 = time.perf_counter()
        r_va = run_epoch(model, va_dl, False, None, dev, a, a.max_batches)
        sched.step()
        rec = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_time_s": t1 - t0,
               "val_time_s": time.perf_counter() - t1, "peak_gib": torch.cuda.max_memory_allocated() / GiB,
               **{f"train_{k}": v for k, v in r_tr.items()}, **{f"val_{k}": v for k, v in r_va.items()}}
        is_best = r_va["loss"] < best  # the original keeps the lowest validation loss
        best = min(best, r_va["loss"])
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
              "best": best, "args": vars(a), "model_cfg": cfg}
        torch.save(ck, str(last) + ".tmp")
        os.replace(str(last) + ".tmp", last)
        if is_best:
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(a), "model_cfg": cfg},
                       run_dir / "best.pt")
        with open(run_dir / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
