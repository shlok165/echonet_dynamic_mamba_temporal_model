r"""PanEcho's own video model as a trained baseline on EchoNet-Dynamic (the zero-shot model is
scripts/panecho_zeroshot.py, whose PanEchoEF and clip loader are reused here).

--stage extract: frozen PanEcho image-encoder outputs for --n-clips evenly spaced clips of 16 consecutive
  frames per video with the released packing (quirk), all splits ->
  <CACHE>/panecho_clips/{split}.npz: names, ef, e (n_videos, n_clips, 16, 768) float16. No labels used.
--stage ridge: linear probe = ridge regression from the frozen PanEcho video embedding (pretrained temporal
  Transformer, mean over time) to EF; TRAIN clips as samples, lambda chosen on VAL MAE (4-clip average).
--stage tune: PanEcho's temporal Transformer + EF head fine-tuned from the released weights on the frozen
  frame features (encoder frozen); one random cached clip per TRAIN video per epoch, AdamW, cosine with
  warm-up, weight EMA, best epoch by VAL MAE (4-clip average). --lr chooses among values on VAL.
  --clips s3: the same pretrained Transformer + head on our clip protocol instead (3-frame-stack cache,
  T 32, stride 2, loop pad, mean over real frames; PanEcho's sinusoidal positions extended to 32).
Predictions in evaluate.py's format: results/panecho_<stage>[_s3]_.../{split}_preds_T<16|32>.npz (+ eval json).
TEST predictions are written only with --split TEST.

Usage: py.cmd scripts\panecho_probe.py --stage extract
       py.cmd scripts\panecho_probe.py --stage ridge
       py.cmd scripts\panecho_probe.py --stage tune --lr 1e-4 [--seed 0] [--split TEST]
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cardiac  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cardiac import config  # noqa: E402
from cardiac import metrics as M  # noqa: E402
from panecho_zeroshot import Clips, PanEchoEF  # noqa: E402

OUT = config.CACHE / "panecho_clips"


@torch.no_grad()
def extract(a, dev):
    OUT.mkdir(parents=True, exist_ok=True)
    model = PanEchoEF("quirk").to(dev).eval()
    for split in ("TRAIN", "VAL", "TEST"):
        f = OUT / f"{split}.npz"
        if f.exists():
            continue
        ds = Clips(split, a.n_clips, "quirk")
        dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=a.workers, pin_memory=True)
        e = np.zeros((len(ds.names), a.n_clips, 16, 768), np.float16)
        t0 = time.perf_counter()
        for i, b in enumerate(dl):
            x = b["video"].to(dev, non_blocking=True)
            n, l = x.shape[:2]
            x = x.permute(0, 2, 1, 3, 4).reshape(n * l, 3, *x.shape[3:])          # PanEcho's packing
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.encoder(x.contiguous(memory_format=torch.channels_last))
            e[b["vid"].numpy(), b["clip"].numpy()] = h.float().view(n, l, 768).cpu().numpy()
            if (i + 1) % 200 == 0:
                print(f"{split} {(i + 1) * 8}/{len(ds)} clips, {(i + 1) * 8 / (time.perf_counter() - t0):.1f} clips/s",
                      flush=True)
        np.savez(f, names=ds.names, ef=ds.ds.ef, e=e)
        print("wrote", f, e.shape, flush=True)


def load(split):
    d = np.load(OUT / f"{split}.npz")
    return d["names"], d["ef"].astype(np.float32), d["e"]


def video_embed(model, e, dev, bs=256):
    """Frozen pretrained temporal Transformer: (V, C, 16, 768) -> (V, C, 768) clip embeddings."""
    V, C = e.shape[:2]
    flat = torch.from_numpy(e.reshape(V * C, 16, 768))
    out = []
    with torch.no_grad():
        for i in range(0, len(flat), bs):
            x = flat[i:i + bs].to(dev).float() + model.pe
            out.append(model.transformer(x).mean(1).cpu())
    return torch.cat(out).view(V, C, 768).numpy()


def write(name, split, names, ef, pred, extra, T=16):
    """pred: (V, C) per-clip EF in %; saved like evaluate.py (one row per clip)."""
    V, C = pred.shape
    out_dir = config.RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    vid = np.repeat(np.arange(V), C)
    np.savez_compressed(out_dir / f"{split.lower()}_preds_T{T}.npz", names=names, vid=vid,
                        clip=np.tile(np.arange(C), V), pred=pred.ravel(), ef=np.repeat(ef, C))
    pm = pred.mean(1)
    res = M.regression_metrics(ef, pm)
    res.update(M.threshold_metrics(ef, pm, 40))
    res.update(M.threshold_metrics(ef, pm, 50))
    (out_dir / f"eval_{split}.json").write_text(json.dumps({"split": split, "n_clips": C, **extra, "ef": res},
                                                           indent=1, default=float))
    print(name, split, {k: round(float(res[k]), 4) for k in ("MAE", "RMSE", "R2")}, flush=True)


def ridge(a, dev):
    model = PanEchoEF("quirk").to(dev).eval()
    splits = ["TRAIN", "VAL"] + (["TEST"] if a.split == "TEST" else [])
    data = {s: load(s) for s in splits}
    Z = {s: video_embed(model, data[s][2], dev) for s in splits}
    with torch.no_grad():   # sanity: the released EF head on the cache = scripts/panecho_zeroshot.py
        zs = model.EF_head(torch.from_numpy(Z["VAL"]).to(dev)).squeeze(-1).mean(1).cpu().numpy()
    print("zero-shot VAL MAE from the cache:", round(float(np.abs(zs - data["VAL"][1]).mean()), 4))
    Xtr =Z["TRAIN"].reshape(-1, 768).astype(np.float64)
    ytr = np.repeat(data["TRAIN"][1], Z["TRAIN"].shape[1]).astype(np.float64)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xs = (Xtr - mu) / sd
    G, b = Xs.T @ Xs, Xs.T @ (ytr - ytr.mean())

    def predict(lam, s):
        w = np.linalg.solve(G + lam * np.eye(768), b)
        return ((Z[s].astype(np.float64) - mu) / sd) @ w + ytr.mean()

    lams = [10.0 ** k for k in range(-2, 7)]
    val = {lam: float(np.abs(predict(lam, "VAL").mean(1) - data["VAL"][1]).mean()) for lam in lams}
    lam = min(val, key=val.get)
    print("VAL MAE by lambda:", {f"{k:g}": round(v, 4) for k, v in val.items()}, "->", lam)
    for s in splits[1:]:
        write("panecho_ridge", s, data[s][0], data[s][1], predict(lam, s), {"lambda": lam, "val_by_lambda": val})


def pe_table(n):
    """PanEcho's fixed sinusoidal positions for n frames (the release stores the first 16)."""
    pos = torch.arange(n).unsqueeze(1)
    div = torch.exp(torch.arange(0, 768, 2) * (-math.log(10000.0) / 768))
    pe = torch.zeros(1, n, 768)
    pe[0, :, 0::2] = torch.sin(pos * div)
    pe[0, :, 1::2] = torch.cos(pos * div)
    return pe


def fwd(m, x, valid):
    """Transformer + mean over valid frames + EF head -> EF in % (b,)."""
    z = m.transformer(x + m.pe[:, :x.shape[1]])
    w = valid.to(z.dtype).unsqueeze(-1)
    return m.EF_head((z * w).sum(1) / w.sum(1).clamp_min(1)).squeeze(-1)


def tune(a, dev):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model = PanEchoEF("quirk")
    model.encoder = None                                  # features are cached; only Transformer + head train
    splits = ["TRAIN", "VAL"] + (["TEST"] if a.split == "TEST" else [])
    if a.clips == "s3":
        # our clip protocol on the 3-frame-stack cache: T 32, stride 2, loop pad, pool over real frames
        from cardiac.dataset import EpochSampler, FeatClips
        pe = pe_table(32)
        assert torch.allclose(pe[:, :16], model.pe, atol=1e-6)
        model.pe = pe
        T, fd, kw = 32, config.CACHE / "feats_panecho_s3", dict(T=32, stride=2, pad="loop")
        train_ds = FeatClips("TRAIN", fd, mode="random", **kw)
        train_ds.sample_seed = a.seed
        sampler = EpochSampler(len(train_ds), seed=a.seed)
        train_dl = DataLoader(train_ds, batch_size=a.batch, sampler=sampler, drop_last=True, num_workers=2,
                              persistent_workers=True)
        evals = {s: FeatClips(s, fd, mode="multi", n_clips=a.n_clips, **kw) for s in splits[1:]}
        meta = {s: (d.names, d.ef) for s, d in evals.items()}
        steps_per_epoch = len(train_dl)

        def batches(ep):
            sampler.set_epoch(ep)
            for b in train_dl:
                yield b["video"].to(dev).float(), b["valid"].to(dev), b["ef"].to(dev)

        def predict(m, s):
            pred = np.zeros((len(evals[s].names), a.n_clips), np.float32)
            m.eval()
            with torch.no_grad():
                for b in DataLoader(evals[s], batch_size=64, num_workers=2):
                    pred[b["vid"].numpy(), b["clip"].numpy()] = fwd(m, b["video"].to(dev).float(),
                                                                    b["valid"].to(dev)).float().cpu().numpy()
            m.train()
            return pred
    else:
        # PanEcho's own clips: 16 consecutive frames, released packing (cached by --stage extract)
        T = 16
        data = {s: load(s) for s in splits}
        meta = {s: data[s][:2] for s in splits[1:]}
        e_tr, y_tr = torch.from_numpy(data["TRAIN"][2]), torch.from_numpy(data["TRAIN"][1])
        V, C = e_tr.shape[:2]
        steps_per_epoch = V // a.batch
        g = torch.Generator().manual_seed(a.seed)
        ones = torch.ones(a.batch, 16, dtype=torch.bool, device=dev)

        def batches(ep):
            perm, clip = torch.randperm(V, generator=g), torch.randint(0, C, (V,), generator=g)
            for i in range(steps_per_epoch):
                j = perm[i * a.batch:(i + 1) * a.batch]
                yield e_tr[j, clip[j]].to(dev).float(), ones, y_tr[j].to(dev)

        def predict(m, s):
            e = torch.from_numpy(data[s][2])
            out = []
            m.eval()
            with torch.no_grad():
                for i in range(0, len(e), 64):
                    x = e[i:i + 64].to(dev).float().flatten(0, 1)
                    out.append(fwd(m, x, torch.ones(x.shape[:2], dtype=torch.bool, device=dev)).view(-1, C).cpu())
            m.train()
            return torch.cat(out).numpy()

    model = model.to(dev)
    total = a.epochs * steps_per_epoch
    warm = int(0.05 * total)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(a.ema))
    best, best_state, best_ep, log = float("inf"), None, -1, []
    for ep in range(a.epochs):
        losses = []
        for x, valid, y in batches(ep):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = fwd(model, x, valid)
            loss = torch.nn.functional.mse_loss((p.float() - y) / config.EF_STD, torch.zeros_like(y))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ema.update_parameters(model)
            losses.append(loss.item())
        mae = float(np.abs(predict(ema.module, "VAL").mean(1) - meta["VAL"][1]).mean())
        log.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_MAE": mae})
        print(json.dumps(log[-1]), flush=True)
        if mae < best:
            best, best_ep, best_state = mae, ep, copy.deepcopy(ema.module.state_dict())
    ema.module.load_state_dict(best_state)
    name = f"panecho_tune{'_s3' if a.clips == 's3' else ''}_lr{a.lr:g}_s{a.seed}"
    extra = {"lr": a.lr, "seed": a.seed, "epochs": a.epochs, "best_epoch": best_ep, "ema": a.ema, "clips": a.clips,
             "log": log}
    for s in splits[1:]:
        write(name, s, *meta[s], predict(ema.module, s), extra, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["extract", "ridge", "tune"])
    ap.add_argument("--n-clips", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default="VAL", choices=["VAL", "TEST"], help="TEST: also write test predictions")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clips", default="panecho", choices=["panecho", "s3"],
                    help="tune: PanEcho's own 16-frame clips, or our protocol on the 3-frame-stack cache")
    a = ap.parse_args()
    dev = torch.device("cuda")
    {"extract": extract, "ridge": ridge, "tune": tune}[a.stage](a, dev)


if __name__ == "__main__":
    main()
