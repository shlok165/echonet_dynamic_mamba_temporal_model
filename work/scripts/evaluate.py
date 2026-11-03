r"""Full test-set evaluation of one trained run (model frozen throughout).

For every T in --Ts (the checkpoint is trained at one T; see the sweep protocol in the README):
  * EF metrics on video-level predictions (mean over --n-clips evenly spaced clips per video):
    MAE, MSE, RMSE, R2, MAPE, Pearson, Spearman, CCC, bias, within5/10, AUROC/sens/spec at EF<40 and <50,
    MAE per EF bin, Bland-Altman (stats + plot)
  * frame-order test: the same clips with frames shuffled (fixed permutation per clip)
  * temporal consistency: cosine(Z_t, Z_t+1) over real (non-padding) frames; EF std across clips
  * periodicity: autocorrelation peak of Z_t per clip (numeric) + PCA loop figures
  * efficiency: model forward time per clip and per video (excl. decoding), peak memory
ED/ES probe (at --probe-Ts): ridge on frozen Z_t (and on encoder F_t as a reference), fitted on TRAIN
traced clips, lambda chosen on VAL, reported on TEST. Search window: metrics.probe_window.
--split VAL (development, used for every design decision): everything is reported on VAL instead and
written to eval_VAL.json; the probe's lambda is then also chosen on VAL (slightly optimistic, same for
every model). The test set is only evaluated with the default --split TEST.

Usage: py.cmd scripts\evaluate.py --run bigru_T32_s0 [--split VAL]
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
from torch.utils.data import DataLoader  # noqa: E402

from cardiac import config  # noqa: E402
from cardiac import metrics as M  # noqa: E402
from cardiac.dataset import EchoClips, FeatClips, StackClips, pixel_stats  # noqa: E402
from cardiac.models import EFModel, FeatModel, count_params  # noqa: E402
from cardiac.panecho import PanEchoEFModel  # noqa: E402

GiB = 1024 ** 3


def load_model(run, dev, override=None):
    """override: dict merged into the run's temporal_kw (inference-only options, e.g. min_clock_T)."""
    ck = torch.load(config.RUNS / run / "best.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    kw = {**json.loads(a["temporal_kw"]), **(override or {})}
    if a.get("features"):
        d_in = ck["model"].get("grid.proj.weight", ck["model"]["temporal.inp.weight"]).shape[1]
        model = FeatModel(a["temporal"], d_in, d_model=a["d_model"], n_layers=a["n_layers"], temporal_kw=kw)
    elif a.get("panecho_tune"):
        model = PanEchoEFModel(a["temporal"], d_model=a["d_model"], n_layers=a["n_layers"], temporal_kw=kw,
                               n_tune=a["panecho_tune"])
    else:
        mean, std = pixel_stats()
        model = EFModel(a["temporal"], mean, std, d_model=a["d_model"], n_layers=a["n_layers"], pretrained=False,
                        temporal_kw=kw)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    if hasattr(model, "encoder"):
        model.encoder.to(memory_format=torch.channels_last)
    return model, a, ck["epoch"]


def make_clips(a, split, **kw):
    """The run's dataset: decoded frames, its cached frozen features (args "features"), or PanEcho's
    3-frame-stack images (args "panecho_tune")."""
    if a.get("features"):
        return FeatClips(split, config.feat_dir(a["features"]), **kw)
    if a.get("panecho_tune"):
        return StackClips(split, **kw)
    return EchoClips(split, **kw)


@torch.no_grad()
def run_pass(model, a, dev, split, T, mode, n_clips=1, frame_order="normal", keep_Z=False, keep_F=False,
             batch=16, workers=8, **ds_kw):
    ds = make_clips(a, split, T=T, stride=a["stride"], pad=a["pad"], mode=mode, n_clips=n_clips,
                    frame_order=frame_order, **ds_kw)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    out = {k: [] for k in ("pred", "ef", "vid", "clip", "valid", "frame_idx", "ed", "es", "stride", "Z", "F")}
    t_model = 0.0
    torch.cuda.reset_peak_memory_stats()
    for b in dl:
        v = b["video"].to(dev, non_blocking=True)
        valid = b["valid"].to(dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(v, valid if a["pool_valid"] else None)
        torch.cuda.synchronize()
        t_model += time.perf_counter() - t0
        out["pred"].append(o["ef"].float().cpu().numpy() * config.EF_STD + config.EF_MEAN)
        if keep_Z:
            out["Z"].append(o["Z"].float().cpu().numpy().astype(np.float16))
        if keep_F:
            out["F"].append(o["F"].float().cpu().numpy().astype(np.float16))
        for k in ("ef", "vid", "clip", "valid", "frame_idx", "ed", "es", "stride"):
            out[k].append(b[k].numpy())
    res = {k: np.concatenate(v) for k, v in out.items() if v}
    res["names"] = ds.names
    res["model_s"] = t_model
    res["peak_gib"] = torch.cuda.max_memory_allocated() / GiB
    return res


def ef_block(r):
    u, pm = M.video_mean(r["vid"], r["pred"])
    _, ym = M.video_mean(r["vid"], r["ef"])
    d = M.regression_metrics(ym, pm)
    d.update(M.threshold_metrics(ym, pm, 40))
    d.update(M.threshold_metrics(ym, pm, 50))
    d["bland_altman"] = M.bland_altman(ym, pm)
    d["mae_by_bin"] = M.binned_mae(ym, pm)
    return d, ym, pm


def periodicity(r):
    peaks = []
    for z, v in zip(r["Z"], r["valid"]):
        zz = z[v].astype(np.float64)
        if len(zz) >= 8:
            p, _ = M.autocorr_periodicity(zz)
            if np.isfinite(p):
                peaks.append(p)
    return {"autocorr_peak_mean": float(np.mean(peaks)) if peaks else float("nan"), "n_clips": len(peaks)}


def probe(model, a, dev, T, args, feat="Z"):
    """Ridge probe on frozen per-frame features; returns frame errors for ED and ES on args.split."""
    sets = {}
    splits = (("TRAIN", args.probe_train_frac), ("VAL", None)) + ((("TEST", None),) if args.split == "TEST" else ())
    for split, frac in splits:
        r = run_pass(model, a, dev, split, T, "traced", keep_Z=feat == "Z", keep_F=feat == "F",
                     require_tracings=True, subset_frac=frac, subset_seed=1, workers=args.workers)
        X = r[feat].astype(np.float64)
        win = M.probe_window(r["frame_idx"], r["ed"], r["es"], valid=r["valid"])
        ok = (M.in_clip(r["frame_idx"], r["valid"], r["ed"], r["stride"])
              & M.in_clip(r["frame_idx"], r["valid"], r["es"], r["stride"]))
        sigma = float(a["stride"])
        Y = np.stack([M.heatmap_targets(r["frame_idx"], r["ed"], sigma),
                      M.heatmap_targets(r["frame_idx"], r["es"], sigma)], -1)
        sets[split] = dict(X=X, Y=Y, win=win, ok=ok, r=r)

    def fit(lam):
        tr = sets["TRAIN"]
        m = tr["ok"][:, None] & tr["r"]["valid"]
        return M.ridge_fit(tr["X"][m], tr["Y"][m], lam)

    def errors(model_, s):
        N, Tn, D = s["X"].shape
        sc = M.ridge_predict(model_, s["X"].reshape(-1, D)).reshape(N, Tn, 2)
        k = s["ok"]
        out = {}
        for j, name in enumerate(("ED", "ES")):
            pred = M.localize(sc[k, :, j], s["r"]["frame_idx"][k], s["win"][k])
            true = s["r"]["ed"][k] if name == "ED" else s["r"]["es"][k]
            out[name] = M.frame_error(pred, true)
        return out

    best = None
    for lam in args.lams:
        mdl = fit(lam)
        e = errors(mdl, sets["VAL"])
        score = e["ED"]["mean"] + e["ES"]["mean"]
        if best is None or score < best[0]:
            best = (score, lam, mdl)
    te = errors(best[2], sets[args.split])
    # trivial reference: always predict the window centre
    s = sets[args.split]
    k = s["ok"]
    w_lo = np.where(s["win"][k], s["r"]["frame_idx"][k], np.inf).min(1)
    w_hi = np.where(s["win"][k], s["r"]["frame_idx"][k], -np.inf).max(1)
    centre = (w_lo + w_hi) / 2
    ref = {"ED": M.frame_error(centre, s["r"]["ed"][k]), "ES": M.frame_error(centre, s["r"]["es"][k])}
    return {"feature": feat, "lambda": best[1], "test": te, "window_centre_reference": ref,
            "n_train_clips": int(sets["TRAIN"]["ok"].sum()), "n_test_clips": int(k.sum()), "sigma": float(a["stride"])}


def pca_figure(r, path, n_videos=6, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    first = np.where(r["clip"] == 0)[0]
    # longest-real-frame clips first, so loops over several beats are visible
    order = first[np.argsort(-r["valid"][first].sum(1))][:n_videos]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), dpi=130)
    for ax, i in zip(axes.ravel(), order):
        v = r["valid"][i]
        z = r["Z"][i][v].astype(np.float64)
        fi = r["frame_idx"][i][v]
        c, ev = M.pca_2d(z)
        ax.plot(c[:, 0], c[:, 1], "-", color="0.75", lw=0.8, zorder=1)
        sc = ax.scatter(c[:, 0], c[:, 1], c=fi, cmap="viridis", s=10, zorder=2)
        for tf, mk, lab in ((r["ed"][i], "s", "ED"), (r["es"][i], "^", "ES")):
            if tf >= 0:
                j = np.argmin(np.abs(fi - tf))
                if abs(fi[j] - tf) <= r["stride"][i]:
                    ax.scatter(c[j, 0], c[j, 1], marker=mk, s=60, facecolor="none", edgecolor="red", zorder=3,
                               label=lab)
        ax.set_title(f"{r['names'][r['vid'][i]]}  EF {r['ef'][i]:.0f}\nPC var {ev[0]:.2f}+{ev[1]:.2f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower right", fontsize=8)
    fig.colorbar(sc, ax=axes, shrink=0.6, label="original frame index")
    fig.suptitle(title, fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--Ts", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    ap.add_argument("--probe-Ts", nargs="+", type=int, default=None, help="default: the training T")
    ap.add_argument("--n-clips", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--probe-train-frac", type=float, default=0.3)
    ap.add_argument("--lams", nargs="+", type=float, default=[1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6])
    ap.add_argument("--probe-only", action="store_true", help="skip the T sweep, (re)run only the probe")
    ap.add_argument("--split", default="TEST", choices=["TEST", "VAL"], help="VAL = development evaluation")
    ap.add_argument("--temporal-override", default="", help="inference-only temporal options, e.g. min_clock_T=24")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--clip-mode", default="even", choices=["even", "phase"],
                    help="phase: clips one whole-video cardiac period apart, at the same phase (features only)")
    ap.add_argument("--warp", type=float, default=1.0, help="time-warp factor of the clips (features only)")
    args = ap.parse_args()
    from train import parse_kw
    override = parse_kw(args.temporal_override) if args.temporal_override else None
    suffix = ("" if args.split == "TEST" else f"_{args.split}") + (f"_{args.tag}" if args.tag else "")
    dev = torch.device("cuda")
    model, a, epoch = load_model(args.run, dev, override)
    out_dir = config.RESULTS / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"run": args.run, "best_epoch": epoch, "train_args": a, "n_clips": args.n_clips,
           "params_total": count_params(model), "params_temporal": count_params(model.temporal), "by_T": {}}
    res["split"] = args.split
    res["temporal_override"] = override
    out_json = out_dir / f"eval{suffix}.json"
    if args.probe_only and out_json.exists():
        res = json.loads(out_json.read_text())
    for T in ([] if args.probe_only else args.Ts):
        t0 = time.perf_counter()
        extra = {"clip_mode": args.clip_mode, "warp": args.warp} if a.get("features") else {}
        r = run_pass(model, a, dev, args.split, T, "multi", args.n_clips, keep_Z=True, workers=args.workers, **extra)
        ef, ym, pm = ef_block(r)
        n_videos = len(np.unique(r["vid"]))
        first_frame = r["frame_idx"][:, 0]
        n_starts = {v: len(np.unique(first_frame[r["vid"] == v])) for v in np.unique(r["vid"])}
        distinct = np.array([n_starts[v] > 1 for v in r["vid"]])
        rec = {"ef": ef,
               "ef_std_across_clips": M.ef_std_across_clips(r["vid"], r["pred"]),
               # only videos whose clips actually differ (short videos give identical looped clips -> sd 0)
               "ef_std_across_distinct_clips": M.ef_std_across_clips(r["vid"][distinct], r["pred"][distinct]),
               "cosine_smoothness": M.cosine_smoothness(r["Z"].astype(np.float32), r["valid"]),
               "periodicity": periodicity(r),
               "frac_padded_frames": float(1 - r["valid"].mean()),
               "infer_model_ms_per_clip": 1000 * r["model_s"] / len(r["pred"]),
               "infer_model_ms_per_video": 1000 * r["model_s"] / n_videos,
               "infer_peak_gib": r["peak_gib"]}
        M.bland_altman_plot(ym, pm, out_dir / f"bland_altman_T{T}{suffix}.png", title=f"{args.run}, T={T}")
        pca_figure(r, out_dir / f"pca_T{T}{suffix}.png", title=f"{args.run}: PCA of Z_t, T={T}")
        np.savez_compressed(out_dir / f"{args.split.lower()}_preds_T{T}{('_' + args.tag) if args.tag else ''}.npz",
                            names=r["names"], vid=r["vid"],
                            clip=r["clip"], pred=r["pred"], ef=r["ef"])
        del r
        rs = run_pass(model, a, dev, args.split, T, "multi", args.n_clips, frame_order="shuffle",
                      workers=args.workers)
        efs, _, _ = ef_block(rs)
        rec["shuffled_ef"] = {k: efs[k] for k in ("MAE", "RMSE", "R2", "Pearson", "Spearman")}
        rec["shuffle_change"] = M.relative_change(ef, efs)
        rec["eval_wall_s"] = time.perf_counter() - t0
        res["by_T"][T] = rec
        print(f"T={T}: MAE {ef['MAE']:.3f} RMSE {ef['RMSE']:.3f} R2 {ef['R2']:.3f} | shuffled MAE "
              f"{efs['MAE']:.3f} | cos {rec['cosine_smoothness']:.4f} | EF sd {rec['ef_std_across_clips']['mean_std']:.3f}",
              flush=True)
        out_json.write_text(json.dumps(res, indent=1, default=float))
    res["probe"] = {}
    for T in (args.probe_Ts or [a["T"]]):
        res["probe"][T] = {f: probe(model, a, dev, T, args, f) for f in ("Z", "F")}
        p = res["probe"][T]
        print(f"probe T={T}: Z ED {p['Z']['test']['ED']['mean']:.2f} ES {p['Z']['test']['ES']['mean']:.2f} | "
              f"F ED {p['F']['test']['ED']['mean']:.2f} ES {p['F']['test']['ES']['mean']:.2f} | window-centre ED "
              f"{p['Z']['window_centre_reference']['ED']['mean']:.2f} (frames)", flush=True)
        out_json.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", out_json)


if __name__ == "__main__":
    main()
