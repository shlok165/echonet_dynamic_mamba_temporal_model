r"""Results of every non-baseline model against its BiGRU counterpart under the same protocol, one figure per
architecture diagram (scripts/draw_architectures.py), written next to them as <n>_<name>_vs_bigru[_<split>].png/.pdf
plus vs_bigru_summary_<split>.csv.

  1, 2, 3  frozen PanEcho 3-frame-stack features, weight EMA, 3 seeds, 4 clips per video: sync variants vs
           BiGRU (per split: --splits val test; a split is skipped until all its predictions exist).
           Paired statistics over videos with seed-averaged absolute errors, as scripts/stats.py.
  4        readout heads: single-frame features, seed 0, best single-clip VAL MAE from the training logs;
           and the volumetric readout on 3-frame-stack features, 3 seeds, 4-clip VAL (no EMA, as trained).
  5        encoder fine-tuning (BiGRU): per-epoch single-clip VAL MAE and 4-clip VAL vs the frozen BiGRU.
  6        end-to-end ResNet-18: per-epoch single-clip VAL MAE of BiGRU and the sync variants (seed 0).

Usage: py.cmd scripts\compare_vs_bigru.py [--splits val test]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cardiac import config  # noqa: E402
from stats import per_video  # noqa: E402

SYNC, BIGRU, THIRD = "#2a78d6", "#eb6834", "#1baf7a"           # validated categorical slots 1-3
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK})
OUT = config.DATA_OUT / "figures" / "architecture"
REF = ("BiGRU", "bigru_ema")
MAIN = [("sync, fixed frequency (final)", "sync_novol_fixed_ema"), ("sync, heart-rate clock", "sync_novol_ema"),
        ("sync, tempered clock", "sync_novol_tempered_ema"), ("sync, learned angles (Mamba-3 style)",
                                                             "sync_novol_learned_ema"),
        ("sync, no rotation", "sync_novol_noclk_ema")]
EVAL = {"val": "eval_VAL.json", "test": "eval.json"}
SPLIT_NAME = {"val": "validation", "test": "test"}


def seeds(m, prefix="pe3"):
    return [f"{prefix}_{m}_s{s}" for s in range(3)]


def errors(runs, split, T=32):
    """(video names, (n_seeds, V) absolute errors, truth) on the split."""
    ps, y = [], None
    for r in runs:
        p, y = per_video(r, split, T)
        ps.append(p)
    vids = sorted(set.intersection(*(set(p) for p in ps)))
    return vids, np.array([[abs(p[v] - y[v]) for v in vids] for p in ps]), np.array([y[v] for v in vids])


def paired(ea, eb, n_boot=10000):
    """Seed-averaged per-video errors a - b: mean, 95% bootstrap CI, two-sided bootstrap p (stats.py)."""
    d = ea.mean(0) - eb.mean(0)
    idx = np.random.default_rng(0).integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))


def evals(m, split):
    return [json.load(open(config.RESULTS / r / EVAL[split])) for r in seeds(m)]


def complete(path):
    """evaluate.py rewrites its json after every clip length and adds the probe last."""
    try:
        return "probe" in json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def have(split, models):
    return all((config.RESULTS / r / f"{split}_preds_T32.npz").exists() and complete(config.RESULTS / r / EVAL[split])
               for m in models for r in seeds(m))


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def seed_dots(ax, x, vals, color, label=None, fs=8):
    """Seeds as dots, their mean as a short bar; label (default: the mean) to the right of the bar."""
    ax.scatter(np.full(len(vals), x) + np.linspace(-0.08, 0.08, len(vals)), vals, s=30, color=color, zorder=3,
               edgecolor="white", linewidth=0.8)
    ax.plot([x - 0.2, x + 0.2], [np.mean(vals)] * 2, color=color, lw=2.2, zorder=4, solid_capstyle="round")
    ax.text(x + 0.26, np.mean(vals), label or f"{np.mean(vals):.3f}", va="center", ha="left", fontsize=fs,
            color=INK, zorder=5)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# ----------------------------------------------------------------------------- 1 headline

ROWS = [("MAE (EF points)", "MAE", "low"), ("RMSE", "RMSE", "low"), ("R²", "R2", "high"), ("Pearson r", "Pearson", "high"),
        ("CCC", "CCC", "high"), ("|bias|", "bias", "abs"), ("within 5 points (%)", "within5", "high"),
        ("within 10 points (%)", "within10", "high"), ("AUROC, EF < 40%", "AUROC_lt40", "high"),
        ("AUROC, EF < 50%", "AUROC_lt50", "high")]


def metric(e, key):
    v = e["by_T"]["32"]["ef"][key]
    return abs(v) if key == "bias" else (100 * v if key.startswith("within") and v <= 1 else v)


def fig_headline(split, rows_out):
    name, key = MAIN[0]
    ev_s, ev_b = evals(key, split), evals(REF[1], split)
    _, es, _ = errors(seeds(key), split)
    _, eb, _ = errors(seeds(REF[1]), split)
    d, lo, hi, p = paired(es, eb)
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 2.3], wspace=0.45)
    for i, (lab, k, better) in enumerate(ROWS[:3]):
        ax = fig.add_subplot(gs[0, i])
        vs, vb = [metric(e, k) for e in ev_s], [metric(e, k) for e in ev_b]
        seed_dots(ax, 0, vs, SYNC)
        seed_dots(ax, 1, vb, BIGRU)
        ax.set_xticks([0, 1], ["sync\n(final)", "BiGRU"])
        ax.set_xlim(-0.5, 1.9)
        ax.set_title(lab + ("  ↓" if better == "low" else "  ↑"), fontsize=9.5)
        grid(ax)
    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    cell, wins = [], []
    for lab, k, better in ROWS:
        a, b = np.mean([metric(e, k) for e in ev_s]), np.mean([metric(e, k) for e in ev_b])
        win = a < b if better in ("low", "abs") else a > b
        wins.append(win)
        f = "{:.1f}" if k.startswith("within") else "{:.3f}"
        cell.append([lab, f.format(a), f.format(b), "sync" if win else "BiGRU"])
        rows_out.append({"split": split, "comparison": f"{name} vs BiGRU", "metric": lab, "model": float(a),
                         "BiGRU": float(b), "better": cell[-1][-1]})
    tb = ax.table(cellText=cell, colLabels=["metric", "sync", "BiGRU", "better"],
                  loc="upper center", cellLoc="center", colWidths=[0.4, 0.18, 0.18, 0.16])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.5)
    tb.scale(1, 1.32)
    for (r, c), cl in tb.get_celld().items():
        cl.set_edgecolor(GRID)
        if r == 0:
            cl.set_text_props(fontweight="bold", color=INK)
        elif c == 3:
            cl.set_text_props(color=SYNC if wins[r - 1] else BIGRU, fontweight="bold")
    ax.text(0.04, -0.02, f"Means of 3 seeds.  Paired over {es.shape[1]} {SPLIT_NAME[split]} videos "
            f"(seed-averaged errors):\nΔMAE (sync − BiGRU) = {d:+.3f}  [95% CI {lo:+.3f}, {hi:+.3f}],  {fmt_p(p)}",
            transform=ax.transAxes, fontsize=9, color=INK, va="top")
    fig.suptitle(f"Final model vs BiGRU — {SPLIT_NAME[split]} set (frozen PanEcho features, weight EMA, 3 seeds, "
                 f"4 clips per video)", x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.02)
    save(fig, f"1_pipeline_vs_bigru_{split}")


# ----------------------------------------------------------------------------- 2 all metrics

def fig_all_metrics(split):
    (ns, ks), (nh, kh) = MAIN[0], MAIN[1]
    groups = [("sync fixed (final)", ks, SYNC), ("BiGRU", REF[1], BIGRU), ("sync hr clock", kh, THIRD)]
    ev = {k: evals(k, split) for _, k, _ in groups}
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.4))
    fig.subplots_adjust(wspace=0.38, hspace=0.55)
    # (a) clip length
    ax = axs[0, 0]
    Ts = ["8", "16", "32", "64", "128"]
    for lab, k, c in groups:
        y = [np.mean([e["by_T"][T]["ef"]["MAE"] for e in ev[k]]) for T in Ts]
        ax.plot(range(len(Ts)), y, color=c, lw=2, marker="o", ms=5, markeredgecolor="white", zorder=3, label=lab)
    ax.set_xticks(range(len(Ts)), Ts)
    ax.set_xlabel("clip length T (frames, stride 2; trained at 32)")
    ax.set_title("(a) MAE vs clip length  ↓")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    grid(ax)
    # (b) heart-rate strata
    ax = axs[0, 1]
    hr = pd.read_csv(config.RESULTS / "hr_reference.csv").set_index("FileName")["hr_bpm"]
    bins = [("< 60", 0, 60), ("60–100", 60, 100), ("> 100", 100, 1e9)]
    for j, (lab, k, c) in enumerate(groups):
        vids, E, _ = errors(seeds(k), split)
        h = hr.loc[vids].to_numpy()
        e = E.mean(0)
        m = [e[(h >= a) & (h < b)].mean() for _, a, b in bins]
        ax.scatter(np.arange(3) + (j - 1) * 0.22, m, color=c, s=34, zorder=3, edgecolor="white", linewidth=0.8,
                   label=lab)
    n = [int(((hr.loc[vids] >= a) & (hr.loc[vids] < b)).sum()) for _, a, b in bins]
    ax.set_xticks(range(3), [f"{lab}\n(n = {k})" for (lab, _, _), k in zip(bins, n)])
    ax.set_xlabel("heart rate (bpm, label-free estimate)")
    ax.set_title("(b) MAE by heart rate  ↓")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    grid(ax)
    # (c)-(g) seed dots
    panels = [("(c) EF SD across a video's clips  ↓", lambda e: e["by_T"]["32"]["ef_std_across_clips"]["mean_std"]),
              ("(d) feature smoothness (cos)  ↑", lambda e: e["by_T"]["32"]["cosine_smoothness"]),
              ("(e) periodicity (autocorr peak)  ↑", lambda e: e["by_T"]["32"]["periodicity"]["autocorr_peak_mean"]),
              ("(f) ED probe error (frames)  ↓", lambda e: e["probe"]["32"]["Z"]["test"]["ED"]["mean"]),
              ("(g) ES probe error (frames)  ↓", lambda e: e["probe"]["32"]["Z"]["test"]["ES"]["mean"])]
    for ax, (title, f) in zip([axs[0, 2], axs[0, 3], axs[1, 0], axs[1, 1], axs[1, 2]], panels):
        for j, (lab, k, c) in enumerate(groups):
            seed_dots(ax, j, [f(e) for e in ev[k]], c, fs=7.5)
        ax.set_xticks(range(3), ["sync\nfixed", "BiGRU", "sync\nhr"])
        ax.set_xlim(-0.5, 3.0)
        ax.set_title(title)
        grid(ax)
    # (h) efficiency
    ax = axs[1, 3]
    ax.axis("off")
    b = json.load(open(config.RESULTS / "bench_feat.json"))["models"]
    rows = []
    for lab, key in (("sync fixed", "sync_fixed"), ("BiGRU", "bigru"), ("sync hr", "sync_hr")):
        r, t = b[key], b[key]["by_T"]["32"]
        rows.append([lab, f"{r['params_total'] / 1e6:.2f}M", f"{t['fwd_gflops_per_clip']:.3f}",
                     f"{t['infer_graph_ms_per_clip_b16']:.3f}", f"{t['train_ms_per_step_b16']:.1f}"])
    tb = ax.table(cellText=rows, colLabels=["model", "params", "GFLOP/\nclip", "ms/clip\nb16", "train\nms/step"],
                  loc="center", cellLoc="center", colWidths=[0.3, 0.18, 0.17, 0.17, 0.18])
    tb.auto_set_font_size(False)
    tb.set_fontsize(7.5)
    tb.scale(1, 2.1)
    for (r, c), cl in tb.get_celld().items():
        cl.set_edgecolor(GRID)
        if r == 0:
            cl.set_text_props(fontweight="bold")
    ax.set_title("(h) efficiency of the head, T = 32  ↓")
    ax.text(0.5, 0.08, "frozen encoder (shared) excluded; RTX 3050,\nCUDA graphs, batch 16", transform=ax.transAxes,
            ha="center", fontsize=7.5, color=MUTED)
    fig.suptitle(f"Sync block vs BiGRU on every evaluation metric — {SPLIT_NAME[split]} set (mean of 3 seeds; "
                 f"dots = seeds)", x=0.01, ha="left", fontsize=12, fontweight="bold", y=0.99)
    save(fig, f"2_sync_block_vs_bigru_{split}")


# ----------------------------------------------------------------------------- 3 clock variants

def fig_clocks(split, rows_out):
    _, eb, _ = errors(seeds(REF[1]), split)
    res = []
    for name, key in MAIN:
        _, e, _ = errors(seeds(key), split)
        res.append((name, *paired(e, eb), float(np.mean([e_.mean() for e_ in e]))))
        rows_out.append({"split": split, "comparison": f"{name} vs BiGRU", "metric": "MAE difference",
                         "model": res[-1][1], "BiGRU": 0.0, "better": f"[{res[-1][2]:+.3f}, {res[-1][3]:+.3f}] "
                                                                        f"{fmt_p(res[-1][4])}"})
    bm = float(np.mean([e_.mean() for e_ in eb]))
    fig, ax = plt.subplots(figsize=(11, 3.9))
    for i, (name, d, lo, hi, p, m) in enumerate(res):
        y = len(res) - 1 - i
        c = SYNC if i == 0 else INK2
        ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round", zorder=3)
        ax.scatter([d], [y], color=c, s=50, zorder=4, edgecolor="white", linewidth=1)
        ax.text(0.205, y, f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]   {fmt_p(p)}   (MAE {m:.3f})", va="center",
                fontsize=8.5, color=INK if i == 0 else INK2, fontweight="bold" if i == 0 else "normal")
    ax.axvline(0, color=BIGRU, lw=1.5, zorder=2)
    ax.text(0.003, len(res) - 0.45, f"BiGRU (MAE {bm:.3f})", color=INK2, fontsize=8.5, ha="left")
    ax.set_yticks(range(len(res)), [r[0] for r in reversed(res)])
    ax.set_xlim(-0.25, 0.2)
    ax.set_ylim(-0.6, len(res) - 0.2)
    ax.set_xlabel("MAE difference vs BiGRU (EF points; < 0 = better than BiGRU), 95% paired bootstrap CI")
    grid(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle(f"Sync clock variants vs BiGRU — {SPLIT_NAME[split]} set (3 seeds, 4 clips per video, weight EMA)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.0)
    save(fig, f"3_clock_variants_vs_bigru_{split}")


# ----------------------------------------------------------------------------- 4 readouts (VAL)

def log(run):
    return [json.loads(line) for line in open(config.RUNS / run / "log.jsonl")]


def fig_readouts(rows_out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.35})
    runs = [("BiGRU", "pe_bigru_lr1e-3_s0", BIGRU), ("BiGRU + harmonic + range", "pe_bigru_harm_range_s0", INK2),
            ("sync (no readout)", "pe_sync_novol_s0", SYNC), ("sync + volumetric", "pe_sync_lr1e-3_s0", INK2),
            ("sync + range", "pe_sync_range_s0", INK2), ("sync + harmonic + range", "pe_sync_harm_range_s0", INK2)]
    complete = [(lab, run, c) for lab, run, c in runs if len(log(run)) == json.load(open(config.RUNS / run /
                                                                                        "args.json"))["epochs"]]
    skipped = [lab for lab, run, _ in runs if (lab, run, _) not in complete]
    best = [(lab, min(r["val_MAE"] for r in log(run)), c) for lab, run, c in complete]
    ref = best[0][1]
    for i, (lab, v, c) in enumerate(best):
        y = len(best) - 1 - i
        a1.scatter([v], [y], color=c, s=46, zorder=3, edgecolor="white", linewidth=0.8)
        a1.text(v + 0.006, y, f"{v:.3f}  ({v - ref:+.3f})" if i else f"{v:.3f}", va="center", fontsize=8.5,
                color=INK2)
        rows_out.append({"split": "val", "comparison": f"{lab} vs BiGRU (single-frame features, seed 0)",
                         "metric": "best single-clip VAL MAE", "model": v, "BiGRU": ref,
                         "better": "BiGRU" if v > ref else "model"})
    a1.axvline(ref, color=BIGRU, lw=1.2, ls=(0, (4, 3)), zorder=2)
    a1.set_yticks(range(len(best)), [b[0] for b in reversed(best)])
    a1.set_xlim(ref - 0.06, max(b[1] for b in best) + 0.12)
    a1.set_xlabel("best single-clip VAL MAE (training log) ↓" +
                  (f"\nnot completed, omitted: {', '.join(skipped)}" if skipped else ""))
    a1.set_title("(a) single-frame PanEcho features, seed 0")
    grid(a1, "x")
    a1.tick_params(axis="y", length=0)
    # (b) s3 features, 3 seeds, 4-clip VAL, no EMA
    groups = [("BiGRU", "bigru", BIGRU), ("sync", "sync_novol", SYNC), ("sync +\nvolumetric", "sync", INK2)]
    _, eb, _ = errors(seeds("bigru"), "val")
    for j, (lab, k, c) in enumerate(groups):
        v = [json.load(open(config.RESULTS / r / "eval_VAL.json"))["by_T"]["32"]["ef"]["MAE"] for r in seeds(k)]
        txt = f"{np.mean(v):.3f}"
        if j:
            _, e, _ = errors(seeds(k), "val")
            d, lo, hi, p = paired(e, eb)
            txt += f"\nΔ {d:+.3f}\n{fmt_p(p)}"
            rows_out.append({"split": "val", "comparison": f"{lab.replace(chr(10), ' ')} vs BiGRU (3-frame stacks, "
                                                           f"no EMA)", "metric": "4-clip VAL MAE",
                             "model": float(np.mean(v)), "BiGRU": float(eb.mean()), "better": f"{fmt_p(p)}"})
        seed_dots(a2, j, v, c, label=txt, fs=7.5)
    a2.set_xticks(range(3), [g[0] for g in groups])
    a2.set_xlim(-0.5, 3.1)
    a2.set_ylabel("4-clip VAL MAE ↓")
    a2.set_title("(b) 3-frame-stack features, 3 seeds")
    grid(a2)
    fig.suptitle("Readout heads vs BiGRU (validation set): no readout variant beats BiGRU",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.02)
    save(fig, "4_readouts_vs_bigru")


# ----------------------------------------------------------------------------- 5 fine-tuning (VAL)

def fig_finetune(rows_out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.35})
    ft = log("ft_bigru_ft_s0")
    fr = min(r["val_MAE"] for r in log("pe3_bigru_ema_s0"))
    a1.plot([r["epoch"] + 1 for r in ft], [r["val_MAE"] for r in ft], color=SYNC, lw=2, marker="o", ms=5,
            markeredgecolor="white", zorder=3)
    a1.text(len(ft) + 0.1, ft[-1]["val_MAE"], "BiGRU, encoder stage 4\nfine-tuned", fontsize=8, color=INK2,
            va="center")
    a1.axhline(fr, color=BIGRU, lw=1.5, ls=(0, (4, 3)))
    a1.text(0.7, fr - 0.004, "frozen-encoder BiGRU (start point)", fontsize=8, color=INK2, va="top")
    a1.set_xlim(0.5, len(ft) + 1.8)
    a1.set_xticks(range(1, len(ft) + 1))
    a1.set_xlabel("fine-tuning epoch")
    a1.set_ylabel("single-clip VAL MAE ↓")
    a1.set_title("(a) validation error per epoch")
    grid(a1)
    _, ea, _ = errors(["ft_bigru_ft_s0"], "val")
    _, eb, _ = errors(["pe3_bigru_ema_s0"], "val")
    d, lo, hi, p = paired(ea, eb)
    for j, (lab, e, c) in enumerate((("frozen\nBiGRU", eb, BIGRU), ("fine-tuned\n(best epoch)", ea, SYNC))):
        a2.bar(j, e.mean(), width=0.5, color=c, zorder=3)
        a2.text(j, e.mean() + 0.01, f"{e.mean():.3f}", ha="center", fontsize=8.5, color=INK)
    a2.set_xticks([0, 1], ["frozen\nBiGRU", "fine-tuned\n(best = epoch 1)"])
    a2.set_ylim(3.8, 4.15)
    a2.set_ylabel("4-clip VAL MAE ↓ (axis from 3.8)")
    a2.set_title("(b) seed 0, 4 clips per video")
    a2.text(0.5, 4.12, f"Δ {d:+.3f} [{lo:+.3f}, {hi:+.3f}], {fmt_p(p)}", ha="center", fontsize=8.5, color=INK)
    grid(a2)
    rows_out.append({"split": "val", "comparison": "fine-tuned BiGRU vs frozen BiGRU (seed 0)",
                     "metric": "4-clip VAL MAE", "model": float(ea.mean()), "BiGRU": float(eb.mean()),
                     "better": fmt_p(p)})
    fig.suptitle("Encoder fine-tuning vs frozen BiGRU (validation set): the encoder overfits after the first epoch",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.03)
    save(fig, "5_finetune_vs_bigru")


# ----------------------------------------------------------------------------- 6 end to end (VAL)

def fig_end_to_end(rows_out):
    fig, ax = plt.subplots(figsize=(11, 4.3))
    runs = [("BiGRU", "bigru_T32_s0", BIGRU), ("sync + latent volume", "sync_hr_vol_T32_s0", INK2),
            ("sync + area readout", "sync_hr_area_T32_s0", SYNC)]
    ref = min(r["val_MAE"] for r in log("bigru_T32_s0"))
    names = []
    for lab, run, c in runs:
        lg = log(run)
        x, y = [r["epoch"] + 1 for r in lg], [r["val_MAE"] for r in lg]
        ax.plot(x, y, color=c, lw=1.8, zorder=3)
        b = int(np.argmin(y))
        ax.scatter([x[b]], [y[b]], color=c, s=44, zorder=4, edgecolor="white", linewidth=1)
        names.append(f"{lab}: best {y[b]:.3f} (epoch {x[b]})" +
                     (f", {y[b] - ref:+.3f} vs BiGRU" if run != "bigru_T32_s0" else ""))
        rows_out.append({"split": "val", "comparison": f"{lab} vs BiGRU (end-to-end ResNet-18, seed 0)",
                         "metric": "best single-clip VAL MAE", "model": y[b], "BiGRU": ref,
                         "better": "model" if y[b] < ref else "BiGRU"})
    ax.set_xlabel("epoch")
    ax.set_ylabel("single-clip VAL MAE ↓")
    ax.set_ylim(4.2, 6.0)
    grid(ax)
    ax.legend([plt.Line2D([], [], color=c, lw=2, marker="o", ms=5, markeredgecolor="white") for _, _, c in runs],
              names, frameon=False, loc="upper right", fontsize=8.5, title="dot = selected (best) epoch",
              title_fontsize=8)
    fig.suptitle("End-to-end ResNet-18 variants vs BiGRU (validation set, seed 0; y-axis clipped at 6)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.0)
    save(fig, "6_end_to_end_vs_bigru")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        rows = []
        if not have(split, [REF[1]] + [k for _, k in MAIN]):
            print(f"{split}: predictions incomplete, skipped")
            continue
        fig_headline(split, rows)
        fig_all_metrics(split)
        fig_clocks(split, rows)
        if split == "val":
            fig_readouts(rows)
            fig_finetune(rows)
            fig_end_to_end(rows)
        with open(OUT / f"vs_bigru_summary_{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["split", "comparison", "metric", "model", "BiGRU", "better"])
            w.writeheader()
            w.writerows(rows)
        print("wrote", OUT / f"vs_bigru_summary_{split}.csv")


if __name__ == "__main__":
    main()
