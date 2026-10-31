r"""Side-by-side comparison tables of every model (proposed sync variants, the four temporal baselines on the same
frozen features, the PanEcho baselines) as matplotlib figures, for scripts/build_arch_pdfs.py.

  table_accuracy(split)  EF accuracy, 4 clips per video, mean over seeds; paired MAE difference vs BiGRU
  table_behaviour(split) clip length, heart rate, clip consistency, smoothness, periodicity, ED/ES probes, cost
  table_explored()       explored variants vs BiGRU and Mamba under each variant's own (matched) protocol, VAL
A model whose predictions for the split do not exist yet is shown as "pending". Best value per row in bold.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cardiac import config  # noqa: E402
from compare_vs_bigru import complete, errors, paired  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
HEAD = {"proposed": "#D6EFD8", "baseline": "#FBE3C8", "panecho": "#DCE3EA", "label": "#F4F4F2"}
SPLIT_NAME = {"val": "validation", "test": "test"}


def s3(m):
    return [f"pe3_{m}_s{s}" for s in range(3)]


# (column label, group, runs, clip length T, result format, benchmark key)
MODELS = [
    ("sync fixed\n(final)", "proposed", s3("sync_novol_fixed_ema"), 32, "ours", "sync_fixed"),
    ("sync\nhr clock", "proposed", s3("sync_novol_ema"), 32, "ours", "sync_hr"),
    ("sync\ntempered", "proposed", s3("sync_novol_tempered_ema"), 32, "ours", None),
    ("sync learned\nangles", "proposed", s3("sync_novol_learned_ema"), 32, "ours", None),
    ("sync no\nrotation", "proposed", s3("sync_novol_noclk_ema"), 32, "ours", None),
    ("BiGRU", "baseline", s3("bigru_ema"), 32, "ours", "bigru"),
    ("Mamba", "baseline", s3("mamba_ema"), 32, "ours", "mamba"),
    ("Transformer\n(UVT)", "baseline", s3("uvt_ema"), 32, "ours", "uvt"),
    ("no temporal\nmodel", "baseline", s3("identity_ema"), 32, "ours", "identity"),
    ("PanEcho TF\nour clips", "panecho", [f"panecho_tune_s3_lr0.0003_s{s}" for s in range(3)], 32, "panecho",
     "panecho_tf"),
    ("PanEcho TF\n16 frames", "panecho", [f"panecho_tune_lr0.0001_s{s}" for s in range(3)], 16, "panecho",
     "panecho_tf"),
    ("PanEcho\nridge probe", "panecho", ["panecho_ridge"], 16, "panecho", None),
    ("PanEcho\nzero-shot", "panecho", ["panecho_zeroshot"], 16, "panecho", None),
]


def ref_of(models):
    return next(m for m in models if m[0] == "BiGRU")


def eval_file(run, fmt, split):
    if fmt == "ours":
        return config.RESULTS / run / ("eval_VAL.json" if split == "val" else "eval.json")
    return config.RESULTS / run / f"eval_{split.upper()}.json"


def available(m, split):
    _, _, runs, T, fmt, _ = m
    return all((complete(eval_file(r, fmt, split)) if fmt == "ours" else eval_file(r, fmt, split).exists()) and
               (config.RESULTS / r / f"{split}_preds_T{T}.npz").exists() for r in runs)


def ef(m, split):
    """Per-seed EF metric dicts at the model's clip length."""
    _, _, runs, T, fmt, _ = m
    ev = [json.load(open(eval_file(r, fmt, split))) for r in runs]
    return [e["by_T"][str(T)]["ef"] if fmt == "ours" else e["ef"] for e in ev]


def bench():
    p = config.RESULTS / "bench_feat.json"
    return json.load(open(p))["models"] if p.exists() else {}


def trained_params(m):
    name, _, runs, _, fmt, key = m
    if fmt == "ours":
        return json.load(open(config.RUNS / runs[0] / "info.json"))["params_total"]
    if key:
        return bench().get(key, {}).get("params_total")
    return 769 if "ridge" in name else 0


def draw_table(title, subtitle, row_labels, cols, cells, better, notes, size=(17.5, 9.0), fs=8.3,
               label_w=0.17):
    """cols: [(label, group)]; cells[r][c]: (text, value or None); better[r]: 'low' | 'high' | None."""
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    text = [[c[0] for c in row] for row in cells]
    tb = ax.table(cellText=text, rowLabels=row_labels, colLabels=[c[0] for c in cols], loc="upper center",
                  cellLoc="center", rowLoc="left", bbox=[label_w, 0.13, 0.995 - label_w, 0.74])
    tb.auto_set_font_size(False)
    tb.set_fontsize(fs)
    for (r, c), cl in tb.get_celld().items():
        cl.set_edgecolor(GRID)
        cl.set_linewidth(0.6)
        if r == 0 and c >= 0:
            cl.set_facecolor(HEAD[cols[c][1]])
            cl.set_text_props(fontweight="bold", color=INK)
            cl.set_height(cl.get_height() * 1.0)
        elif c == -1:
            cl.set_facecolor(HEAD["label"])
            cl.set_text_props(color=INK, ha="left")
            cl.PAD = 0.02
    for r, b in enumerate(better):
        vals = [v for _, v in cells[r]]
        ok = [v for v in vals if v is not None]
        if b is None or len(ok) < 2:
            continue
        best = min(ok) if b == "low" else max(ok)
        for c, v in enumerate(vals):
            if v is not None and np.isclose(v, best):
                tb[r + 1, c].set_text_props(fontweight="bold", color=INK)
    fig.text(0.01, 0.975, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.935, subtitle, fontsize=9.5, color=INK2, va="top")
    groups = [("proposed", "proposed: sync temporal block variants"),
              ("baseline", "baselines on the same frozen features (+ weight EMA)"), ("panecho", "PanEcho baselines")]
    x = 0.01
    for g, lab in groups:
        if any(c[1] == g for c in cols):
            fig.patches.append(plt.Rectangle((x, 0.903), 0.012, 0.016, transform=fig.transFigure,
                                             fc=HEAD[g], ec=MUTED, lw=0.6))
            fig.text(x + 0.016, 0.911, lab, fontsize=8.5, color=INK2, va="center")
            x += 0.035 + 0.0046 * len(lab)
    fig.text(0.01, 0.10, "\n".join(notes), fontsize=8, color=INK2, va="top", linespacing=1.5)
    return fig


def fmt(v, d=3):
    return "—" if v is None else f"{v:.{d}f}"


# ----------------------------------------------------------------------------- table 1

ACC = [("MAE (EF points) ↓", "MAE", "low", 3), ("RMSE ↓", "RMSE", "low", 3), ("R² ↑", "R2", "high", 3),
       ("Pearson r ↑", "Pearson", "high", 3), ("CCC ↑", "CCC", "high", 3), ("|bias| ↓", "bias", "low", 3),
       ("within 5 points (%) ↑", "within5", "high", 1), ("within 10 points (%) ↑", "within10", "high", 1),
       ("AUROC, EF < 40% ↑", "AUROC_lt40", "high", 3), ("AUROC, EF < 50% ↑", "AUROC_lt50", "high", 3)]


def acc_value(e, key):
    v = e[key]
    if key == "bias":
        return abs(v)
    return 100 * v if key.startswith("within") and v <= 1 else v


def table_accuracy(split, models=None, title="Table 1. EF accuracy of all models"):
    models = models or MODELS
    ref = ref_of(models)
    cols = [(m[0], m[1]) for m in models]
    ok = [available(m, split) for m in models]
    rows = (["trained parameters", "seeds"] + [r[0] for r in ACC] +
            ["MAE difference vs BiGRU ↓", "  95% CI of the difference", "p vs BiGRU (paired bootstrap)"])
    cells = [[] for _ in rows]
    eb = errors(ref[2], split)[1] if available(ref, split) else None
    for m, a in zip(models, ok):
        p = trained_params(m)
        cells[0].append(("—" if p is None else (f"{p / 1e6:.2f}M" if p >= 1e5 else f"{p:,d}"), None))
        cells[1].append((str(len(m[2])), None))
        if not a:
            for r in range(2, len(rows)):
                cells[r].append(("pending", None))
            continue
        evs = ef(m, split)
        for i, (_, key, _, d) in enumerate(ACC):
            v = float(np.mean([acc_value(e, key) for e in evs]))
            cells[2 + i].append((fmt(v, d), v))
        if eb is None or m is ref:
            cells[-3].append(("reference" if m is ref else "—", 0.0 if m is ref else None))
            cells[-2].append(("—", None))
            cells[-1].append(("—", None))
            continue
        e = errors([r + (f"@{m[3]}" if m[3] != 32 else "") for r in m[2]], split)[1]
        d, lo, hi, pv = paired(e, eb)
        cells[-3].append((f"{d:+.3f}", d))
        cells[-2].append((f"[{lo:+.3f}, {hi:+.3f}]", None))
        cells[-1].append(("< 0.001" if pv < 0.001 else f"{pv:.3f}", None))
    better = [None, None] + [r[2] for r in ACC] + ["low", None, None]
    notes = [f"{SPLIT_NAME[split].capitalize()} set ({'1,288' if split == 'val' else '1,277'} videos), EF averaged over "
             "4 evenly spaced clips per video, metrics averaged over seeds. Paired statistics over videos with "
             "seed-averaged absolute errors (10,000 bootstrap resamples).",
             "Our clips: T = 32 frames at stride 2 (63-frame window); PanEcho's native clips: 16 consecutive frames. "
             "All models except the zero-shot one are trained on EchoNet-Dynamic TRAIN only; design choices made on "
             "validation.",
             "Trained parameters: temporal model + head (the frozen PanEcho image encoder, 27.8M, is shared and "
             "excluded); ridge = 768 weights + bias. Bold = best in the row."]
    return draw_table(f"{title} — {SPLIT_NAME[split]} set",
                      "Every model uses the same frozen PanEcho ConvNeXt-T frame encoder.", rows, cols, cells, better,
                      notes, size=(18.5, 8.6), fs=8.2, label_w=0.15)


# ----------------------------------------------------------------------------- table 2

def table_behaviour(split, sel=None, title="Table 2. Temporal behaviour and cost",
                    subtitle="Proposed variants and baselines on the same frozen PanEcho 3-frame-stack features."):
    sel = sel or [m for m in MODELS if m[4] == "ours"] + [MODELS[9]]
    cols = [(m[0], m[1]) for m in sel]
    hr = pd.read_csv(config.RESULTS / "hr_reference.csv").set_index("FileName")["hr_bpm"]
    b = bench()
    rows, better, cells = [], [], []

    def add(label, direction, values, d=3):
        rows.append(label)
        better.append(direction)
        cells.append([(fmt(v, d) if not isinstance(v, str) else v, v if not isinstance(v, str) else None)
                      for v in values])

    evs = {m[0]: ([json.load(open(eval_file(r, "ours", split))) for r in m[2]] if m[4] == "ours" and
                  available(m, split) else None) for m in sel}

    def per(f):
        return [(float(np.mean([f(e) for e in evs[m[0]]])) if evs[m[0]] else
                 ("pending" if m[4] == "ours" else "—")) for m in sel]

    for T in ("8", "16", "32", "64", "128"):
        add(f"MAE at T = {T}{' (trained)' if T == '32' else ''} ↓", "low",
            per(lambda e, T=T: e["by_T"][T]["ef"]["MAE"]))
    split_hr = pd.read_csv(config.RESULTS / "hr_reference.csv")
    split_hr = split_hr[split_hr.Split == split.upper()]["hr_bpm"]
    for lab, lo_, hi_ in (("< 60", 0, 60), ("60–100", 60, 100), ("> 100", 100, 1e9)):
        vals = []
        for m in sel:
            if m[4] != "ours":
                vals.append("—")
                continue
            if not available(m, split):
                vals.append("pending")
                continue
            vids, E, _ = errors(m[2], split)
            h = hr.loc[vids].to_numpy()
            vals.append(float(E.mean(0)[(h >= lo_) & (h < hi_)].mean()))
        n = int(((split_hr >= lo_) & (split_hr < hi_)).sum())
        add(f"MAE, heart rate {lab} bpm (n = {n}) ↓", "low", vals)
    add("EF SD across a video's 4 clips ↓", "low", per(lambda e: e["by_T"]["32"]["ef_std_across_clips"]["mean_std"]))
    add("feature smoothness, cos(z_t, z_t+1) (descriptive)", None, per(lambda e: e["by_T"]["32"]["cosine_smoothness"]))
    add("periodicity, autocorrelation peak (descriptive)", None,
        per(lambda e: e["by_T"]["32"]["periodicity"]["autocorr_peak_mean"]))
    add("ED frame probe error (frames) ↓", "low", per(lambda e: e["probe"]["32"]["Z"]["test"]["ED"]["mean"]))
    add("ES frame probe error (frames) ↓", "low", per(lambda e: e["probe"]["32"]["Z"]["test"]["ES"]["mean"]))
    add("MAE with frames shuffled (higher = uses order)", None,
        per(lambda e: e["by_T"]["32"]["shuffled_ef"]["MAE"]))
    for lab, key, d in (("forward GFLOP per clip ↓", "fwd_gflops_per_clip", 3),
                        ("latency, ms per clip (batch 16) ↓", "infer_graph_ms_per_clip_b16", 3),
                        ("training step, ms (batch 16) ↓", "train_ms_per_step_b16", 1)):
        add(lab, "low", [(b[m[5]]["by_T"]["32"][key] if m[5] in b else "—") for m in sel], d)
    notes = [f"{SPLIT_NAME[split].capitalize()} set, mean of 3 seeds, 4 clips per video. Models trained once at T = 32 "
             "(stride 2) and evaluated at every T. Heart rate: label-free whole-video estimate from the frame "
             "features (scripts/hr_reference.py).",
             "Probes: ridge regression from the frozen block outputs to the ED/ES frame (fitted on TRAIN, lambda on "
             "validation); error in sampled frames. Smoothness and periodicity are descriptive, not accuracy.",
             "Cost: temporal model + head only (the shared frozen encoder is excluded), RTX 3050 laptop GPU, bf16, "
             "CUDA graphs; '—' = not measured or not applicable (PanEcho TF has no per-frame evaluation outputs)."]
    return draw_table(f"{title} — {SPLIT_NAME[split]} set",
                      subtitle, rows, cols,
                      cells, better, notes, size=(17.5, 9.6), fs=8.2, label_w=0.21)


# ----------------------------------------------------------------------------- table 3

def log_best(run):
    return min(json.loads(line)["val_MAE"] for line in open(config.RUNS / run / "log.jsonl"))


def four_clip(runs):
    return float(np.mean([json.load(open(config.RESULTS / r / "eval_VAL.json"))["by_T"]["32"]["ef"]["MAE"]
                          for r in runs]))


def table_explored():
    proto_rf = "single-frame PanEcho features; seed 0; best single-clip VAL MAE (training log)"
    proto_s3 = "3-frame-stack features, no EMA; 3 seeds; 4-clip VAL MAE"
    proto_ft = "3-frame-stack clips; seed 0; 4-clip VAL MAE (baselines frozen, EMA)"
    proto_e2e = "ResNet-18 trained end to end; seed 0; best single-clip VAL MAE (training log)"
    rf_b = log_best("pe_bigru_lr1e-3_s0")
    s3_b, s3_m = four_clip(s3("bigru")), four_clip(s3("mamba"))
    ft_b, ft_m = four_clip(["pe3_bigru_ema_s0"]), four_clip(["pe3_mamba_ema_s0"])
    e2_b, e2_m = log_best("bigru_T32_s0"), log_best("mamba_T32_s0")
    rows = [
        ("Readout heads", "sync, no readout", proto_rf, log_best("pe_sync_novol_s0"), rf_b, None),
        ("", "sync + volumetric readout", proto_rf, log_best("pe_sync_lr1e-3_s0"), rf_b, None),
        ("", "sync + harmonic + range readouts", proto_rf, log_best("pe_sync_harm_range_s0"), rf_b, None),
        ("", "BiGRU + harmonic + range readouts", proto_rf, log_best("pe_bigru_harm_range_s0"), rf_b, None),
        ("", "sync, no readout", proto_s3, four_clip(s3("sync_novol")), s3_b, s3_m),
        ("", "sync + volumetric readout", proto_s3, four_clip(s3("sync")), s3_b, s3_m),
        ("Encoder fine-tuning", "BiGRU, encoder stage 4 fine-tuned", proto_ft, four_clip(["ft_bigru_ft_s0"]), ft_b,
         ft_m),
        ("End to end", "sync (hr clock) + latent volume", proto_e2e, log_best("sync_hr_vol_T32_s0"), e2_b, e2_m),
        ("", "sync (hr clock) + area readout", proto_e2e, log_best("sync_hr_area_T32_s0"), e2_b, e2_m),
    ]
    fig = plt.figure(figsize=(17.5, 7.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    text = [[g, v, p, f"{x:.3f}", f"{b:.3f}", "—" if m is None else f"{m:.3f}", f"{x - b:+.3f}"]
            for g, v, p, x, b, m in rows]
    tb = ax.table(cellText=text, colLabels=["family", "variant", "protocol (same for the variant and its baselines)",
                                            "variant", "BiGRU", "Mamba", "Δ vs BiGRU"],
                  loc="upper center", cellLoc="center", colWidths=[0.11, 0.2, 0.39, 0.075, 0.075, 0.075, 0.075],
                  bbox=[0.01, 0.22, 0.98, 0.64])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.5)
    for (r, c), cl in tb.get_celld().items():
        cl.set_edgecolor(GRID)
        cl.set_linewidth(0.6)
        if r == 0:
            cl.set_facecolor(HEAD["label"])
            cl.set_text_props(fontweight="bold")
        elif c in (0, 1, 2):
            cl.set_text_props(ha="left")
            cl.PAD = 0.02
        if r > 0 and c == 6:
            d = rows[r - 1][3] - rows[r - 1][4]
            cl.set_text_props(fontweight="bold" if d < 0 else "normal")
    fig.text(0.01, 0.975, "Table 3. Explored variants vs BiGRU and Mamba under matched protocols — validation set",
             fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.93, "MAE in EF points (lower is better). Each variant is compared only with baselines trained and "
             "evaluated exactly as it was; values are not comparable across protocols.", fontsize=9.5, color=INK2,
             va="top")
    fig.text(0.01, 0.17, "\n".join([
        "Readout heads: 'sync + range' alone did not complete (1 of 40 epochs) and is omitted. Δ in bold = better "
        "than BiGRU. Single-seed rows carry seed noise of about ±0.03-0.05 MAE.",
        "Encoder fine-tuning: best epoch is the first; validation error rose in every later epoch "
        "(5_encoder_finetuning). End to end: the first project track, before the frozen-PanEcho features.",
        "None of these variants was adopted; the proposed model is the sync block with a fixed learned frequency on "
        "frozen 3-frame-stack features (Tables 1-2)."]), fontsize=8, color=INK2, va="top", linespacing=1.5)
    return fig


# ----------------------------------------------------------------------------- end-to-end baselines

def table_e2e():
    """First-track baselines (ResNet-18 trained end to end, seed 0): VAL from the training log, TEST from the
    evaluation run at the end of that track (4 clips per video)."""
    runs = [("B9  BiGRU", "bigru_T32_s0"), ("B10  Mamba", "mamba_T32_s0"), ("B11  Transformer (UVT)", "uvt_T32_s0")]
    text, vals = [], []
    for lab, r in runs:
        info = json.load(open(config.RUNS / r / "info.json"))
        ev = json.load(open(config.RESULTS / r / "eval.json"))["by_T"]
        e = ev["32"]["ef"]
        row = [lab, f"{info['params_total'] / 1e6:.2f}M", f"{log_best(r):.3f}", f"{e['MAE']:.3f}", f"{e['RMSE']:.3f}",
               f"{e['R2']:.3f}"] + [f"{ev[T]['ef']['MAE']:.3f}" for T in ("8", "16", "64", "128")]
        text.append(row)
        vals.append([None, None, log_best(r), e["MAE"], e["RMSE"], -e["R2"]] +
                    [ev[T]["ef"]["MAE"] for T in ("8", "16", "64", "128")])
    fig = plt.figure(figsize=(17.5, 5.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    tb = ax.table(cellText=text, colLabels=["model", "parameters\n(encoder + temporal\n+ head)",
                                            "VAL MAE\n(best single clip)", "TEST MAE", "TEST RMSE", "TEST R²",
                                            "TEST MAE\nT = 8", "TEST MAE\nT = 16", "TEST MAE\nT = 64",
                                            "TEST MAE\nT = 128"],
                  loc="upper center", cellLoc="center", bbox=[0.01, 0.3, 0.98, 0.52])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.8)
    for (r, c), cl in tb.get_celld().items():
        cl.set_edgecolor(GRID)
        cl.set_linewidth(0.6)
        if r == 0:
            cl.set_facecolor(HEAD["baseline"])
            cl.set_text_props(fontweight="bold")
        elif c == 0:
            cl.set_text_props(ha="left")
            cl.PAD = 0.03
    for c in range(2, len(vals[0])):
        best = min(v[c] for v in vals)
        for r, v in enumerate(vals):
            if np.isclose(v[c], best):
                tb[r + 1, c].set_text_props(fontweight="bold")
    fig.text(0.01, 0.96, "Table B3. End-to-end ResNet-18 baselines (first project track, seed 0)", fontsize=14,
             fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.89, "4 clips per video on TEST (T = 32 at stride 2 unless stated); VAL = best single-clip MAE "
             "of the training log. Lower is better; bold = best.", fontsize=9.5, color=INK2, va="top")
    fig.text(0.01, 0.24, "Trained before the frozen-PanEcho track with lr 1e-4, 25 epochs, pad-crop augmentation and "
             "no weight EMA; these numbers are not comparable with Tables B1-B2, where every model shares the frozen "
             "PanEcho encoder.", fontsize=8, color=INK2, va="top")
    return fig
