r"""Architecture diagrams of every non-baseline model in this project (PNG 300 dpi + vector PDF).

Drawn from the code (cardiac/sync.py, cardiac/models.py, cardiac/panecho.py), dimensions as trained:
  1_pipeline        final model: 3-frame stacks -> frozen PanEcho ConvNeXt-T -> sync block -> pool -> head
  2_sync_block      SyncTemporal (2 layers) and the SyncMixer inside each layer
  3_clock_variants  where the rotation frequency comes from: fixed (final), hr, tempered, learned, none
  4_readouts        explored readout heads on the block output: volumetric, range, harmonic
  5_finetune        PanEcho encoder fine-tuning (stage 4 trainable) + warm-started temporal block
  6_end_to_end      first track: ResNet-18 trained end to end + sync (+ latent-volume / area readouts)
VAL MAE values on the clock panel are read from results/stats_val_pe3_ema_panel.json and
stats_val_pe3_tempered.json (3 seeds, 4 clips per video, weight EMA).

Usage: py.cmd scripts\draw_architectures.py [--out E:\echo_project\figures\architecture]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

from cardiac import config  # noqa: E402

FILL = {"frozen": "#DCE3EA", "train": "#FBE3C8", "op": "#FFFFFF", "data": "#F2F2F2", "final": "#D6EFD8"}
EDGE = {"frozen": "#5B7083", "train": "#C77A2A", "op": "#7A7A7A", "data": "#9A9A9A", "final": "#3C8D4A"}
LABEL = {"frozen": "frozen (pretrained)", "train": "trained", "op": "no parameters", "data": "tensor",
         "final": "proposed block (trained)"}
INK, MUTED = "#222222", "#555555"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "mathtext.fontset": "dejavusans"})
U = 36.0          # points per data unit: every canvas uses 1 unit = 0.5 inch, so text sizes are predictable


def canvas(w, h):
    """w x h data units (0.5 inch each), axes filling the figure."""
    fig = plt.figure(figsize=(w / 2, h / 2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def text_size(text, fs):
    lines = text.split("\n")
    n = max(len(re.sub(r"\$[^$]*\$", lambda m: "x" * (len(m.group(0)) // 2), ln)) for ln in lines)
    return n * fs * 0.52 / U, len(lines) * fs * 1.42 / U


class Box:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    l = property(lambda s: (s.x - s.w / 2, s.y))   # noqa: E731,E741
    r = property(lambda s: (s.x + s.w / 2, s.y))
    t = property(lambda s: (s.x, s.y + s.h / 2))
    b = property(lambda s: (s.x, s.y - s.h / 2))


def box(ax, x, y, text, kind="train", fs=9, w=None, h=None, bold=False, minw=0.0, minh=0.0, ty=0.0):
    tw, th = text_size(text, fs)
    w = w or max(minw, tw + 0.8)
    h = h or max(minh, th + 0.6)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.18",
                                fc=FILL[kind], ec=EDGE[kind], lw=1.3, zorder=2))
    ax.text(x, y + ty, text, ha="center", va="center", fontsize=fs, color=INK, zorder=3, linespacing=1.4,
            fontweight="bold" if bold else "normal")
    return Box(x, y, w, h)


def row(ax, y, specs, x0, gap, minh=0.0):
    """Boxes left to right; specs: (text, kind, kwargs). Returns the boxes."""
    out, x = [], x0
    for text, kind, kw in specs:
        tw, th = text_size(text, kw.get("fs", 9))
        w = kw.pop("w", None) or max(kw.pop("minw", 0.0), tw + 0.8)
        b = box(ax, x + w / 2, y, text, kind, w=w, minh=minh, **kw)
        out.append(b)
        x += w + gap
    return out


def arrow(ax, p, q, label=None, fs=8, dx=0.0, dy=0.32, color=INK):
    ax.annotate("", xy=q, xytext=p, zorder=4,
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color=color, shrinkA=0, shrinkB=0, mutation_scale=11))
    if label:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, label, fontsize=fs, ha="center", va="center",
                color=color, zorder=5)


def path(ax, pts, label=None, at=0, fs=8, dx=0.0, dy=0.0, rot=0, color=INK):
    """Orthogonal polyline through pts with an arrow head on the last segment; label next to segment `at`."""
    xs, ys = zip(*pts[:-1])
    ax.plot(xs, ys, color=color, lw=1.2, zorder=4, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], color=color)
    if label:
        (x0, y0), (x1, y1) = pts[at], pts[at + 1]
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, label, fontsize=fs, ha="center", va="center",
                rotation=rot, color=color, zorder=5, bbox=dict(fc="white", ec="none", pad=0.5))


def group(ax, x0, y0, x1, y1, label, color):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.3",
                                fc="none", ec=color, lw=1.2, ls=(0, (4, 3)), zorder=1))
    ax.text(x0 + 0.25, y1 - 0.25, label, ha="left", va="top", fontsize=9, color=color, fontweight="bold")


def header(ax, w, h, title, kinds=None):
    ax.text(0.3, h - 0.45, title, fontsize=13, fontweight="bold", va="center", color=INK)
    if kinds:
        ax.legend(handles=[Patch(fc=FILL[k], ec=EDGE[k], label=LABEL[k]) for k in kinds], loc="upper right",
                  bbox_to_anchor=(1.0, 1.0), ncol=len(kinds), frameon=False, fontsize=8.5, handlelength=1.4,
                  columnspacing=1.6, borderaxespad=0.3)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


# ----------------------------------------------------------------------------- 1 pipeline

def fig_pipeline(out):
    W, H = 36, 7.8
    fig, ax = canvas(W, H)
    header(ax, W, H, "Proposed model: frozen PanEcho frame encoder + sync temporal block",
           ("frozen", "final", "train", "op", "data"))
    bs = row(ax, 4.3, [
        ("Echo clip\nT = 32 frames, stride 2\n(63-frame window)\n112 × 112 grey", "data", {}),
        ("3-frame stack\nchannels = frames\nt−1, t, t+1\nresize 256 → crop 224\nImageNet normalisation", "op", {}),
        ("PanEcho ConvNeXt-T\nimage encoder\nfrozen, 27.8M params\n→ 768-d per frame", "frozen", {}),
        ("Sync temporal block\nbidirectional SSM\nwith rotating state\n1.09M params", "final", {"bold": False}),
        ("Mean over\nreal frames\n(padding excluded)", "op", {}),
        ("MLP head\n256 → 256 → GELU → 1\n66k params", "train", {}),
        ("EF (%)", "data", {"bold": True, "minw": 2.0}),
    ], x0=0.4, gap=1.75, minh=3.0)
    labels = ["", "(T, 3, 224, 224)", "(T, 768)", "(T, 256)", "256", ""]
    for a, b, lab in zip(bs, bs[1:], labels):
        arrow(ax, a.r, b.l, lab or None, fs=7.5, dy=0.35)
    ax.text(bs[2].x, 2.3, "features cached once\n(the encoder never trains)", ha="center", va="top", fontsize=8,
            color=EDGE["frozen"], style="italic")
    ax.text(bs[3].x, 2.3, "details: 2_sync_block,\n3_clock_variants", ha="center", va="top", fontsize=8,
            color=EDGE["final"], style="italic")
    ax.text(0.4, 0.55, "Training: MSE on standardised EF, AdamW lr 1e-3 (cosine, 5% warm-up), batch 16, 40 epochs; "
            "weight EMA 0.999 for validation, checkpoint selection and evaluation.   "
            "Evaluation: mean EF over 4 evenly spaced clips per video.", fontsize=8.5, color=MUTED)
    save(fig, out, "1_pipeline")


# ----------------------------------------------------------------------------- 2 sync block

def fig_sync_block(out):
    W, H = 36, 23
    fig, ax = canvas(W, H)
    header(ax, W, H, "Sync temporal block (proposed)", ("train", "final", "op", "data"))

    # (a) SyncTemporal stack
    ax.text(0.4, 21.2, "(a) SyncTemporal — 2 layers", fontsize=11, fontweight="bold")
    xa = 6.2
    f = box(ax, xa, 1.4, "f : (T, 768) frame features", "data", minw=5.6)
    inp = box(ax, xa, 3.3, "Linear 768 → 256", "train", minw=4.6)
    group(ax, 3.3, 4.6, 9.9, 12.4, "× 2 layers", EDGE["train"])
    n1 = box(ax, xa, 6.0, "RMSNorm", "train", minw=4.0)
    mx = box(ax, xa, 8.3, "SyncMixer\n(panel b)", "final", bold=True, minw=4.0, minh=1.8)
    add = box(ax, xa, 10.9, "+", "op", fs=13, w=0.9, h=0.9)
    n2 = box(ax, xa, 13.7, "RMSNorm", "train", minw=4.0)
    z = box(ax, xa, 15.6, "z : (T, 256) → mean pool → head", "data", minw=5.6)
    for a, b in ((f, inp), (inp, n1), (n1, mx), (mx, add), (add, n2), (n2, z)):
        arrow(ax, a.t, b.b)
    path(ax, [(xa, 5.15), (8.9, 5.15), (8.9, 10.9), add.r], "residual", at=1, rot=90, dx=0.0, fs=7.5,
         color=MUTED)
    clk = box(ax, xa, 18.6, "Clock: angular frequency ω per clip\nfinal model: ω = 2π / P, one learned period P\n"
              "(other variants: 3_clock_variants)", "train", fs=8.5)
    rot = box(ax, 1.5, 11.6, "cos, sin\nof k_j·ω·t\n(T, 4)\n\nshared by\nboth layers", "op", fs=7.5, w=2.2)
    path(ax, [clk.l, (1.5, clk.y), rot.t])
    path(ax, [rot.b, (1.5, mx.y), mx.l])

    # (b) SyncMixer
    ax.text(12.2, 21.2, "(b) SyncMixer — one layer: d_model 256, d_inner 512, 16 heads × 32 per direction, "
            "state N = 16", fontsize=11, fontweight="bold")
    xm, xs = 19.2, 23.6
    u = box(ax, 21.4, 1.4, "u : (T, 256)", "data", minw=4.0)
    ip = box(ax, 21.4, 3.3, "in_proj: Linear 256 → 2 × 512  (split into x and z)", "train")
    cv = box(ax, xm, 5.5, "Depthwise Conv1d over time\nk = 3, replicate padding, SiLU", "train", w=7.4)
    xp = box(ax, xm, 7.9, "x_proj: Linear 512 → 96\nB_fwd, C_fwd, B_bwd, C_bwd (16 each)\nΔt: 16 heads × 2 directions",
             "train", fs=8.5, w=7.4)
    rt = box(ax, xm, 10.5, "Rotate state pairs 1–4 of each B and C\nby angle k_j·ω·t   (k = 1, 2, 1, 2)\n"
             "pairs 5–8 not rotated", "op", fs=8.5, w=7.4)
    ss = box(ax, xs, 14.0, "Bidirectional selective SSM, exact matrix form (float32)\n"
             "forward: causal (s ≤ t)          backward: anti-causal (s > t)\n"
             "Δt = softplus(· + bias),   A = −exp(A_log): one decay per head and direction",
             "train", fs=8.5, w=14.2, h=3.4, ty=0.55)
    ax.text(xs, ss.y - 0.95, r"$y_t = \sum_s L_{ts}\,\Delta_s\,x_s\,[\,C^0_t \cdot B^0_s + \sum_j C^j_t \cdot "
            r"R(k_j\,\omega\,(s-t))\,B^j_s\,] + D\,x_t$", ha="center", va="center", fontsize=10, zorder=3)
    gt = box(ax, xs, 17.0, "y ⊙ SiLU(z)", "op", minw=4.0)
    on = box(ax, xs, 18.8, "RMSNorm → out_proj: Linear 512 → 256", "train")
    ax.text(xs, 20.0, "output (T, 256) → residual add in panel (a)", ha="center", fontsize=8.5, color=MUTED)
    arrow(ax, u.t, ip.b)
    arrow(ax, (xm, ip.b[1] + ip.h), cv.b, "x", dx=-0.35, dy=0.0)
    arrow(ax, cv.t, xp.b)
    arrow(ax, xp.t, rt.b, "B, C", dx=0.7, dy=0.0)
    arrow(ax, rt.t, (xm, ss.b[1]))
    path(ax, [cv.r, (27.0, cv.y), (27.0, ss.b[1])], "x (values)", at=1, rot=90, fs=7.5)
    path(ax, [xp.r, (24.6, xp.y), (24.6, ss.b[1])], "Δt", at=1, fs=8)
    path(ax, [ip.r, (32.6, ip.y), (32.6, gt.y), gt.r], "z (gate)", at=1, rot=90, fs=8)
    path(ax, [(12.4, rt.y), rt.l], "cos, sin\nfrom the clock", at=0, dy=0.6, fs=7.5)
    arrow(ax, ss.t, gt.b)
    arrow(ax, gt.t, on.b)
    save(fig, out, "2_sync_block")


# ----------------------------------------------------------------------------- 3 clocks

def val_mae():
    d = {}
    for f in ("stats_val_pe3_ema_panel.json", "stats_val_pe3_tempered.json"):
        p = config.RESULTS / f
        if p.exists():
            for m, v in json.loads(p.read_text())["models"].items():
                d[m.split(",")[0].replace("pe3_", "").replace("_ema_s0", "")] = v["MAE_mean_over_seeds"]
    return d


def fig_clocks(out):
    W, H = 36, 12.4
    mae = val_mae()
    fig, ax = canvas(W, H)
    header(ax, W, H, "Where the rotation frequency of the sync block comes from", ("final", "train", "op"))
    cols = [
        ("fixed  (final)", "sync_novol_fixed", "final",
         "ω = 2π / P\nP: one learned period\n(init 24 sampled frames)\nthe same for every clip",
         "rotation of state pairs\ncos, sin(k_j·ω·t), k = 1, 2, 1, 2",
         "an LTI complex-diagonal SSM;\nno heart-rate estimate needed\n(robust on short clips)"),
        ("hr", "sync_novol", "train",
         "ω = periodogram peak of the\nclip's own frame features\n(period grid 6–48 frames,\nlabel-free, no gradient)",
         "rotation of state pairs\ncos, sin(k_j·ω·t), k = 1, 2, 1, 2",
         "adapts to each heart rate;\nneeds ≥ ~1 beat in the clip"),
        ("tempered", "sync_novol_tempered", "train",
         "ω = (2π / P) · (ω_hr / ω_ref)^γ\nγ = sigmoid(learned), init 0.5\nω_ref = 2π / 22.5 (median)\n"
         "clips < 24 frames: ratio = 1",
         "rotation of state pairs\ncos, sin(k_j·ω·t), k = 1, 2, 1, 2",
         "γ = 0 → fixed,  γ = 1 → hr\n(the learned γ stayed ≈ 0.5)"),
        ("learned  (Mamba-3 style)", "sync_novol_learned", "train",
         "angle_t = Σ_{s ≤ t} π·tanh(W x_s)\none angle per state pair,\nfrom the local input\n(data-dependent, per step)",
         "rotation of state pairs\nby the accumulated angle",
         "replaces k_j·ω·t"),
        ("none", "sync_novol_noclk", "op",
         "no rotation\n(real decay only,\nMamba-2-like SSD)",
         "identity\n(no rotation)",
         "same parameters as\nthe other variants"),
    ]
    cw, x0 = 6.6, 0.6
    for i, (name, key, kind, how, rot, note) in enumerate(cols):
        x = x0 + cw / 2 + i * (cw + 0.5)
        ax.text(x, 10.6, name, ha="center", fontsize=11, fontweight="bold",
                color=EDGE["final"] if kind == "final" else INK)
        b = box(ax, x, 8.0, how, kind, fs=8.5, w=cw, h=3.2)
        r = box(ax, x, 5.0, rot, "op", fs=8.5, w=cw, h=1.5)
        arrow(ax, b.b, r.t)
        ax.text(x, 3.75, note, ha="center", va="top", fontsize=8, color=MUTED, style="italic")
        if key in mae:
            ax.text(x, 1.4, f"VAL MAE {mae[key]:.3f}", ha="center", fontsize=10.5, fontweight="bold",
                    color=EDGE["final"] if kind == "final" else INK)
    ax.text(0.6, 0.4, "VAL MAE: 3 seeds, 4 clips per video, frozen PanEcho 3-frame-stack features, weight EMA. "
            "Everything else in the block is identical across the five variants.", fontsize=8, color=MUTED)
    save(fig, out, "3_clock_variants")


# ----------------------------------------------------------------------------- 4 readouts

def fig_readouts(out):
    W, H = 36, 14.2
    fig, ax = canvas(W, H)
    header(ax, W, H, "Explored readout heads on the sync block output", ("train", "op", "data"))
    ax.text(0.6, 12.6, "Each attaches to the block output z (after the final RMSNorm) and adds a vector to every "
            "frame, so the shared mean pool + head can use it.  Explored; none was adopted (no VAL gain).",
            fontsize=8.5, color=MUTED)
    specs = [
        ("Volumetric readout (vol_readout)",
         [("l_t = w · z_t   (log-volume curve)", "train"),
          ("soft max and soft min of l over the clip\n(learned temperature, 2 edge frames trimmed)", "op"),
          ("EF_phys = 1 − exp(−(softmax l − softmin l))\n= 1 − ESV / EDV", "op"),
          ("z_t + e(EF_phys) · v   (every frame)", "op")],
         "auxiliary loss MSE(EF_phys, EF label)\n(EF labels only, no ED/ES)"),
        ("Range readout (range_k = 16)",
         [("c_t = W z_t   (16 channels)", "train"),
          ("per channel: soft max and soft min\nover the clip (learned temperature)", "op"),
          ("[range, midrange] → Linear 32 → 256\n(zero-initialised)", "train"),
          ("z_t + g   (every frame)", "op")],
         "EF is a ratio of temporal extremes;\na linear SSM cannot take a maximum"),
        ("Harmonic readout (harm_k = 16)",
         [("c_t = W z_t   (16 channels)", "train"),
          ("least squares per channel at the clip's ω:\na₀ + Σ_h a_h cos(hωt) + b_h sin(hωt),  h = 1, 2", "op"),
          ("amplitudes √(a_h² + b_h²) → Linear 32 → 256\n(zero-initialised)", "train"),
          ("z_t + g   (every frame)", "op")],
         "phase-invariant oscillation amplitude\nat the heart rate"),
    ]
    cw = 10.6
    for i, (title, steps, note) in enumerate(specs):
        x = 0.6 + cw / 2 + i * (cw + 1.0)
        ax.text(x, 11.5, title, ha="center", fontsize=10.5, fontweight="bold")
        prev = box(ax, x, 10.4, "z : (T, 256)", "data", minw=3.6)
        for j, (s, kind) in enumerate(steps):
            b = box(ax, x, 8.7 - j * 1.95, s, kind, fs=8.5, w=cw, h=1.35)
            arrow(ax, prev.b, b.t)
            prev = b
        ax.text(x, 1.45, note, ha="center", va="top", fontsize=8, color=MUTED, style="italic")
    save(fig, out, "4_readouts")


# ----------------------------------------------------------------------------- 5 fine-tuning

def fig_finetune(out):
    W, H = 36, 9.4
    fig, ax = canvas(W, H)
    header(ax, W, H, "Encoder fine-tuning variant (explored)", ("frozen", "train", "data"))
    y = 4.9
    group(ax, 6.4, 2.4, 22.6, 7.7, "PanEcho ConvNeXt-T (weights from the release)", EDGE["frozen"])
    b0 = box(ax, 3.2, y, "3-frame-stack clip\n(T, 3, 224, 224) uint8\nT = 32, stride 2", "data", minh=2.6)
    b1 = box(ax, 10.3, y, "stem + stages 1–3\n(15 ConvNeXt blocks)\nfrozen: no gradient, eval mode\n"
             "not stored in checkpoints", "frozen", fs=8.5, minh=2.6)
    b2 = box(ax, 18.6, y, "downsample + stage 4\n(3 ConvNeXt blocks)\n+ avg pool + LayerNorm\n"
             "trainable: 15.5M params, lr 3e-5", "train", fs=8.5, minh=2.6)
    b3 = box(ax, 26.4, y, "temporal block\n(sync or BiGRU)\nwarm-started from the\nfrozen-feature run", "train", fs=8.5,
             minh=2.6)
    b4 = box(ax, 31.4, y, "mean pool\n+ MLP head", "train", fs=8.5, minh=2.6)
    b5 = box(ax, 34.6, y, "EF (%)", "data", bold=True, minw=1.9)
    arrow(ax, b0.r, b1.l)
    arrow(ax, b1.r, b2.l, "(T, 384,\n14, 14)", fs=7.5, dy=0.55)
    arrow(ax, b2.r, b3.l, "(T, 768)", fs=7.5, dx=0.9, dy=0.35)
    arrow(ax, b3.r, b4.l)
    arrow(ax, b4.r, b5.l)
    ax.text(0.4, 1.0, "Linear probe, then fine-tune (LP-FT): head lr 1e-4, encoder lr 3e-5, 5 epochs, weight EMA 0.999.  "
            "Explored and not adopted: validation error rose every epoch (the encoder overfits 7,465 training videos).",
            fontsize=8.5, color=MUTED)
    save(fig, out, "5_finetune")


# ----------------------------------------------------------------------------- 6 end to end

def fig_end_to_end(out):
    W, H = 36, 12.2
    fig, ax = canvas(W, H)
    header(ax, W, H, "End-to-end variants: ResNet-18 + sync block (+ latent-volume / area readouts)",
           ("train", "final", "data"))
    y = 8.0
    b0 = box(ax, 2.6, y, "Echo clip\nT = 32, stride 2\n112 × 112 RGB", "data", minh=2.6)
    b1 = box(ax, 9.6, y, "ResNet-18 (ImageNet init)\ntrained end to end\nstem + 4 stages →\nglobal avg pool → 512-d",
             "train", fs=8.5, minh=2.6)
    b2 = box(ax, 17.0, y, "Sync temporal block\nclock = hr\n512 → 256, 2 layers", "final", fs=8.5, minh=2.6)
    b3 = box(ax, 23.6, y, "volumetric readout\nlatent variant:\nl_t = w · z_t", "train", fs=8.5, minh=2.6)
    b4 = box(ax, 29.2, y, "mean pool\n+ MLP head", "train", fs=8.5, minh=2.6)
    b5 = box(ax, 33.6, y, "EF (%)", "data", bold=True, minw=1.9)
    arrow(ax, b0.r, b1.l)
    arrow(ax, b1.r, b2.l, "(T, 512)", fs=7.5, dy=0.35)
    arrow(ax, b2.r, b3.l, "z", fs=8, dy=0.35)
    arrow(ax, b3.r, b4.l)
    arrow(ax, b4.r, b5.l)
    a1 = box(ax, 9.6, 3.6, "stage-3 feature map\n(T, 256, 7, 7)", "data", fs=8.5, minw=5.4)
    a2 = box(ax, 17.0, 3.6, "1×1 conv → sigmoid\nsoft LV mask, area A_t", "train", fs=8.5)
    a3 = box(ax, 23.6, 3.6, "l_t = γ·log A_t\n+ correction from z", "train", fs=8.5)
    arrow(ax, b1.b, a1.t)
    arrow(ax, a1.r, a2.l)
    arrow(ax, a2.r, a3.l)
    arrow(ax, a3.t, b3.b, "area variant", fs=8, dx=1.3, dy=0.0)
    ax.text(29.0, 4.1, "EF_phys = 1 − exp(−(softmax l − softmin l))\nauxiliary loss MSE(EF_phys, EF label)\n"
            "γ learned (init 1.5: area-length V ∝ A^1.5)", ha="center", va="center", fontsize=8, color=MUTED,
            style="italic")
    ax.text(0.4, 0.8, "First track (before the frozen-PanEcho track): same clip protocol, pooling and head as the "
            "proposed model; the area variant adds the mask branch.", fontsize=8.5, color=MUTED)
    save(fig, out, "6_end_to_end")


# ----------------------------------------------------------------------------- baselines

def fig_baseline_blocks(out):
    W, H = 36, 21
    fig, ax = canvas(W, H)
    header(ax, W, H, "Baselines B1-B4: temporal models on the same frozen PanEcho features",
           ("frozen", "train", "op", "data"))
    bs = row(ax, 18.2, [
        ("Echo clip\nT = 32, stride 2", "data", {"fs": 8.5}),
        ("3-frame stack\n(t−1, t, t+1)", "op", {"fs": 8.5}),
        ("PanEcho ConvNeXt-T\nfrozen, 27.8M", "frozen", {"fs": 8.5}),
        ("temporal model\n(B1-B4 below)", "train", {"fs": 8.5, "bold": True}),
        ("mean over\nreal frames", "op", {"fs": 8.5}),
        ("MLP head\n256 → 256 → 1", "train", {"fs": 8.5}),
        ("EF (%)", "data", {"bold": True, "minw": 1.9}),
    ], x0=2.2, gap=1.6, minh=2.0)
    for a, b in zip(bs, bs[1:]):
        arrow(ax, a.r, b.l)
    ax.text(18.0, 16.5, "identical pipeline, recipe and evaluation for every model (same as the proposed model); "
            "only the temporal model changes", ha="center", fontsize=8.5, color=MUTED, style="italic")
    cols = [
        ("B1  BiGRU", "1.15M parameters (with head)", [
            ("f : (T, 768)", "data"), ("Linear 768 → 256", "train"),
            ("bidirectional GRU, 3 layers\nhidden 128 per direction\n(cuDNN, float32)", "train"),
            ("LayerNorm", "train"), ("z : (T, 256)", "data")]),
        ("B2  Mamba (bidirectional)", "1.14M parameters", [
            ("f : (T, 768)", "data"), ("Linear 768 → 256", "train"),
            ("× 2 layers:  RMSNorm →\nforward Mamba mixer\n+ time-reversed Mamba mixer\n→ + residual", "train"),
            ("RMSNorm", "train"), ("z : (T, 256)", "data")]),
        ("B3  Transformer (UVT-style)", "1.19M parameters", [
            ("f : (T, 768)", "data"), ("Linear 768 → 256\n+ learned positions, LayerNorm", "train"),
            ("× 2 post-LN BERT layers\n4 heads, FFN 320, dropout 0.1", "train"),
            ("mean of all hidden states\n(input + every layer), LayerNorm", "op"), ("z : (T, 256)", "data")]),
        ("B4  No temporal model", "0.26M parameters", [
            ("f : (T, 768)", "data"), ("Linear 768 → 256", "train"),
            ("(no mixing over time:\nevery frame on its own)", "op"),
            ("LayerNorm", "train"), ("z : (T, 256)", "data")]),
    ]
    cw = 8.0
    for i, (title, sub, steps) in enumerate(cols):
        x = 0.6 + cw / 2 + i * (cw + 0.9)
        ax.text(x, 15.2, title, ha="center", fontsize=11, fontweight="bold")
        ax.text(x, 14.55, sub, ha="center", fontsize=8.5, color=MUTED)
        prev = None
        ys = [13.4, 11.7, 9.2, 6.6, 4.9]
        for (s, kind), y in zip(steps, ys):
            b = box(ax, x, y, s, kind, fs=8.5, w=cw - 0.6 if kind != "data" else 3.6,
                    h=2.2 if s.count("\n") >= 2 else None)
            if prev is not None:
                arrow(ax, prev.b, b.t)
            prev = b
    mm = ("Mamba mixer (B2, one direction): in_proj 256 → 2 × 256 (expand 1) → causal depthwise conv (k = 4) + SiLU → "
          "x_proj → Δ (rank 16), B, C (state N = 16) → selective scan\n(S6, A = −exp(A_log), S4D-real init, + D skip) "
          "→ y ⊙ SiLU(z) → out_proj 256 → 256.   The backward mixer sees the clip reversed in time; the two are summed.")
    ax.text(0.6, 2.6, mm, fontsize=8.3, color=MUTED, va="top", linespacing=1.5)
    ax.text(0.6, 1.0, "Training as the proposed model: MSE on standardised EF, AdamW lr 1e-3, cosine, batch 16, "
            "40 epochs, weight EMA 0.999; 3 seeds; evaluation on 4 clips per video.", fontsize=8.3, color=MUTED)
    save(fig, out, "B1_temporal_baselines")


def fig_panecho_baselines(out):
    W, H = 36, 16
    fig, ax = canvas(W, H)
    header(ax, W, H, "Baselines B5-B8: PanEcho's own video model (Holste et al., JAMA 2025)",
           ("frozen", "train", "op", "data"))
    rows_ = [
        ("B5  zero-shot (as released)", "data", "16 consecutive frames\nreleased packing", "frozen",
         "PanEcho temporal Transformer\n(frozen, as released)", "frozen", "released EF head\nDropout 0.25 + Linear",
         "no training on EchoNet-Dynamic"),
        ("B6  ridge probe", "data", "16 consecutive frames\nreleased packing", "frozen",
         "PanEcho temporal Transformer\n(frozen) → 768-d embedding", "train", "ridge regression\n768 → EF (λ on VAL)",
         "linear probe of the frozen video embedding"),
        ("B7  fine-tuned, native clips", "data", "16 consecutive frames\nreleased packing", "train",
         "PanEcho temporal Transformer\n(init: release, trained)", "train", "EF head\n(init: release, trained)",
         "lr 1e-4 (best of 3e-5 / 1e-4 / 3e-4 on VAL), EMA"),
        ("B8  fine-tuned, our clips", "data", "T = 32, stride 2, 3-frame\nstacks (our protocol)", "train",
         "PanEcho temporal Transformer\npositions extended to 32", "train", "mean over real frames\n+ EF head (trained)",
         "lr 3e-4 (best of 4 on VAL; 1e-3 diverged), EMA"),
    ]
    enc_note = ("shared, frozen: PanEcho ConvNeXt-T image encoder (27.8M) — per-frame 768-d features, cached once.   "
                "Transformer: 4 post-norm layers, d 768, 8 heads, FFN 768, ReLU, sinusoidal positions, mean over time "
                "(14.2M parameters).")
    ax.text(0.6, 14.2, enc_note, fontsize=8.5, color=MUTED)
    for i, (name, k0, clip, k1, tf, k2, head, note) in enumerate(rows_):
        y = 12.0 - i * 3.1
        ax.text(0.6, y, name, fontsize=10.5, fontweight="bold", va="center")
        b0 = box(ax, 10.2, y, clip, k0, fs=8.5, w=5.4, h=1.9)
        b1 = box(ax, 16.8, y, "frozen\nencoder", "frozen", fs=8.5, w=2.6, h=1.9)
        b2 = box(ax, 22.4, y, tf, k1, fs=8.5, w=6.0, h=1.9)
        b3 = box(ax, 29.0, y, head, k2, fs=8.5, w=4.8, h=1.9)
        b4 = box(ax, 34.2, y, "EF (%)", "data", bold=True, w=2.0)
        for a, b in ((b0, b1), (b1, b2), (b2, b3), (b3, b4)):
            arrow(ax, a.r, b.l)
        ax.text(0.6, y - 0.75, note, fontsize=8, color=MUTED, style="italic", va="center")
    save(fig, out, "B2_panecho_baselines")


def fig_e2e_baselines(out):
    W, H = 36, 9.6
    fig, ax = canvas(W, H)
    header(ax, W, H, "Baselines B9-B11: end-to-end ResNet-18 + temporal model (first project track)",
           ("train", "op", "data"))
    y = 5.4
    b0 = box(ax, 2.6, y, "Echo clip\nT = 32, stride 2\n112 × 112, pad-crop aug.", "data", fs=8.5, minh=2.6)
    b1 = box(ax, 9.6, y, "ResNet-18 (ImageNet init)\ntrained end to end\n→ 512-d per frame", "train", fs=8.5, minh=2.6)
    b2 = box(ax, 17.6, y, "temporal model\nB9 BiGRU (3 layers)\nB10 Mamba (2 layers)\nB11 Transformer (UVT, 2 layers)",
             "train", fs=8.5, minh=2.6)
    b3 = box(ax, 25.2, y, "mean over real frames\n+ MLP head", "train", fs=8.5, minh=2.6)
    b4 = box(ax, 31.0, y, "EF (%)", "data", bold=True, minw=2.0)
    for a, b in ((b0, b1), (b1, b2), (b2, b3), (b3, b4)):
        arrow(ax, a.r, b.l)
    ax.text(0.6, 2.2, "Same temporal blocks as B1-B3 with 512-d input. AdamW lr 1e-4, cosine, batch 16, 25 epochs, "
            "seed 0, no EMA (this track predates it). Superseded by the frozen-PanEcho track (B1-B8), which is better "
            "for every model.", fontsize=8.5, color=MUTED)
    save(fig, out, "B3_end_to_end_baselines")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.DATA_OUT / "figures" / "architecture"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_pipeline, fig_sync_block, fig_clocks, fig_readouts, fig_finetune, fig_end_to_end,
              fig_baseline_blocks, fig_panecho_baselines, fig_e2e_baselines):
        f(out)


if __name__ == "__main__":
    main()
