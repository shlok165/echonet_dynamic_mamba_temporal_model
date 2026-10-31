r"""One PDF per non-baseline model: page 1 its architecture (scripts/draw_architectures.py), then its results
against the BiGRU baseline under the same protocol (scripts/compare_vs_bigru.py): validation, and test when
the test-set predictions exist. Plus all_architectures_and_comparison.pdf: a cover, all six architectures and
side-by-side tables of every model including the baselines (scripts/comparison_tables.py).
All pages vector. Default output: <project>\architecture\.

Usage: py.cmd scripts\build_arch_pdfs.py [--out C:\...\project\architecture]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

import compare_vs_bigru as C  # noqa: E402
import comparison_tables as T  # noqa: E402
import draw_architectures as D  # noqa: E402

figs = []


def grab(fig, *_):
    figs.append(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "architecture"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D.save = C.save = grab                    # collect the figures instead of writing image files
    splits = [s for s in ("val", "test") if C.have(s, [C.REF[1]] + [k for _, k in C.MAIN])]
    rows = []
    docs = [
        ("1_final_model", D.fig_pipeline, [lambda s: C.fig_headline(s, rows)], True),
        ("2_sync_block", D.fig_sync_block, [C.fig_all_metrics], True),
        ("3_clock_variants", D.fig_clocks, [lambda s: C.fig_clocks(s, rows)], True),
        ("4_readouts", D.fig_readouts, [lambda s: C.fig_readouts(rows)], False),
        ("5_encoder_finetuning", D.fig_finetune, [lambda s: C.fig_finetune(rows)], False),
        ("6_end_to_end", D.fig_end_to_end, [lambda s: C.fig_end_to_end(rows)], False),
    ]
    for name, arch, comps, per_split in docs:
        figs.clear()
        arch(None)
        for comp in comps:
            for s in (splits if per_split else ["val"]):
                comp(s)
        path = out / f"{name}_vs_bigru.pdf"
        with PdfPages(path) as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches="tight", pad_inches=0.15, facecolor="white")
                plt.close(fig)
        print(f"wrote {path} ({len(figs)} pages: architecture + {', '.join(splits if per_split else ['val'])})")
    build_all(out, splits)
    build_baselines(out.parent / "baseline", splits)


def build_baselines(out, splits):
    """baseline/baselines.pdf: cover, baseline architectures, then their results in tables (with the final
    proposed model as the last column for reference)."""
    out.mkdir(parents=True, exist_ok=True)
    by = {m[0]: m for m in T.MODELS}
    base = [m for m in T.MODELS if m[1] != "proposed"]
    final = by["sync fixed\n(final)"]
    acc_models = base + [(final[0] + "\n[reference]", *final[1:])]
    beh_models = [m for m in base if m[4] == "ours"] + [by["PanEcho TF\nour clips"]] + [acc_models[-1]]
    fig = plt.figure(figsize=(17.5, 6.4))
    fig.text(0.05, 0.88, "Baselines: architectures and results", fontsize=20, fontweight="bold", color=T.INK)
    fig.text(0.05, 0.8, "EF estimation on EchoNet-Dynamic; every baseline evaluated with the same protocol as the "
             "proposed sync model", fontsize=13, color=T.INK2)
    contents = ["Contents",
                "  B1-B4   Temporal baselines on the frozen PanEcho features: BiGRU, Mamba, Transformer (UVT), "
                "no temporal model",
                "  B5-B8   PanEcho's own video model: zero-shot, ridge probe, fine-tuned (native clips), "
                "fine-tuned (our clips)",
                "  B9-B11  End-to-end ResNet-18 + BiGRU / Mamba / Transformer (first project track)",
                "  Table B1   EF accuracy of B1-B8 (" + " and ".join(T.SPLIT_NAME[s] for s in splits) + ")",
                "  Table B2   Temporal behaviour and cost of B1-B4 (" + " and ".join(T.SPLIT_NAME[s] for s in splits)
                + ")",
                "  Table B3   End-to-end baselines B9-B11 (validation and test)",
                "",
                "The proposed model (sync, fixed frequency) appears as the last column of Tables B1-B2 for reference "
                "only; its architecture is in ../architecture."]
    fig.text(0.05, 0.7, "\n".join(contents), fontsize=11.5, color=T.INK, va="top", linespacing=1.75)
    fig.text(0.05, 0.04, "Data: EchoNet-Dynamic, Stanford AIMI (doi:10.71718/yqp5-y078), research use only.  "
             "PanEcho (Holste et al., JAMA 2025), weights CC BY-NC-SA 4.0.", fontsize=8.5, color=T.MUTED)
    pages = [fig]
    figs.clear()
    for arch in (D.fig_baseline_blocks, D.fig_panecho_baselines, D.fig_e2e_baselines):
        arch(None)
    pages += list(figs)
    for s in splits:
        pages.append(T.table_accuracy(s, acc_models, "Table B1. EF accuracy of the baselines"))
    for s in splits:
        pages.append(T.table_behaviour(s, beh_models, "Table B2. Temporal behaviour and cost of the baselines",
                                       "Baselines on the same frozen PanEcho 3-frame-stack features (B1-B4, B8)."))
    pages.append(T.table_e2e())
    path = out / "baselines.pdf"
    with PdfPages(path) as pdf:
        for f in pages:
            pdf.savefig(f, bbox_inches="tight", pad_inches=0.2, facecolor="white")
            plt.close(f)
    print(f"wrote {path} ({len(pages)} pages)")


def cover(splits):
    fig = plt.figure(figsize=(17.5, 6.4))
    t = fig.text
    t(0.05, 0.9, "Sync: a rotating-state SSM temporal block for EF estimation from echocardiography",
      fontsize=20, fontweight="bold", color=T.INK)
    t(0.05, 0.83, "Architectures of every model in the project, then side-by-side comparison with the baselines",
      fontsize=13, color=T.INK2)
    contents = ["Contents",
                "  1   Proposed model: frozen PanEcho frame encoder + sync temporal block",
                "  2   Sync temporal block (2 layers) and the SyncMixer",
                "  3   Clock variants: where the rotation frequency comes from",
                "  4   Explored readout heads (volumetric, range, harmonic)",
                "  5   Encoder fine-tuning variant",
                "  6   End-to-end ResNet-18 variants",
                "  Table 1   EF accuracy of all models (" + " and ".join(T.SPLIT_NAME[s] for s in splits) + ")",
                "  Table 2   Temporal behaviour and cost (" + " and ".join(T.SPLIT_NAME[s] for s in splits) + ")",
                "  Table 3   Explored variants vs BiGRU and Mamba under matched protocols (validation)"]
    t(0.05, 0.71, "\n".join(contents), fontsize=11.5, color=T.INK, va="top", linespacing=1.75)
    setup = ["Setup",
             "• EchoNet-Dynamic, official split: 7,465 train / 1,288 validation / 1,277 test videos; EF label per video.",
             "• Per-frame features from the frozen PanEcho ConvNeXt-T image encoder, each frame packed with its "
             "neighbours (t−1, t, t+1).",
             "• Every temporal model: clips of 32 frames at stride 2, mean pool over real frames, the same MLP head, "
             "AdamW, 40 epochs, weight EMA.",
             "• Baselines: BiGRU, Mamba, Transformer (UVT), no temporal model (same features and recipe); PanEcho's own "
             "temporal model (fine-tuned), ridge probe and zero-shot.",
             "• All design decisions on the validation set; the test set evaluated once on the locked lineup.",
             "  Test-set tables are included when the test evaluation has finished (" +
             ("included" if "test" in splits else "pending at the time of this build") + ")."]
    t(0.5, 0.71, "\n".join(setup), fontsize=10, color=T.INK, va="top", linespacing=1.8, wrap=True)
    t(0.05, 0.04, "Data: EchoNet-Dynamic, Stanford AIMI (doi:10.71718/yqp5-y078), research use only.  "
      "Encoder: PanEcho (Holste et al., JAMA 2025), weights CC BY-NC-SA 4.0.", fontsize=8.5, color=T.MUTED)
    return fig


def build_all(out, splits):
    """Single PDF: cover, the six architectures, then Tables 1-3."""
    pages = [cover(splits)]
    figs.clear()
    for arch in (D.fig_pipeline, D.fig_sync_block, D.fig_clocks, D.fig_readouts, D.fig_finetune, D.fig_end_to_end):
        arch(None)
    pages += list(figs)
    for s in splits:
        pages.append(T.table_accuracy(s))
    for s in splits:
        pages.append(T.table_behaviour(s))
    pages.append(T.table_explored())
    path = out / "all_architectures_and_comparison.pdf"
    with PdfPages(path) as pdf:
        for fig in pages:
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.2, facecolor="white")
            plt.close(fig)
    print(f"wrote {path} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
