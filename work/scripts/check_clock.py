r"""Validate the label-free cardiac clock (cardiac/sync.py) on VAL videos only.

Reference: the period of each WHOLE video (3-5 beats), from the pixel periodogram of all frames
(28x28 grey-level frames, no window, fundamental only). Checked against the traced ED/ES frames:
|ED-ES| / period should be the systolic fraction of the cycle (~0.3-0.45 at normal heart rates).
Clip-level estimates (the model's input: T frames at the training stride) from several sources and
estimator settings are compared with the reference: median relative error, share within 10%, and
octave errors (ratio ~2 or ~0.5).

Usage: py.cmd scripts\check_clock.py [--n-videos 300] [--T 32] [--run bigru_T32_s0]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cardiac  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cardiac import config  # noqa: E402
from cardiac.dataset import EchoClips, Normalize, load_frames, pixel_stats  # noqa: E402
from cardiac.models import EFModel, ResNet18Encoder  # noqa: E402
from cardiac.sync import cardiac_clock  # noqa: E402


def pixels(video, size=28):
    """uint8 (B, T, 3, H, W) -> (B, T, size*size) grey levels in [0, 1]."""
    B, T = video.shape[:2]
    g = video.float().mean(2, keepdim=True).flatten(0, 1) / 255.0
    return F.adaptive_avg_pool2d(g, size).view(B, T, -1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-videos", type=int, default=300)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--run", default="bigru_T32_s0", help="trained encoder to test as a source too")
    ap.add_argument("--out", default=str(config.RESULTS / "clock_check.json"))
    a = ap.parse_args()
    dev = torch.device("cuda")
    mean, std = pixel_stats()
    norm = Normalize(mean, std).to(dev)
    enc_in = ResNet18Encoder(pretrained=True).to(dev).eval()
    enc_tr = None
    if a.run:
        ck = torch.load(config.RUNS / a.run / "best.pt", map_location="cpu", weights_only=False)
        m = EFModel(ck["args"]["temporal"], mean, std, d_model=ck["args"]["d_model"],
                    n_layers=ck["args"]["n_layers"], pretrained=False,
                    temporal_kw=json.loads(ck["args"]["temporal_kw"]))
        m.load_state_dict(ck["model"])
        enc_tr = m.encoder.to(dev).eval()

    ds = EchoClips("VAL", T=a.T, stride=a.stride, pad="loop", mode="traced", require_tracings=True,
                   max_videos=a.n_videos)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    variants = {"hann_detrend": dict(), "hann": dict(detrend=False), "rect": dict(detrend=False, window="none"),
                "rect_detrend": dict(window="none"),
                "hann_detrend_oct.5": dict(octave_ratio=0.5), "hann_oct.5": dict(detrend=False, octave_ratio=0.5),
                "hann_oct.3": dict(detrend=False, octave_ratio=0.3), "rect_oct.5": dict(detrend=False, window="none", octave_ratio=0.5)}
    est = {}
    meta = {k: [] for k in ("vid", "fps", "ed", "es", "n_frames", "valid_frac")}
    for b in dl:
        v = b["video"].to(dev)
        B, T = v.shape[:2]
        feats = {"pixels": pixels(v)}
        x = norm(v).flatten(0, 1).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats["imagenet"] = enc_in(x).view(B, T, -1).float()
            if enc_tr is not None:
                feats["trained"] = enc_tr(x).view(B, T, -1).float()
        for src, f in feats.items():
            for name, kw in variants.items():
                om, pk = cardiac_clock(f, **kw)
                est.setdefault(f"{src}/{name}", []).append((2 * np.pi / om.cpu().numpy()) * a.stride)
        for k in ("vid", "fps", "ed", "es", "n_frames"):
            meta[k].append(b[k].numpy())
        meta["valid_frac"].append(b["valid"].float().mean(1).numpy())
    est = {k: np.concatenate(v) for k, v in est.items()}
    meta = {k: np.concatenate(v) for k, v in meta.items()}

    # whole-video reference periods (raw frames), pixel periodogram over all frames
    ref = []
    for vi in meta["vid"]:
        n = int(ds.n_frames[vi])
        idx = np.arange(min(n, 600))
        fr = torch.from_numpy(np.ascontiguousarray(load_frames(f"{ds.video_dir}/{ds.names[vi]}.avi", idx)))
        fr = fr.permute(0, 3, 1, 2)
        om, _ = cardiac_clock(pixels(fr[None].to(dev)), min_period=12.0, max_period=110.0, n_harm=1, n_grid=400)
        ref.append(float(2 * np.pi / om.item()))
    ref = np.array(ref)
    sysfrac = np.abs(meta["ed"] - meta["es"]) / ref
    hr = 60.0 * meta["fps"] / ref
    report = {"n_clips": int(len(ref)), "T": a.T, "stride": a.stride,
              "reference": {"hr_bpm_quantiles_5_25_50_75_95": np.percentile(hr, [5, 25, 50, 75, 95]).round(1).tolist(),
                            "systolic_fraction_quantiles_5_25_50_75_95":
                                np.percentile(sysfrac, [5, 25, 50, 75, 95]).round(3).tolist(),
                            "share_sysfrac_0.2_0.55": float(((sysfrac > 0.2) & (sysfrac < 0.55)).mean())},
              "clip_vs_reference": {}}
    full = meta["valid_frac"] > 0.999
    plaus = (sysfrac > 0.25) & (sysfrac < 0.5)      # references consistent with a normal systolic fraction
    report["n_plausible_reference"] = int(plaus.sum())
    for k, p in est.items():
        r = p / ref
        rp = r[plaus]
        report["clip_vs_reference"][k] = {
            "median_rel_err": float(np.median(np.abs(r - 1))),
            "within_10pct": float((np.abs(r - 1) < 0.10).mean()),
            "within_20pct": float((np.abs(r - 1) < 0.20).mean()),
            "octave_up_(ratio<0.6)": float((r < 0.6).mean()),
            "octave_down_(ratio>1.7)": float((r > 1.7).mean()),
            "within_10pct_no_padding": float((np.abs(r[full] - 1) < 0.10).mean()),
            "plausible_ref_within_10pct": float((np.abs(rp - 1) < 0.10).mean()),
            "plausible_ref_within_20pct": float((np.abs(rp - 1) < 0.20).mean()),
            "plausible_ref_octave_errors": float(((rp < 0.6) | (rp > 1.7)).mean()),
        }
    print(json.dumps(report["reference"]), "n plausible", report["n_plausible_reference"])
    for k, d in report["clip_vs_reference"].items():
        print(f"{k:28s} " + " ".join(f"{kk.split('_(')[0]}={v:.3f}" for kk, v in d.items()))
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
