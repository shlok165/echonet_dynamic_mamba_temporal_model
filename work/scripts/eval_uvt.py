r"""Evaluate a UVT model on full TEST videos, as Network/test.py does with use_full_videos=True.

Each video is passed whole (batch 1, every frame, 112 -> 128 zero padding, /255). EF = model EF output
x 100. ED/ES frames are found from the SD signal with the original peak decoding
(cardiac.uvt.decode_sd_peaks), and the error is the nearest same-class peak to the labelled frame.
EF metrics come from cardiac.metrics so they are directly comparable with our other models.

Weights:
  --run <name>               our checkpoint runs/<name>/best.pt (config read from it)
  --released <best.pt>       a checkpoint from the UVT repo (paper config, see --sdmode / download script)

Notes
  * Videos are filtered as in the original test (|ED-ES| > 10 frames); report the count.
  * The paper preset needs ~0.7 GB of weights in bf16; inference fits on the 6 GB GPU (frames are encoded
    in chunks). Training it does not (see train_uvt.py).
Example: py.cmd scripts\eval_uvt.py --released cache\uvt_weights\UVT_M_REG\best.pt --sdmode reg --tag M_REG
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
from cardiac.uvt import (PAPER_CONFIG, count_params, decode_sd_peaks, get_uvt,  # noqa: E402
                         load_released_checkpoint, peak_errors)
from cardiac.uvt_data import UVTEchoSet  # noqa: E402

GiB = 1024 ** 3


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run")
    g.add_argument("--released")
    ap.add_argument("--sdmode", choices=["reg", "cla"], default="reg", help="for --released")
    ap.add_argument("--tag", default=None, help="output name (default: run name)")
    ap.add_argument("--label-order", choices=["verified", "csv"], default="verified")
    ap.add_argument("--split", default="TEST")
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    a = ap.parse_args()
    dev = torch.device("cuda")

    if a.run:
        ck = torch.load(config.RUNS / a.run / "best.pt", map_location="cpu", weights_only=False)
        ta = ck["args"]
        model = get_uvt(SDmode=ta["sdmode"], rm_branch=ta["rm_branch"], pos_emb=ta["pos_emb"], **ck["model_cfg"])
        model.load_state_dict(ck["model"])
        sdmode, tag = ta["sdmode"], a.tag or a.run
    else:
        model = get_uvt(SDmode=a.sdmode, **PAPER_CONFIG)
        res, meta = load_released_checkpoint(model, a.released)
        print("loaded released weights:", res, meta)
        sdmode, tag = a.sdmode, a.tag or Path(a.released).parent.name
    model.to(dev).eval()

    ds = UVTEchoSet(a.split, mode="full", SDmode=sdmode, label_order=a.label_order, max_videos=a.max_videos)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=a.workers)
    rows, t_model = [], 0.0
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for b in dl:
            v = b["video"].to(dev)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp == "bf16"):
                out = model(v)
            torch.cuda.synchronize()
            t_model += time.perf_counter() - t0
            es, ed = int(b["es"]), int(b["ed"])
            r = {"name": ds.names[int(b["vid"])], "ef": float(b["ef"]), "n_frames": v.shape[1], "es": es, "ed": ed}
            if out["ef"] is not None:
                r["ef_pred"] = float(out["ef"].float()) * 100
            if out["sd"] is not None:
                if sdmode == "reg":
                    sig = out["sd"].float()[0].cpu()
                    peaks = decode_sd_peaks(sig)
                else:  # cla: ED-vs-ES score, decoded the same way (an adaptation; the original cla path differs)
                    pr = out["sd"].float()[0].softmax(-1).cpu()
                    sig = pr[:, 2] - pr[:, 1]
                    peaks = decode_sd_peaks(sig)
                r["decode_failed"] = peaks is None
                r["es_err"], r["ed_err"] = peak_errors(peaks, es, ed)
                r["argmin_es_err"] = int(sig.argmin()) - es  # simple single-extremum reference
                r["argmax_ed_err"] = int(sig.argmax()) - ed
            rows.append(r)

    out_dir = config.RESULTS / f"uvt_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"tag": tag, "n_videos": len(rows), "label_order": a.label_order, "params": count_params(model),
           "infer_model_ms_per_video": 1000 * t_model / max(len(rows), 1),
           "infer_peak_gib": torch.cuda.max_memory_allocated() / GiB}
    if rows and "ef_pred" in rows[0]:
        y = np.array([r["ef"] for r in rows])
        p = np.array([r["ef_pred"] for r in rows])
        ef = M.regression_metrics(y, p)
        ef.update(M.threshold_metrics(y, p, 40))
        ef.update(M.threshold_metrics(y, p, 50))
        ef["bland_altman"] = M.bland_altman(y, p)
        res["ef"] = ef
        M.bland_altman_plot(y, p, out_dir / "bland_altman.png", title=f"UVT {tag}")
    if rows and "es_err" in rows[0]:
        for k in ("es_err", "ed_err", "argmin_es_err", "argmax_ed_err"):
            e = np.array([r[k] for r in rows], dtype=float)
            ok = np.isfinite(e)
            res[k] = {"aFD": float(np.abs(e[ok]).mean()) if ok.any() else None,
                      "mean_signed": float(e[ok].mean()) if ok.any() else None,
                      "sd": float(e[ok].std()) if ok.any() else None, "n": int(ok.sum())}
        res["decode_failed"] = int(sum(r["decode_failed"] for r in rows))
    (out_dir / "eval.json").write_text(json.dumps(res, indent=1, default=float))
    with open(out_dir / "per_video.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "ef"}, indent=1, default=float))
    if "ef" in res:
        print({k: round(res["ef"][k], 4) for k in ("MAE", "RMSE", "R2", "Pearson")})


if __name__ == "__main__":
    main()
