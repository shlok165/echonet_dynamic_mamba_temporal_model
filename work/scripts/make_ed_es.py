r"""Derive ED/ES frame indices from VolumeTracings.csv and sanity-check the assignment.

Rule: in each traced frame, row 0 is taken as the long axis and rows 1.. as chords
perpendicular to it. The left-ventricular volume is estimated with the method of disks
(Simpson's rule, as in EchoNet): V = pi/4 * sum_i d_i^2 * (L / n_chords).
The traced frame with the larger volume is ED, the smaller is ES.

Checks printed:
  1. row 0 really is the longest segment in (almost) every traced frame,
  2. disk volumes correlate with the EDV/ESV columns of FileList.csv (and better than swapped),
  3. agreement of this rule with the simpler sum-of-squared-chords heuristic and with
     the order in which the two frames appear in the CSV.

Usage:  python scripts/make_ed_es.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardiac import config  # noqa: E402


def main():
    fl = pd.read_csv(config.FILELIST)
    vt = pd.read_csv(config.TRACINGS)
    vt["FileName"] = vt["FileName"].str.replace(".avi", "", regex=False)
    vt["seg2"] = (vt.X2 - vt.X1) ** 2 + (vt.Y2 - vt.Y1) ** 2
    vt["row"] = vt.groupby(["FileName", "Frame"], sort=False).cumcount()

    # order in which each video's traced frames first appear in the CSV
    first_pos = vt.reset_index().groupby(["FileName", "Frame"], sort=False)["index"].min()

    # Most traced frames have 21 rows (1 long axis + 20 chords). ~0.6% have 42/63/84/105 rows,
    # i.e. 2-5 tracings stacked; each 21-row block is one tracing with its own long axis first.
    rows = []
    n_blocks = n_axis_longest = n_dup_identical = n_multi = 0
    for (fn, fr), g in vt.groupby(["FileName", "Frame"], sort=False):
        seg2 = g.seg2.to_numpy()
        xy = g[["X1", "Y1", "X2", "Y2"]].to_numpy()
        # 3 traced frames have 16/18/41 rows: treat them as one tracing with n-1 chords.
        k = len(seg2) // 21 if len(seg2) % 21 == 0 else 1
        bl = len(seg2) // k
        vols, sumsqs = [], []
        for b in range(k):
            s = seg2[bl * b: bl * (b + 1)]
            n_blocks += 1
            n_axis_longest += int(np.argmax(s) == 0)
            vols.append(np.pi / 4 * s[1:].sum() * (np.sqrt(s[0]) / (len(s) - 1)))
            sumsqs.append(s[1:].sum())
        if k > 1:
            n_multi += 1
            n_dup_identical += int(all(np.allclose(xy[:bl], xy[bl * b: bl * (b + 1)]) for b in range(1, k)))
        rows.append(dict(FileName=fn, Frame=int(fr), vol=float(np.mean(vols)), sumsq=float(np.mean(sumsqs)),
                         n_rows=len(g), pos=int(first_pos[(fn, fr)])))
    per = pd.DataFrame(rows)

    print(f"traced frames: {len(per)}, tracing blocks: {n_blocks}; row 0 of a block is its longest "
          f"segment in {n_axis_longest} ({100 * n_axis_longest / n_blocks:.2f}%)")
    print("rows per traced frame:", per.n_rows.value_counts().to_dict())
    print(f"frames with >1 tracing: {n_multi}, of which all blocks identical: {n_dup_identical} "
          f"(volumes averaged over blocks)")

    out = []
    n_sumsq_agree = n_csvorder_small_first = 0
    for fn, g in per.groupby("FileName", sort=False):
        if len(g) != 2:
            print("skip (not exactly 2 traced frames):", fn, len(g))
            continue
        g = g.sort_values("pos")
        a, b = g.iloc[0], g.iloc[1]
        ed, es = (a, b) if a.vol > b.vol else (b, a)
        ed_ss = a if a.sumsq > b.sumsq else b
        n_sumsq_agree += int(ed_ss.Frame == ed.Frame)
        n_csvorder_small_first += int(es.Frame == a.Frame)
        out.append(dict(FileName=fn, ED=ed.Frame, ES=es.Frame, vol_ED=ed.vol, vol_ES=es.vol))
    out = pd.DataFrame(out)
    m = out.merge(fl[["FileName", "EDV", "ESV", "EF", "NumberOfFrames", "Split"]], on="FileName")
    n = len(m)
    print(f"videos with 2 traced frames: {n}")
    print(f"disk-volume rule agrees with sum-of-squared-chords rule: {n_sumsq_agree}/{n}")
    print(f"ES is the first traced frame listed in the CSV: {n_csvorder_small_first}/{n}")
    print(f"ED before ES in time: {(m.ED < m.ES).sum()}, ES before ED: {(m.ES < m.ED).sum()}")

    # Tracings are in pixels and the pixel->mm scale varies per video (imaging depth), so raw
    # volume-vs-EDV correlations are weak and NOT diagnostic. The EF ratio cancels the scale and is
    # the decisive check: a swapped assignment would give negative EF.
    print("(volume correlations below are scale-confounded; see the EF check)")
    r_ed = stats.pearsonr(m.vol_ED, m.EDV)[0]
    r_es = stats.pearsonr(m.vol_ES, m.ESV)[0]
    r_swap_ed = stats.pearsonr(m.vol_ES, m.EDV)[0]
    r_swap_es = stats.pearsonr(m.vol_ED, m.ESV)[0]
    print(f"Pearson(vol_ED, EDV)={r_ed:.4f}  Pearson(vol_ES, ESV)={r_es:.4f}")
    print(f"swapped:  Pearson(vol_ES, EDV)={r_swap_ed:.4f}  Pearson(vol_ED, ESV)={r_swap_es:.4f}")
    ef_trace = 100 * (m.vol_ED - m.vol_ES) / m.vol_ED
    print(f"EF from tracings vs FileList EF: Pearson={stats.pearsonr(ef_trace, m.EF)[0]:.4f}, "
          f"MAE={np.abs(ef_trace - m.EF).mean():.2f}")
    # In which videos is the time order unusual? Check they are not label errors.
    odd = m[m.ES < m.ED]
    print(f"ES-before-ED videos: mean EF {odd.EF.mean():.1f}, "
          f"Pearson(vol_ED,EDV) within them={stats.pearsonr(odd.vol_ED, odd.EDV)[0]:.4f}")

    missing = sorted(set(fl.FileName) - set(m.FileName))
    print("FileList videos without tracings:", missing)

    config.CACHE.mkdir(parents=True, exist_ok=True)
    m[["FileName", "ES", "ED", "NumberOfFrames", "Split", "vol_ED", "vol_ES"]].to_csv(
        config.ED_ES_CSV, index=False)
    print("wrote", config.ED_ES_CSV)


if __name__ == "__main__":
    main()
