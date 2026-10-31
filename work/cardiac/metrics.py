"""Evaluation metrics. Pure numpy/scipy (no sklearn in this env); every function takes array-likes.

EF regression ........ regression_metrics, bland_altman, bland_altman_plot, auroc,
                       threshold_metrics, binned_mae
Temporal quality ..... cosine_smoothness, ef_std_across_clips, video_mean, pca_2d,
                       autocorr_periodicity
ED/ES probe .......... heatmap_targets, ridge_fit, ridge_predict, localize, frame_error
Frame-order test ..... relative_change
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _f64(x):
    return np.asarray(x, dtype=np.float64).ravel()


# ----------------------------------------------------------------------------- EF regression

def regression_metrics(y_true, y_pred) -> dict:
    """MAE, MSE, RMSE, R2, MAPE(%), Pearson, Spearman + extras:
    CCC     Lin's concordance correlation: penalises bias and scale error, which Pearson ignores.
    bias    mean(pred - true), the Bland-Altman bias.
    within5 / within10   fraction of videos within +-5 / +-10 EF points (clinically readable).
    """
    y, p = _f64(y_true), _f64(y_pred)
    assert y.shape == p.shape and y.size >= 2
    e = p - y
    mse = float(np.mean(e ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    vy, vp = y.var(), p.var()
    cov = np.mean((y - y.mean()) * (p - p.mean()))
    return {
        "MAE": float(np.mean(np.abs(e))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(1 - np.sum(e ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "MAPE": float(100 * np.mean(np.abs(e) / np.abs(y))),
        "Pearson": float(stats.pearsonr(y, p)[0]),
        "Spearman": float(stats.spearmanr(y, p)[0]),
        "CCC": float(2 * cov / (vy + vp + (y.mean() - p.mean()) ** 2)),
        "bias": float(e.mean()),
        "within5": float(np.mean(np.abs(e) <= 5)),
        "within10": float(np.mean(np.abs(e) <= 10)),
        "n": int(y.size),
    }


def bland_altman(y_true, y_pred) -> dict:
    """Bias, SD of differences, 95% limits of agreement, proportional bias (slope of diff on mean)."""
    y, p = _f64(y_true), _f64(y_pred)
    d = p - y
    m = (p + y) / 2
    bias = float(d.mean())
    sd = float(d.std(ddof=1))
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    lr = stats.linregress(m, d)
    return {"bias": bias, "sd": sd, "loa_low": lo, "loa_high": hi,
            "frac_outside_loa": float(np.mean((d < lo) | (d > hi))),
            "prop_bias_slope": float(lr.slope), "prop_bias_p": float(lr.pvalue)}


def bland_altman_plot(y_true, y_pred, path=None, title="", ax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y, p = _f64(y_true), _f64(y_pred)
    ba = bland_altman(y, p)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    ax.scatter((y + p) / 2, p - y, s=4, alpha=0.4, lw=0)
    for v, ls in ((ba["bias"], "-"), (ba["loa_low"], "--"), (ba["loa_high"], "--")):
        ax.axhline(v, color="k", ls=ls, lw=1)
    ax.set_xlabel("mean of predicted and true EF (%)")
    ax.set_ylabel("predicted - true EF (%)")
    ax.set_title(f"{title}\nbias {ba['bias']:.2f}, LoA [{ba['loa_low']:.2f}, {ba['loa_high']:.2f}]",
                 fontsize=9)
    if own:
        fig.tight_layout()
        if path is not None:
            fig.savefig(path)
        plt.close(fig)
    return ba


def auroc(labels, scores) -> float:
    """Area under ROC via the Mann-Whitney U statistic (ties get half credit)."""
    lab = np.asarray(labels).astype(bool).ravel()
    s = _f64(scores)
    n1, n0 = lab.sum(), (~lab).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(s)
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def threshold_metrics(y_true, y_pred, thr: float = 50.0) -> dict:
    """Reduced-EF detection (true EF < thr): AUROC of -pred, and sensitivity/specificity of pred < thr.
    Justification: EF is used clinically through cut-offs (HFrEF < 40, abnormal < 50)."""
    y, p = _f64(y_true), _f64(y_pred)
    pos = y < thr
    pp = p < thr
    return {f"AUROC_lt{thr:g}": auroc(pos, -p),
            f"sens_lt{thr:g}": float(pp[pos].mean()) if pos.any() else float("nan"),
            f"spec_lt{thr:g}": float((~pp[~pos]).mean()) if (~pos).any() else float("nan"),
            f"n_pos_lt{thr:g}": int(pos.sum())}


def binned_mae(y_true, y_pred, edges=(0, 30, 40, 50, 60, 70, 101)) -> dict:
    """MAE per true-EF bin. Low-EF videos are rare, so overall MAE can hide poor accuracy there."""
    y, p = _f64(y_true), _f64(y_pred)
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y >= lo) & (y < hi)
        out[f"[{lo},{hi})"] = (float(np.abs(p[m] - y[m]).mean()) if m.any() else float("nan"), int(m.sum()))
    return out


# ----------------------------------------------------------------------------- temporal quality

def cosine_smoothness(Z, valid=None) -> float:
    """Mean cosine similarity between Z_t and Z_{t+1}.

    Z: (N, T, D) or (T, D). valid: optional (N, T) bool; a pair counts only if both frames are valid
    (excludes loop/repeat padding, whose seam would otherwise distort the score).
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 2:
        Z = Z[None]
        valid = None if valid is None else np.asarray(valid)[None]
    a, b = Z[:, :-1], Z[:, 1:]
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        m = v[:, :-1] & v[:, 1:]
        return float(cos[m].mean())
    return float(cos.mean())


def video_mean(video_ids, preds):
    """Average clip predictions per video. Returns (unique_ids, mean_pred)."""
    ids = np.asarray(video_ids).ravel()
    p = _f64(preds)
    u, inv = np.unique(ids, return_inverse=True)
    return u, np.bincount(inv, weights=p) / np.bincount(inv)


def ef_std_across_clips(video_ids, preds, ddof: int = 1) -> dict:
    """Std of predicted EF across clips of the same video, summarised over videos.
    Videos with a single clip are skipped. Lower = more consistent."""
    ids = np.asarray(video_ids).ravel()
    p = _f64(preds)
    u, inv = np.unique(ids, return_inverse=True)
    sds = []
    for k in range(len(u)):
        x = p[inv == k]
        if x.size > ddof:
            sds.append(x.std(ddof=ddof))
    sds = np.asarray(sds)
    if sds.size == 0:
        return {"mean_std": float("nan"), "median_std": float("nan"), "n_videos": 0}
    return {"mean_std": float(sds.mean()), "median_std": float(np.median(sds)), "n_videos": int(sds.size)}


def pca_2d(Z):
    """Project (T, D) or (N*T, D) features to 2D. Returns (coords (.,2), explained_variance_ratio (2,))."""
    X = np.asarray(Z, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    ev = S ** 2 / max((S ** 2).sum(), 1e-12)
    return X @ Vt[:2].T, ev[:2]


def autocorr_periodicity(Z, min_lag: int = 2):
    """Numeric periodicity of one clip's feature trajectory Z (T, D).

    r(l) = sum_t <z_t, z_{t+l}> / sum_t ||z_t||^2 after removing the temporal mean. Returns
    (peak_value, peak_lag): the highest autocorrelation at lags past the first non-positive value
    (or past min_lag). Periodic motion gives a clear peak at one cycle length.
    Justification: makes the qualitative PCA-loop plot comparable across models.
    """
    X = np.asarray(Z, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    T = X.shape[0]
    denom = (X ** 2).sum() + 1e-12
    r = np.array([(X[:T - l] * X[l:]).sum() / denom for l in range(T)])
    start = min_lag
    nonpos = np.where(r[1:] <= 0)[0]
    if nonpos.size:
        start = max(start, int(nonpos[0]) + 1)
    if start >= T - 1:
        return float("nan"), -1
    lag = start + int(np.argmax(r[start:T - 1]))
    return float(r[lag]), int(lag)


# ----------------------------------------------------------------------------- ED/ES linear probe

def heatmap_targets(frame_idx, target_frame, sigma: float):
    """Gaussian target over clip frames centred on an original frame index.
    frame_idx: (N, T) original indices of clip frames; target_frame: (N,). sigma in original frames."""
    fi = np.asarray(frame_idx, dtype=np.float64)
    tf = np.asarray(target_frame, dtype=np.float64)[:, None]
    return np.exp(-0.5 * ((fi - tf) / sigma) ** 2)


def ridge_fit(X, Y, lam: float = 1.0):
    """Closed-form ridge with intercept. X (n, D), Y (n, K). Features standardised internally."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    ym = Y.mean(0)
    A = Xs.T @ Xs + lam * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xs.T @ (Y - ym))
    return {"W": W, "mu": mu, "sd": sd, "b": ym}


def ridge_predict(model, X):
    X = np.asarray(X, dtype=np.float64)
    return ((X - model["mu"]) / model["sd"]) @ model["W"] + model["b"]


def localize(scores, frame_idx, valid=None):
    """Per clip: original frame index at the argmax of a per-frame score. scores/frame_idx (N, T)."""
    s = np.array(scores, dtype=np.float64)
    if valid is not None:
        s[~np.asarray(valid, dtype=bool)] = -np.inf
    return np.take_along_axis(np.asarray(frame_idx), s.argmax(1)[:, None], 1)[:, 0]


def probe_window(frame_idx, ed, es, margin_frac: float = 0.5, valid=None):
    """Search mask (N, T) for ED/ES localisation.

    EchoNet traces ONE ED and ONE ES per video, but a clip can contain several heartbeats, so a
    global argmax could hit an unlabelled (but correct) ED of another beat. The search is therefore
    restricted to [min(ED,ES) - m, max(ED,ES) + m] with m = margin_frac*|ED-ES|. With
    margin_frac=0.5 the window is ~2|ED-ES| (systole ~0.3-0.4 cycle), i.e. under one cycle, and it is
    symmetric so it does not reveal which end is ED.
    """
    fi = np.asarray(frame_idx)
    ed = np.asarray(ed)[:, None]
    es = np.asarray(es)[:, None]
    m = np.ceil(margin_frac * np.abs(ed - es))
    w = (fi >= np.minimum(ed, es) - m) & (fi <= np.maximum(ed, es) + m)
    if valid is not None:
        w &= np.asarray(valid, dtype=bool)
    return w


def frame_error(pred_frame, true_frame) -> dict:
    """Absolute error in ORIGINAL frames (not sampled frames, so different strides are comparable)."""
    e = np.abs(_f64(pred_frame) - _f64(true_frame))
    return {"mean": float(e.mean()), "median": float(np.median(e)), "n": int(e.size)}


def in_clip(frame_idx, valid, target_frame, stride):
    """Mask (N,) of clips whose real (valid) frames bracket the target frame within one stride."""
    fi = np.asarray(frame_idx)
    v = np.asarray(valid, dtype=bool)
    tf = np.asarray(target_frame)[:, None]
    d = np.where(v, np.abs(fi - tf), np.iinfo(np.int64).max)
    return d.min(1) <= np.asarray(stride) // 2


# ----------------------------------------------------------------------------- frame-order test

def relative_change(normal: dict, perturbed: dict, keys=("MAE", "RMSE", "R2", "Pearson")) -> dict:
    """perturbed - normal for each metric (positive MAE change = worse after shuffling)."""
    return {k: perturbed[k] - normal[k] for k in keys}
