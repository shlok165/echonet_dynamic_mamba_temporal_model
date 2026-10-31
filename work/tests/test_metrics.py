"""Unit tests for cardiac.metrics on synthetic data with known answers.
Run:  py.cmd -m unittest discover -s tests -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cardiac  # noqa: E402,F401  (sets MKL_THREADING_LAYER before numpy BLAS is used)
from cardiac import metrics as M  # noqa: E402


class TestRegression(unittest.TestCase):
    def test_perfect(self):
        y = np.linspace(10, 90, 50)
        r = M.regression_metrics(y, y)
        for k in ("MAE", "MSE", "RMSE", "MAPE", "bias"):
            self.assertAlmostEqual(r[k], 0.0)
        for k in ("R2", "Pearson", "Spearman", "CCC", "within5"):
            self.assertAlmostEqual(r[k], 1.0)

    def test_hand_computed(self):
        y = np.array([50.0, 60.0, 40.0, 70.0])
        p = np.array([52.0, 57.0, 40.0, 74.0])  # errors +2, -3, 0, +4
        r = M.regression_metrics(y, p)
        self.assertAlmostEqual(r["MAE"], 9 / 4)
        self.assertAlmostEqual(r["MSE"], 29 / 4)
        self.assertAlmostEqual(r["RMSE"], np.sqrt(29 / 4))
        self.assertAlmostEqual(r["R2"], 1 - 29 / 500)  # SST = 0+100+100+400
        self.assertAlmostEqual(r["MAPE"], 100 / 4 * (2 / 50 + 3 / 60 + 0 + 4 / 70))
        self.assertAlmostEqual(r["bias"], 0.75)
        self.assertAlmostEqual(r["within5"], 1.0)

    def test_against_numpy_large(self):
        rng = np.random.default_rng(0)
        y = rng.normal(55, 12, 5000)  # n >= 5000 also exercises the MKL/BLAS path that crashed
        p = y + rng.normal(0, 5, y.size)
        r = M.regression_metrics(y, p)
        self.assertAlmostEqual(r["Pearson"], np.corrcoef(y, p)[0, 1], places=10)
        ry, rp = np.argsort(np.argsort(y)), np.argsort(np.argsort(p))
        self.assertAlmostEqual(r["Spearman"], np.corrcoef(ry, rp)[0, 1], places=10)
        self.assertAlmostEqual(r["R2"], 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2), places=10)

    def test_ccc_penalises_offset(self):
        y = np.linspace(20, 80, 100)
        r = M.regression_metrics(y, y + 10)
        self.assertAlmostEqual(r["Pearson"], 1.0)
        self.assertLess(r["CCC"], 0.9)
        self.assertAlmostEqual(r["bias"], 10.0)
        # closed form: 2 var / (2 var + d^2) for a pure shift d
        v = y.var()
        self.assertAlmostEqual(r["CCC"], 2 * v / (2 * v + 100))


class TestBlandAltman(unittest.TestCase):
    def test_stats(self):
        rng = np.random.default_rng(1)
        y = rng.uniform(20, 80, 20000)
        p = y + rng.normal(2.0, 3.0, y.size)
        ba = M.bland_altman(y, p)
        self.assertAlmostEqual(ba["bias"], 2.0, delta=0.1)
        self.assertAlmostEqual(ba["sd"], 3.0, delta=0.1)
        self.assertAlmostEqual(ba["loa_high"] - ba["loa_low"], 2 * 1.96 * ba["sd"])
        self.assertAlmostEqual(ba["frac_outside_loa"], 0.05, delta=0.01)
        # regressing d on the mean has a known positive artefact: cov(d,m)/var(m) = (s^2/2)/(var y + s^2/4)
        self.assertAlmostEqual(ba["prop_bias_slope"], 4.5 / (60 ** 2 / 12 + 9 / 4), delta=0.005)

    def test_proportional_bias(self):
        y = np.linspace(20, 80, 200)
        p = 1.2 * y - 11  # diff = 0.2 y - 11, mean = 1.1 y - 5.5 -> slope 0.2/1.1
        ba = M.bland_altman(y, p)
        self.assertAlmostEqual(ba["prop_bias_slope"], 0.2 / 1.1, places=6)

    def test_plot_writes_file(self):
        rng = np.random.default_rng(2)
        y = rng.uniform(20, 80, 300)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "ba.png"
            M.bland_altman_plot(y, y + rng.normal(0, 4, y.size), f, title="test")
            self.assertGreater(f.stat().st_size, 1000)


class TestClassification(unittest.TestCase):
    def test_auroc_extremes_and_bruteforce(self):
        self.assertEqual(M.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(M.auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)
        self.assertEqual(M.auroc([0, 1], [0.5, 0.5]), 0.5)
        rng = np.random.default_rng(3)
        lab = rng.integers(0, 2, 300).astype(bool)
        s = rng.normal(lab * 0.7, 1.0).round(1)  # rounding creates ties
        pos, neg = s[lab], s[~lab]
        brute = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (pos.size * neg.size)
        self.assertAlmostEqual(M.auroc(lab, s), brute, places=12)

    def test_threshold(self):
        y = np.array([30, 45, 55, 65.0])
        p = np.array([35, 52, 54, 60.0])  # second positive missed
        t = M.threshold_metrics(y, p, 50)
        self.assertAlmostEqual(t["sens_lt50"], 0.5)
        self.assertAlmostEqual(t["spec_lt50"], 1.0)
        self.assertEqual(t["n_pos_lt50"], 2)

    def test_binned(self):
        b = M.binned_mae([25, 35, 65], [27, 30, 65], edges=(0, 30, 40, 101))
        self.assertEqual(b["[0,30)"], (2.0, 1))
        self.assertEqual(b["[30,40)"], (5.0, 1))
        self.assertEqual(b["[40,101)"], (0.0, 1))


class TestTemporal(unittest.TestCase):
    def test_cosine(self):
        z = np.ones((5, 8))
        self.assertAlmostEqual(M.cosine_smoothness(z), 1.0)
        alt = np.stack([np.ones(8) * (-1) ** t for t in range(6)])
        self.assertAlmostEqual(M.cosine_smoothness(alt), -1.0)
        # mask: frames 0-2 constant (cos 1), frames 3-5 alternate; masking 3..5 must give 1
        z = np.concatenate([np.ones((3, 4)), alt[:3, :4] * -1, ], 0)[None]
        valid = np.array([[1, 1, 1, 0, 0, 0]], bool)
        self.assertAlmostEqual(M.cosine_smoothness(z, valid), 1.0)
        self.assertLess(M.cosine_smoothness(z), 1.0)

    def test_cosine_rotation(self):
        th = np.linspace(0, 2 * np.pi, 33)[:-1]
        z = np.stack([np.cos(th), np.sin(th)], 1)
        self.assertAlmostEqual(M.cosine_smoothness(z), np.cos(th[1] - th[0]))

    def test_ef_std(self):
        ids = [0, 0, 0, 1, 1, 2]
        p = [50, 52, 54, 60, 60, 70]
        r = M.ef_std_across_clips(ids, p)
        self.assertEqual(r["n_videos"], 2)  # video 2 has one clip
        self.assertAlmostEqual(r["mean_std"], (2.0 + 0.0) / 2)
        u, m = M.video_mean(ids, p)
        np.testing.assert_allclose(m, [52, 60, 70])

    def test_pca_circle(self):
        rng = np.random.default_rng(4)
        th = np.linspace(0, 4 * np.pi, 200)  # two "heartbeats"
        X2 = np.stack([3 * np.cos(th), 3 * np.sin(th)], 1)
        Q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
        X = X2 @ Q[:2] + 0.01 * rng.normal(size=(200, 16)) + 5.0
        c, ev = M.pca_2d(X)
        self.assertGreater(ev.sum(), 0.99)
        np.testing.assert_allclose(np.linalg.norm(c, axis=1), 3.0, atol=0.15)

    def test_periodicity(self):
        rng = np.random.default_rng(5)
        t = np.arange(64)
        Z = np.stack([np.sin(2 * np.pi * t / 16), np.cos(2 * np.pi * t / 16)], 1) @ rng.normal(size=(2, 32))
        peak, lag = M.autocorr_periodicity(Z)
        self.assertEqual(lag, 16)
        self.assertGreater(peak, 0.6)
        noise_peak, _ = M.autocorr_periodicity(rng.normal(size=(64, 32)))
        self.assertLess(noise_peak, 0.3)


class TestProbe(unittest.TestCase):
    def test_ridge_probe_localizes(self):
        rng = np.random.default_rng(6)
        N, T, D, stride = 400, 32, 24, 2
        starts = rng.integers(0, 40, N)
        fidx = starts[:, None] + stride * np.arange(T)[None]
        period = 40.0  # clip span 63 > period: the clip holds a second, unlabelled ED
        ed = starts + rng.integers(10, 30, N)
        es = ed + 15  # ES ~0.375 cycle after ED
        phase = 2 * np.pi * (fidx - ed[:, None]) / period  # ED at phase 0
        basis = np.stack([np.cos(phase), np.sin(phase), np.cos(2 * phase), np.sin(2 * phase)], -1)
        Z = basis @ rng.normal(size=(4, D)) + 0.1 * rng.normal(size=(N, T, D))
        Y = np.stack([M.heatmap_targets(fidx, ed, 3.0), M.heatmap_targets(fidx, es, 3.0)], -1)
        tr, te = slice(0, 300), slice(300, None)
        model = M.ridge_fit(Z[tr].reshape(-1, D), Y[tr].reshape(-1, 2), lam=1.0)
        s = M.ridge_predict(model, Z[te].reshape(-1, D)).reshape(-1, T, 2)
        win = M.probe_window(fidx[te], ed[te], es[te])
        self.assertTrue((win.sum(1) * stride < period).all())  # window shorter than one cycle
        for k, tgt in ((0, ed[te]), (1, es[te])):
            err = M.frame_error(M.localize(s[..., k], fidx[te], win), tgt)
            self.assertLess(err["mean"], 2.5)  # sampled grid is every 2 frames, so floor ~0.5
            bad = M.frame_error(M.localize(rng.normal(size=s[..., k].shape), fidx[te], win), tgt)
            self.assertGreater(bad["mean"], 5)  # random scores inside the same window

    def test_localize_respects_valid(self):
        s = np.array([[0.0, 5.0, 1.0]])
        fi = np.array([[10, 12, 14]])
        self.assertEqual(M.localize(s, fi)[0], 12)
        self.assertEqual(M.localize(s, fi, valid=np.array([[1, 0, 1]], bool))[0], 14)

    def test_in_clip(self):
        fi = np.array([[0, 2, 4], [0, 2, 4]])
        v = np.ones((2, 3), bool)
        np.testing.assert_array_equal(M.in_clip(fi, v, np.array([3, 9]), 2), [True, False])

    def test_heatmap_peak(self):
        h = M.heatmap_targets(np.array([[0, 2, 4, 6]]), np.array([4]), 1.0)
        self.assertEqual(h.argmax(), 2)
        self.assertAlmostEqual(h[0, 2], 1.0)


if __name__ == "__main__":
    unittest.main()
