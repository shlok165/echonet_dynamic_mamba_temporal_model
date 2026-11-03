"""Model tests: scan correctness, causality/direction, shapes."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cardiac  # noqa: E402,F401
from cardiac.models import EFModel, MambaMixer, make_temporal, selective_scan  # noqa: E402


def naive_scan(u, delta, A, Bm, Cm, D):
    """Literal per-element loop of the S6 recurrence (float64) as ground truth."""
    Bsz, L, Din = u.shape
    N = A.shape[1]
    y = torch.zeros(Bsz, L, Din, dtype=torch.float64)
    for b in range(Bsz):
        h = torch.zeros(Din, N, dtype=torch.float64)
        for t in range(L):
            h = torch.exp(delta[b, t, :, None] * A) * h + delta[b, t, :, None] * Bm[b, t][None] * u[b, t, :, None]
            y[b, t] = (h * Cm[b, t][None]).sum(-1) + D * u[b, t]
    return y


class TestScan(unittest.TestCase):
    def _inputs(self, Bsz=2, L=37, Din=6, N=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        u = torch.randn(Bsz, L, Din, generator=g, dtype=torch.float64)
        delta = torch.nn.functional.softplus(torch.randn(Bsz, L, Din, generator=g, dtype=torch.float64))
        A = -torch.rand(Din, N, generator=g, dtype=torch.float64) * 2
        Bm = torch.randn(Bsz, L, N, generator=g, dtype=torch.float64)
        Cm = torch.randn(Bsz, L, N, generator=g, dtype=torch.float64)
        D = torch.randn(Din, generator=g, dtype=torch.float64)
        return u, delta, A, Bm, Cm, D

    def test_matches_naive(self):
        args = self._inputs()
        ref = naive_scan(*args)
        f32 = [a.float() for a in args]
        for m in ("sequential", "parallel"):
            out = selective_scan(*f32, method=m).double()
            torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4, msg=m)

    def test_parallel_eq_sequential_grads(self):
        args = [a.float().requires_grad_(True) for a in self._inputs(L=128, seed=1)]
        ys = selective_scan(*args, method="sequential")
        gs = torch.autograd.grad(ys.square().sum(), args)
        yp = selective_scan(*args, method="parallel")
        gp = torch.autograd.grad(yp.square().sum(), args)
        torch.testing.assert_close(yp, ys, rtol=1e-4, atol=1e-4)
        for a, b in zip(gp, gs):
            torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)

    def test_unidirectional_is_causal(self):
        torch.manual_seed(0)
        m = MambaMixer(16, d_state=4).eval()
        x = torch.randn(1, 20, 16)
        x2 = x.clone()
        x2[:, 12:] += torch.randn(1, 8, 16)
        y, y2 = m(x), m(x2)
        torch.testing.assert_close(y[:, :12], y2[:, :12])
        self.assertFalse(torch.allclose(y[:, 12:], y2[:, 12:]))

    def test_bidirectional_sees_future(self):
        torch.manual_seed(0)
        t = make_temporal("mamba", d_in=8, d_model=16, n_layers=1, d_state=4).eval()
        x = torch.randn(1, 20, 8)
        x2 = x.clone()
        x2[:, 15:] += 1.0
        self.assertFalse(torch.allclose(t(x)[:, :10], t(x2)[:, :10]))


class TestEFModel(unittest.TestCase):
    def test_shapes_all_kinds(self):
        v = torch.randint(0, 255, (2, 8, 3, 112, 112), dtype=torch.uint8)
        valid = torch.ones(2, 8, dtype=torch.bool)
        for kind in ("identity", "bigru", "mamba", "mamba_uni"):
            m = EFModel(kind, [30.0] * 3, [50.0] * 3, d_model=32, n_layers=1, pretrained=False).eval()
            with torch.no_grad():
                out = m(v, valid)
            self.assertEqual(tuple(out["ef"].shape), (2,), kind)
            self.assertEqual(tuple(out["Z"].shape), (2, 8, 32), kind)
            self.assertEqual(tuple(out["F"].shape), (2, 8, 512), kind)

    def test_frames_independent_in_encoder(self):
        m = EFModel("identity", [30.0] * 3, [50.0] * 3, d_model=32, n_layers=1, pretrained=False).eval()
        v = torch.randint(0, 255, (1, 4, 3, 112, 112), dtype=torch.uint8)
        v2 = v.clone()
        v2[:, 3] = 0
        with torch.no_grad():
            a, b = m(v)["F"], m(v2)["F"]
        torch.testing.assert_close(a[:, :3], b[:, :3])  # BN in eval mode -> per-frame encoding


if __name__ == "__main__":
    unittest.main()


class TestPanEchoEFModel(unittest.TestCase):
    """Encoder split for fine-tuning (needs the PanEcho weights file; skipped without it)."""

    @classmethod
    def setUpClass(cls):
        from cardiac import panecho
        if not Path(panecho.WEIGHTS).exists():
            raise unittest.SkipTest("PanEcho weights not found")
        cls.m = panecho.PanEchoEFModel("bigru", n_layers=1, n_tune=3)

    def test_split_and_frozen(self):
        from torchvision.models.convnext import CNBlock
        names = [type(x).__name__ for x in self.m.tuned]
        self.assertEqual(names[:5], ["LayerNorm2d", "Conv2d", "CNBlock", "CNBlock", "CNBlock"])
        self.assertEqual(sum(isinstance(x, CNBlock) for x in self.m.frozen), 15)
        self.assertFalse(any(p.requires_grad for p in self.m.frozen.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.m.tuned.parameters()))
        self.m.train()
        self.assertFalse(self.m.frozen.training)
        self.assertTrue(self.m.tuned.training)

    def test_state_dict_round_trip_without_frozen(self):
        sd = self.m.state_dict()
        self.assertFalse(any(k.startswith("frozen.") for k in sd))
        self.m.load_state_dict(sd)                    # strict: frozen keys may be missing, nothing else
        bad = dict(sd)
        bad.pop(next(k for k in sd if k.startswith("tuned.")))
        with self.assertRaises(RuntimeError):
            self.m.load_state_dict(bad)

    def test_forward_shapes_and_grads(self):
        x = torch.randint(0, 256, (1, 4, 3, 224, 224), dtype=torch.uint8)
        out = self.m.train()(x)
        self.assertEqual(tuple(out["F"].shape), (1, 4, 768))
        out["ef"].sum().backward()
        self.assertIsNotNone(next(self.m.tuned.parameters()).grad)
        self.m.zero_grad(set_to_none=True)


class TestGridFeatures(unittest.TestCase):
    def test_grid_adapter_shapes_and_position_sensitivity(self):
        from cardiac.models import FeatModel
        torch.manual_seed(0)
        for kind, nl, kw in (("sync", 2, {"clock": "fixed"}), ("bigru", 3, {})):
            m = FeatModel(kind, 768, 64, nl, {**kw, "grid_tokens": 9, "grid_dim": 16}).eval()
            x = torch.randn(2, 12, 9, 768)
            out = m(x)
            self.assertEqual(tuple(out["ef"].shape), (2,))
            self.assertEqual(tuple(out["F"].shape), (2, 12, 9 * 16))
            # swapping two grid positions changes the output: the model knows where features came from
            y = x.clone()
            y[:, :, [0, 8]] = x[:, :, [8, 0]]
            self.assertFalse(torch.allclose(m(y)["ef"], out["ef"]))


class TestPhasePool(unittest.TestCase):
    def test_phase_pool_all_clocks_and_bigru(self):
        from cardiac.models import FeatModel
        torch.manual_seed(0)
        for kind, nl, kw in (("sync", 2, {"clock": "fixed"}), ("sync", 2, {"clock": "mix"}),
                             ("sync", 2, {"clock": "hr"}), ("bigru", 3, {})):
            m = FeatModel(kind, 768, 64, nl, {**kw, "pool": "phase", "pool_k": 8})
            out = m(torch.randn(2, 32, 768), torch.ones(2, 32, dtype=torch.bool))
            self.assertEqual(tuple(out["ef"].shape), (2,))
            out["ef"].sum().backward()
            self.assertIsNotNone(m.phase.harm.proj.weight.grad, kind)
            self.assertIsNotNone(m.phase.range.proj.weight.grad, kind)
