"""Tests for the heart-rate-tuned SSM: matrix form vs recurrence, rotation trick vs complex SSM, clock."""
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cardiac  # noqa: E402,F401
from cardiac.models import count_params, make_temporal  # noqa: E402
from cardiac.sync import SyncMixer, cardiac_clock, rotate_pairs, ssd_bidirectional  # noqa: E402


def naive_bidirectional(x, dt, A, Bf, Cf, Bb, Cb, D):
    """Literal recurrences (float64): forward incl. the current frame, backward excl. it."""
    Bsz, T, H, P = x.shape
    N = Bf.shape[-1]
    y = torch.zeros_like(x)
    for b in range(Bsz):
        for h in range(H):
            st = torch.zeros(N, P, dtype=x.dtype)
            for t in range(T):  # forward
                st = torch.exp(dt[b, t, h] * A[h]) * st + dt[b, t, h] * Bf[b, t][:, None] * x[b, t, h][None]
                y[b, t, h] += Cf[b, t] @ st
            st = torch.zeros(N, P, dtype=x.dtype)
            for t in range(T - 1, -1, -1):  # backward; output read before adding frame t itself
                hb = H + h
                y[b, t, h] += Cb[b, t] @ (torch.exp(dt[b, t, hb] * A[hb]) * st)
                st = torch.exp(dt[b, t, hb] * A[hb]) * st + dt[b, t, hb] * Bb[b, t][:, None] * x[b, t, h][None]
            y[b, :, h] += D[h] * x[b, :, h]
    return y


def inputs(Bsz=2, T=19, H=3, P=4, N=6, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=dtype)  # noqa: E731
    x = r(Bsz, T, H, P)
    dt = torch.nn.functional.softplus(r(Bsz, T, 2 * H))
    A = -torch.rand(2 * H, generator=g, dtype=dtype) * 2
    return x, dt, A, r(Bsz, T, N), r(Bsz, T, N), r(Bsz, T, N), r(Bsz, T, N), r(H)


class TestSSD(unittest.TestCase):
    def test_matches_recurrence(self):
        args = inputs()
        torch.testing.assert_close(ssd_bidirectional(*args), naive_bidirectional(*args), rtol=1e-9, atol=1e-9)

    def test_rotation_trick_equals_complex_ssm(self):
        """Forward part with rotated pairs == complex-diagonal SSM with eigenvalue exp(dt A + i k w)."""
        x, dt, A, Bf, Cf, _, _, D = inputs(Bsz=1, T=23, H=2, P=3, N=4, seed=1)
        Bsz, T, H, P = x.shape
        omega, k = 0.37, torch.tensor([1.0, 2.0], dtype=torch.float64)       # 2 pairs = 4 state dims
        t = torch.arange(T, dtype=torch.float64)
        ang = omega * t[None, :, None] * k[None, None, :]
        zeros = torch.zeros_like(Bf)
        y = ssd_bidirectional(x, dt, A, rotate_pairs(Bf, ang), rotate_pairs(Cf, ang), zeros, zeros, D * 0)
        beta = torch.complex(Bf[..., 0::2], Bf[..., 1::2])                    # (1, T, 2)
        gamma = torch.complex(Cf[..., 0::2], Cf[..., 1::2])
        ref = torch.zeros_like(x)
        for h in range(H):
            st = torch.zeros(2, P, dtype=torch.complex128)
            for i in range(T):
                lam = torch.exp(dt[0, i, h] * A[h] + 1j * omega * k)           # (2,)
                st = lam[:, None] * st + dt[0, i, h] * beta[0, i].conj()[:, None] * x[0, i, h][None]
                ref[0, i, h] = (gamma[0, i][:, None] * st).sum(0).real
        torch.testing.assert_close(y, ref, rtol=1e-9, atol=1e-9)

    def test_phase_offset_invariance(self):
        """Only phase differences matter: adding a constant angle leaves the output unchanged."""
        x, dt, A, Bf, Cf, Bb, Cb, D = inputs(seed=2)
        T = x.shape[1]
        ang = 0.4 * torch.arange(T, dtype=torch.float64)[None, :, None].expand(2, T, 3)
        rot = lambda v, a: rotate_pairs(v, a)  # noqa: E731
        y0 = ssd_bidirectional(x, dt, A, rot(Bf, ang), rot(Cf, ang), rot(Bb, ang), rot(Cb, ang), D)
        a1 = ang + 1.234
        y1 = ssd_bidirectional(x, dt, A, rot(Bf, a1), rot(Cf, a1), rot(Bb, a1), rot(Cb, a1), D)
        torch.testing.assert_close(y0, y1, rtol=1e-9, atol=1e-9)


class TestClock(unittest.TestCase):
    def _clips(self, periods, T, C=48, harm2=0.5, noise=0.3, seed=0):
        g = torch.Generator().manual_seed(seed)
        t = torch.arange(T, dtype=torch.float32)
        out = []
        for p in periods:
            a = torch.randn(C, generator=g)
            ph1, ph2 = torch.rand(C, generator=g) * 6.28, torch.rand(C, generator=g) * 6.28
            x = (a[None] * torch.cos(2 * math.pi * t[:, None] / p + ph1[None])
                 + harm2 * a[None] * torch.cos(4 * math.pi * t[:, None] / p + ph2[None])
                 + noise * torch.randn(T, C, generator=g) + 3.0)
            out.append(x)
        return torch.stack(out)

    def test_recovers_period(self):
        # 1.2-3.5 beats per clip, 2nd harmonic at half the fundamental's amplitude, noise, offset
        for T, periods in ((32, [9.0, 14.0, 21.0, 26.0]), (64, [12.0, 24.0, 37.0])):
            om, _ = cardiac_clock(self._clips(periods, T))
            est = 2 * math.pi / om
            rel = (est - torch.tensor(periods)).abs() / torch.tensor(periods)
            self.assertTrue(bool((rel < 0.08).all()), f"T={T}: true {periods}, estimated {est.tolist()}")

    def test_dominant_fundamental_with_harmonic(self):
        periods = [16.0, 22.0]
        om, _ = cardiac_clock(self._clips(periods, 64, harm2=0.8))
        est = 2 * math.pi / om
        rel = (est - torch.tensor(periods)).abs() / torch.tensor(periods)
        self.assertTrue(bool((rel < 0.05).all()), f"estimated {est.tolist()}")


class TestBlock(unittest.TestCase):
    def test_param_count_matches_baselines(self):
        n = count_params(make_temporal("sync", 512, 256, 2))
        self.assertEqual(n, 1_022_112)                            # BiGRU 1,021,184; Mamba 1,007,616
        self.assertEqual(count_params(make_temporal("sync", 512, 256, 2, clock="none")), n)

    def test_shapes_and_grads_all_clocks(self):
        torch.manual_seed(0)
        f = torch.randn(3, 20, 512)
        for clock in ("hr", "none", "fixed", "learned", "tempered"):
            m = make_temporal("sync", 512, 64, 2, clock=clock, vol_readout=True, aux_weight=1.0)
            z = m(f)
            self.assertEqual(tuple(z.shape), (3, 20, 64))
            self.assertEqual(tuple(m.aux.shape), (3,))
            (z.square().mean() + m.aux.square().mean()).backward()
            for name, p in m.named_parameters():
                if clock != "fixed" and name == "log_period":
                    continue
                self.assertIsNotNone(p.grad, f"{clock}: no grad for {name}")

    def test_area_readout_through_efmodel(self):
        from cardiac.models import EFModel
        torch.manual_seed(0)
        m = EFModel("sync", [30.0] * 3, [50.0] * 3, pretrained=False,
                    temporal_kw=dict(vol_readout=True, area_readout=True, aux_weight=1.0))
        v = torch.randint(0, 256, (2, 12, 3, 112, 112), dtype=torch.uint8)
        out = m(v, torch.ones(2, 12, dtype=torch.bool))
        self.assertEqual(tuple(out["ef"].shape), (2,))
        self.assertEqual(tuple(m.temporal.last_area.shape), (2, 12))
        (out["ef"].sum() + m.temporal.aux.sum()).backward()
        self.assertIsNotNone(m.temporal.mask.weight.grad)
        self.assertIsNotNone(m.encoder.stages[2][0].conv1.weight.grad)   # the mask trains the encoder

    def test_tempered_clock_limits(self):
        # gamma -> 0: the learned fixed frequency for every clip; gamma -> 1 with the base period at the
        # population median: the clip's own cardiac clock; short clips always get the base frequency
        f = TestClock()._clips([12.0, 20.0, 30.0], 48, C=512)
        m = make_temporal("sync", 512, 64, 2, clock="tempered", prior_period=22.5, min_clock_T=24)
        with torch.no_grad():
            m.temper.fill_(-30.0)
            m(f)
            torch.testing.assert_close(m.last_omega, torch.full((3,), 2 * math.pi / 24.0))
            m.temper.fill_(30.0)
            m.log_period.fill_(math.log(22.5))
            m(f)
            torch.testing.assert_close(m.last_omega, cardiac_clock(f)[0])
            m(f[:, :16])
            torch.testing.assert_close(m.last_omega, torch.full((3,), 2 * math.pi / 22.5))

    def test_clock_none_equals_zero_frequency(self):
        torch.manual_seed(0)
        mix = SyncMixer(64, clock="hr")
        u = torch.randn(2, 17, 64)
        y_zero = mix(u, torch.zeros(2))
        mix.clock = "none"
        torch.testing.assert_close(mix(u, None), y_zero)

    def test_bidirectional_sees_future(self):
        torch.manual_seed(0)
        m = make_temporal("sync", 32, 32, 1, clock="none").double()
        f = torch.randn(1, 12, 32, dtype=torch.float64, requires_grad=True)
        m(f)[0, 3].sum().backward()
        self.assertGreater(float(f.grad[0, 8:].abs().sum()), 0.0)   # output at t=3 depends on t>3


class TestRangeReadout(unittest.TestCase):
    def test_identity_at_init_and_trains(self):
        from cardiac.models import RangeReadout
        torch.manual_seed(0)
        r = RangeReadout(32, 4, trim=2)
        z = torch.randn(3, 20, 32)
        torch.testing.assert_close(r(z), z)                         # zero-initialised output layer
        r(z).sum().backward()
        self.assertGreater(float(r.inject.weight.grad.abs().sum()), 0.0)

    def test_soft_extremes_approach_hard_ones(self):
        from cardiac.models import RangeReadout
        r = RangeReadout(1, 1, tau=1e-3, trim=0)
        with torch.no_grad():
            r.proj.weight.fill_(1.0)
            r.proj.bias.zero_()
            r.inject.weight.copy_(torch.tensor([[1.0, 0.0]]))      # inject the range only
        z = torch.tensor([[[0.0], [3.0], [-1.0], [2.0]]])
        self.assertAlmostEqual(float(r(z)[0, 0, 0] - z[0, 0, 0]), 4.0, places=1)   # max 3 - min -1

    def test_baselines_accept_range(self):
        for kind, nl in (("bigru", 3), ("sync", 2)):
            m = make_temporal(kind, 48, 32, nl, range_k=4)
            out = m(torch.randn(2, 16, 48))
            self.assertEqual(tuple(out.shape), (2, 16, 32))
            self.assertIsNotNone(m.range)


class TestHarmonicReadout(unittest.TestCase):
    def _readout(self):
        from cardiac.models import HarmonicReadout
        r = HarmonicReadout(1, 1, harmonics=(1, 2), ridge=0.0).double()
        with torch.no_grad():
            r.proj.weight.fill_(1.0)
            r.proj.bias.zero_()
        return r

    def test_amplitude_on_non_integer_cycles_and_phase_invariance(self):
        r = self._readout()
        T, period = 32, 21.0                                        # 1.52 beats: a plain DFT would leak
        t = torch.arange(T, dtype=torch.float64)
        w = torch.tensor([2 * math.pi / period], dtype=torch.float64)
        for phase in (0.0, 1.0, 2.5):
            z = (3.0 + 2.0 * torch.cos(w * t + phase) + 0.5 * torch.sin(2 * w * t + phase)).view(1, T, 1)
            r(z, w)
            amp = r.last_amp[0, :, 0]
            self.assertAlmostEqual(float(amp[0]), 2.0, places=6)
            self.assertAlmostEqual(float(amp[1]), 0.5, places=6)

    def test_identity_at_init_and_all_clocks(self):
        from cardiac.models import HarmonicReadout
        torch.manual_seed(0)
        r = HarmonicReadout(32, 4)
        z = torch.randn(3, 20, 32)
        torch.testing.assert_close(r(z, torch.full((3,), 0.3)), z)
        for clock in ("hr", "none", "fixed", "learned"):
            m = make_temporal("sync", 48, 32, 2, clock=clock, harm_k=4, range_k=4)
            out = m(torch.randn(2, 24, 48))
            out.sum().backward()
            self.assertEqual(tuple(out.shape), (2, 24, 32))
        for hc in ("hr", "prior"):
            m = make_temporal("bigru", 48, 32, 3, harm_k=4, harm_clock=hc)
            self.assertEqual(tuple(m(torch.randn(2, 24, 48)).shape), (2, 24, 32))


if __name__ == "__main__":
    unittest.main()
