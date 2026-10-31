"""Heart-rate-tuned bidirectional state space model ("sync") for echocardiography video.

Idea. An echo clip is quasi-periodic: every cardiac phase recurs once per beat, and EF compares two
phases (end-diastole vs end-systole). A temporal model that measures time in frames has to learn the
ED-ES lag separately for every heart rate. Here part of the state rotates at the clip's own cardiac
frequency w and its harmonics, so the mixing weight between frames t and s depends on their *cardiac
phase difference*:

    y_t = sum_s L_ts dt_s x_s [ C0_t . B0_s  +  sum_j Cj_t . R(k_j w (s - t)) Bj_s ]

R = 2x2 rotation, k_j = harmonic of state pair j, L_ts = selective decay (Mamba-2 style, one scalar
per head and direction). This is a complex-diagonal SSM written with the "RoPE trick" of Mamba-3
(Lahoti et al. 2026, Prop. 3): rotating B_s by +k w s and C_t by +k w t. The difference is where the
angle comes from: Mamba-3 learns a per-step angle from the local input, whereas here w is a global,
physiological clock estimated from the whole clip (cardiac_clock), without labels or parameters.
Frames one beat apart then reinforce each other (beat averaging), and the ED-ES comparison becomes a
fixed phase offset instead of a heart-rate-dependent lag.

The mixer is evaluated in its exact matrix ("attention-like") form, O(T^2) per head, which for clip
lengths up to a few hundred frames is faster on a GPU than a sequential or log-depth scan.

Optional volumetric readout (vol_readout=True): the block also emits a per-frame log-volume curve
l_t and the physical EF of that curve, EF = 1 - exp(-(softmax_t l - softmin_t l)), i.e.
1 - ESV/EDV; it is added to every frame's output (so the shared mean-pool + head can use it) and,
with aux_weight > 0, supervised by the EF label itself (no ED/ES labels).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from .models import HarmonicReadout, RangeReadout, RMSNorm, clip_omega

TWO_PI = 2 * math.pi


# ----------------------------------------------------------------------------- cardiac clock

@torch.no_grad()
def cardiac_clock(x, min_period: float = 6.0, max_period: float = 48.0, n_grid: int = 160, n_harm: int = 1,
                  detrend: bool = False, window: str = "none", octave_ratio: float | None = None):
    """Dominant cardiac angular frequency of each clip, in radians per (sampled) frame.

    x: (B, T, C) per-frame features (or flattened pixels). Every channel is mean-centred (optionally
    linearly detrended and windowed); the power spectrum on a log-spaced period grid
    [min_period, max_period] is summed over channels (and, if n_harm > 1, over harmonics with weight
    1/k). The peak is refined by parabolic interpolation in log-period. octave_ratio: if the power at
    half the peak frequency exceeds octave_ratio x the peak power, the half frequency is taken (the
    peak was the 2nd harmonic). No parameters, no labels, no gradient.
    Defaults chosen on VAL videos only (scripts/check_clock.py): on 1-2-beat clips, detrending and a
    Hann window remove/blur the fundamental and cause octave errors; the plain periodogram of the
    fundamental was best (median error ~5% against whole-video periods), octave correction hurt.
    Returns (omega (B,), peakiness (B,) = peak / mean of the summed spectrum).
    """
    x = x.float()
    B, T, C = x.shape
    dev = x.device
    t = torch.arange(T, device=dev, dtype=torch.float32)
    tc = t - t.mean()
    x = x - x.mean(1, keepdim=True)
    if detrend:
        x = x - (x * tc[None, :, None]).sum(1, keepdim=True) / (tc ** 2).sum().clamp_min(1e-6) * tc[None, :, None]
    if window == "hann" and T > 2:
        x = x * torch.hann_window(T, periodic=False, device=dev)[None, :, None]
    log_p = torch.linspace(math.log(max_period), math.log(min_period), n_grid, device=dev)
    om = TWO_PI / log_p.exp()                                                   # (G,) increasing
    k = torch.arange(1, n_harm + 1, device=dev, dtype=torch.float32)
    ang = (k[:, None] * om[None, :]).reshape(-1, 1) * t[None, :]                # (K*G, T)
    re = torch.einsum("gt,btc->bgc", torch.cos(ang), x)
    im = torch.einsum("gt,btc->bgc", torch.sin(ang), x)
    P = (re.square() + im.square()).sum(-1).view(B, n_harm, n_grid)             # (B, K, G)
    ok = (k[:, None] * om[None, :] <= math.pi).float()                          # below Nyquist
    S = (P * ok / k[None, :, None]).sum(1)                                       # (B, G)
    g = S.argmax(1)
    # parabolic refinement of the peak (in log S over log-period); edges are left unrefined
    gi = g.clamp(1, n_grid - 2)
    ls = S.clamp_min(1e-30).log()
    y0, y1, y2 = ls.gather(1, (gi - 1)[:, None]), ls.gather(1, gi[:, None]), ls.gather(1, (gi + 1)[:, None])
    den = (y0 - 2 * y1 + y2).squeeze(1)
    delta = torch.where(den < 0, 0.5 * (y0 - y2).squeeze(1) / den.clamp(max=-1e-12), torch.zeros_like(den))
    delta = torch.where((g > 0) & (g < n_grid - 1), delta.clamp(-0.5, 0.5), torch.zeros_like(delta))
    step = log_p[1] - log_p[0]
    log_period = log_p[g] + delta * step
    peak = S.gather(1, g[:, None]).squeeze(1) / S.mean(1).clamp_min(1e-30)
    if octave_ratio is not None:
        # grid index of twice the period (half the frequency); log-spaced grid -> constant offset
        j = g - int(round(math.log(2.0) / float(-step)))
        ok2 = j >= 0
        p2 = S.gather(1, j.clamp_min(0)[:, None]).squeeze(1)
        use = ok2 & (p2 > octave_ratio * S.gather(1, g[:, None]).squeeze(1))
        log_period = torch.where(use, log_p[j.clamp_min(0)], log_period)
    return TWO_PI / log_period.exp(), peak


def rotate_pairs(v, ang):
    """Rotate consecutive pairs (2j, 2j+1) of the last dim of v (..., 2R) by ang (..., R)."""
    return rotate_leading_pairs(v, torch.cos(ang), torch.sin(ang), ang.shape[-1])


def rotate_leading_pairs(v, cos, sin, R):
    """Rotate pairs (2j, 2j+1), j < R, of the last dim of v (B, T, ..., N); cos/sin (B, T, R).
    One fused expression for all of B and C of both directions (few kernel launches)."""
    head = v[..., :2 * R].unflatten(-1, (R, 2))
    v1, v2 = head[..., 0], head[..., 1]
    shape = cos.shape[:2] + (1,) * (v.dim() - 3) + cos.shape[2:]
    c, s = cos.view(shape), sin.view(shape)
    head = torch.stack((v1 * c - v2 * s, v1 * s + v2 * c), -1).flatten(-2)
    return torch.cat((head, v[..., 2 * R:]), -1) if v.shape[-1] > 2 * R else head


# ----------------------------------------------------------------------------- SSD core

def ssd_bidirectional(x, dt, A, Bf, Cf, Bb, Cb, D):
    """Bidirectional selective SSM, matrix form, float32.

    x: (B, T, H, P); dt: (B, T, 2H) positive (forward heads first); A: (2H,) negative;
    Bf, Cf, Bb, Cb: (B, T, N) (already rotated if a clock is used); D: (H,).
    Forward (causal, includes s = t):  h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t,  y_t = C_t . h_t
    Backward (anti-causal, excludes s = t): the same recurrence run from the end of the clip.
    """
    return ssd_core(x, dt, A, torch.stack((Bf, Bb), 2), torch.stack((Cf, Cb), 2), D)


def ssd_core(x, dt, A, Bs, Cs, D):
    """ssd_bidirectional with B/C stacked as (B, T, 2, N) (forward, backward)."""
    Bsz, T, H, P = x.shape
    a = dt * A                                                                    # (B, T, 2H) <= 0
    cs = a.cumsum(1)
    # forward: inclusive cumsum, L[t,s] = exp(c[t] - c[s]), s <= t
    # backward: exclusive cumsum, L[t,s] = exp(c[s] - c[t]), s > t
    c = torch.cat((cs[..., :H], cs[..., H:] - a[..., H:]), -1).transpose(1, 2).unflatten(1, (2, H))
    diff = c[..., :, None] - c[..., None, :]                                      # (B, 2, H, T, T)
    ti = torch.arange(T, device=x.device)
    low = ti[:, None] >= ti[None, :]                                              # [t, s]: s <= t
    keep = torch.stack((low, ~low))[None, :, None]                                # (1, 2, 1, T, T)
    sign = (1 - 2 * torch.arange(2, device=x.device, dtype=diff.dtype)).view(1, 2, 1, 1, 1)  # no H2D copy
    L = torch.where(keep, diff * sign, float("-inf")).exp()
    G = torch.einsum("btkn,bskn->bkts", Cs, Bs)[:, :, None]                       # (B, 2, 1, T, T)
    M = (L * G * dt.transpose(1, 2).unflatten(1, (2, H))[..., None, :]).sum(1)  # (B, H, T, T)
    xh = x.transpose(1, 2)                                                        # (B, H, T, P)
    y = M @ xh + D[None, :, None, None] * xh
    return y.transpose(1, 2)                                                      # (B, T, H, P)


# ----------------------------------------------------------------------------- mixer / block

class SyncMixer(nn.Module):
    """Bidirectional selective SSM mixer with heart-rate-tuned rotating state pairs.

    The input/gate projections and the local conv are shared by both directions; each direction has
    its own B, C (shared across heads, Mamba-2 "ngroups=1"), dt and A (one scalar per head).
    clock: "hr" rotates pair j by harmonic k_j of the supplied omega; "none" never rotates (ablation
    with identical parameters); "learned" uses Mamba-3-style data-dependent angles from the local
    input instead of the cardiac clock (ablation; adds a small projection).
    """

    def __init__(self, d_model, expand=2, headdim=32, d_state=16, rot_pairs=4, harmonics=(1, 2),
                 d_conv=3, dt_min=1e-3, dt_max=1e-1, A_init_range=(1.0, 16.0), clock="hr", conv_pad="zeros"):
        """conv_pad="replicate" pads the local temporal conv with the edge frames instead of zeros, which
        removes a systematic artefact at the first/last frame of the clip."""
        super().__init__()
        self.d_inner = expand * d_model
        assert self.d_inner % headdim == 0 and 2 * rot_pairs <= d_state and d_conv % 2 == 1
        self.H, self.P, self.N, self.R = self.d_inner // headdim, headdim, d_state, rot_pairs
        self.clock = clock
        harm = torch.tensor([float(harmonics[j % len(harmonics)]) for j in range(rot_pairs)])
        self.register_buffer("harm", harm, persistent=False)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv // 2, groups=self.d_inner,
                              padding_mode=conv_pad)
        self.x_proj = nn.Linear(self.d_inner, 4 * self.N + 2 * self.H, bias=False)
        self.theta_proj = nn.Linear(self.d_inner, rot_pairs) if clock == "learned" else None
        dt = torch.exp(torch.rand(2 * self.H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))           # softplus^-1(dt)
        self.A_log = nn.Parameter(torch.log(torch.empty(2 * self.H).uniform_(*A_init_range)))
        self.D = nn.Parameter(torch.ones(self.H))
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def rotation(self, omega, T, device):
        """(cos, sin) of the clock angle k_j * omega * t, each (B, T, R); None if not rotating.
        Depends only on the clip's clock, so SyncTemporal computes it once for all layers."""
        if self.R == 0 or self.clock in ("none", "learned") or omega is None:
            return None
        t = torch.arange(T, device=device, dtype=torch.float32)
        ang = omega.float()[:, None, None] * t[None, :, None] * self.harm[None, None, :]
        return torch.cos(ang), torch.sin(ang)

    def forward(self, u, omega=None, rot=None):
        Bsz, T, _ = u.shape
        x, z = self.in_proj(u).chunk(2, dim=-1)
        x = F.silu(self.conv(x.transpose(1, 2)).transpose(1, 2))
        if self.clock == "learned":
            # Mamba-3-style: angle increments from the local input, accumulated over time
            ang = torch.cumsum(math.pi * torch.tanh(self.theta_proj(x).float()), 1)
            rot = (torch.cos(ang), torch.sin(ang))
        elif self.clock == "none":
            rot = None
        elif rot is None:
            rot = self.rotation(omega, T, u.device)
        p = self.x_proj(x)
        with torch.autocast(device_type=u.device.type, enabled=False):
            p = p.float()
            BC = p[..., :4 * self.N].unflatten(-1, (4, self.N))                  # Bf, Cf, Bb, Cb
            if rot is not None:
                BC = rotate_leading_pairs(BC, rot[0], rot[1], self.R)
            dt = F.softplus(p[..., 4 * self.N:] + self.dt_bias.float())
            A = -torch.exp(self.A_log.float())
            y = ssd_core(x.float().view(Bsz, T, self.H, self.P), dt, A, BC[:, :, 0::2], BC[:, :, 1::2],
                         self.D.float()).reshape(Bsz, T, self.d_inner)
        y = self.norm(y.to(z.dtype) * F.silu(z))
        return self.out_proj(y)


class SyncLayer(nn.Module):
    def __init__(self, d_model, **kw):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = SyncMixer(d_model, **kw)

    def forward(self, x, omega=None, rot=None):
        return x + self.mixer(self.norm(x), omega, rot)


class SyncTemporal(nn.Module):
    """(B, T, d_in) per-frame features -> (B, T, d_model).

    clock: "hr" (cardiac clock from the input features; the proposed model), "none" (no rotation),
    "fixed" (one learned frequency shared by all clips, i.e. an LTI complex SSM; ablation),
    "learned" (Mamba-3-style data-dependent angles; ablation),
    "tempered" (the fixed frequency scaled by the clip's heart rate relative to the population median
    raised to a learned power gamma in (0, 1): gamma 0 = "fixed", gamma 1 = "hr" up to the base frequency;
    gamma < 1 also shrinks the effect of an octave error of the clock; init gamma 0.5).
    """

    # defaults: 2 layers, expand 2, d_state 16 -> 1,022,112 params (BiGRU 1,021,184; Mamba 1,007,616)
    def __init__(self, d_in=512, d_model=256, n_layers=2, clock="hr", min_period=6.0, max_period=48.0,
                 vol_readout=False, aux_weight=0.0, vol_tau=0.1, min_clock_T=0, prior_period=22.5,
                 area_readout=False, area_level=2, area_gamma=1.5, vol_trim=0, range_k=0, range_trim=2, harm_k=0,
                 harm_clock="hr", **kw):
        """min_clock_T: clips shorter than this many frames cannot contain a beat, so the clock falls back
        to prior_period (median VAL period in sampled frames at stride 2, scripts/check_clock.py).
        area_readout (needs vol_readout): the log-volume curve is gamma * log(area of a learned soft mask
        on encoder stage `area_level`'s map) plus a learned per-frame correction from the SSM output
        (initialised to zero), i.e. single-plane area-length volumes V ~ A^1.5 from a weakly supervised
        mask; gamma is learned (init 1.5).
        vol_trim: frames excluded at each end of the clip from the soft range (boundary artefacts).
        range_k > 0: soft-extreme readout of range_k learned channels (models.RangeReadout), applied to
        the block output before the volumetric readout.
        harm_k > 0: heart-rate harmonic-amplitude readout (models.HarmonicReadout) of harm_k channels; its
        frequency is the label-free clock ("hr", also when the mixer's clock is none/learned) or the prior
        period for every clip ("prior", ablation)."""
        super().__init__()
        assert vol_readout or not area_readout
        self.vol_trim = vol_trim
        self.clock = clock
        self.clock_kw = dict(min_period=min_period, max_period=max_period)
        self.min_clock_T, self.prior_period = min_clock_T, prior_period
        self.inp = nn.Linear(d_in, d_model)
        mixer_clock = "hr" if clock in ("hr", "fixed", "tempered") else clock
        self.layers = nn.ModuleList([SyncLayer(d_model, clock=mixer_clock, **kw) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.range = RangeReadout(d_model, range_k, trim=range_trim) if range_k else None
        self.harm = HarmonicReadout(d_model, harm_k) if harm_k else None
        self.harm_clock = harm_clock
        self.log_period = nn.Parameter(torch.tensor(math.log(24.0))) if clock in ("fixed", "tempered") else None
        self.temper = nn.Parameter(torch.tensor(0.0)) if clock == "tempered" else None   # gamma = sigmoid
        self.vol_readout, self.aux_weight = vol_readout, aux_weight
        if vol_readout:
            self.vol = nn.Linear(d_model, 1)
            self.vol_tau = nn.Parameter(torch.tensor(math.log(math.expm1(vol_tau))))  # softplus^-1(tau)
            self.vol_inject = nn.Parameter(torch.randn(d_model) * 0.02)
        self.area_readout, self.area_level = area_readout, area_level
        self.wants_maps = area_readout           # EFModel then passes the encoder's spatial maps
        if area_readout:
            self.mask = nn.Conv2d((64, 128, 256, 512)[area_level], 1, 1)
            self.log_gamma = nn.Parameter(torch.tensor(math.log(area_gamma)))
            nn.init.zeros_(self.vol.weight)      # start from the pure area curve
            nn.init.zeros_(self.vol.bias)
        self.out_dim = d_model
        self.aux = None          # standardised physical EF of the last batch (vol_readout)
        self.last_omega = None   # clock of the last batch (diagnostics)
        self.last_ell = None     # log-volume curve of the last batch (diagnostics)
        self.last_area = None    # mask area curve of the last batch (diagnostics)
        self.last_mask = None    # soft masks (B, T, h, w) of the last eval batch (diagnostics)

    def forward(self, f, maps=None):
        x = self.inp(f)
        omega = None
        if self.clock == "hr" and f.shape[1] >= self.min_clock_T:
            omega, _ = cardiac_clock(f.detach(), **self.clock_kw)
        elif self.clock == "hr":
            omega = torch.full((f.shape[0],), TWO_PI / self.prior_period, device=f.device)
        elif self.clock == "fixed":
            omega = (TWO_PI / self.log_period.exp()).expand(f.shape[0])
        elif self.clock == "tempered":
            ratio = torch.ones(f.shape[0], device=f.device)          # short clip: no beat to measure
            if f.shape[1] >= self.min_clock_T:
                ratio = cardiac_clock(f.detach(), **self.clock_kw)[0] * self.prior_period / TWO_PI
            omega = TWO_PI / self.log_period.exp() * ratio.clamp_min(1e-3) ** torch.sigmoid(self.temper)
        self.last_omega = None if omega is None else omega.detach()
        rot = self.layers[0].mixer.rotation(omega, f.shape[1], f.device)       # shared by all layers
        for layer in self.layers:
            x = layer(x, omega, rot)
        z = self.norm(x)
        if self.range is not None:
            z = self.range(z)
        if self.harm is not None:
            hr_ok = self.harm_clock == "hr" and omega is not None and self.clock == "hr"
            z = self.harm(z, omega if hr_ok else clip_omega(f, self.harm_clock, self.prior_period, self.min_clock_T))
        if self.vol_readout:
            with torch.autocast(device_type=z.device.type, enabled=False):
                ell = self.vol(z.float()).squeeze(-1)                               # log-volume curve
                if self.area_readout:
                    m = maps[self.area_level]                                       # (B, T, C, h, w)
                    soft = torch.sigmoid(self.mask(m.flatten(0, 1).float()))        # (B*T, 1, h, w)
                    area = soft.mean((1, 2, 3)).view(m.shape[0], m.shape[1])
                    self.last_area = area.detach()
                    if not self.training:
                        self.last_mask = soft.detach().view(m.shape[0], m.shape[1], *soft.shape[2:])
                    ell = self.log_gamma.exp() * torch.log(area + 1e-4) + ell
                tau = F.softplus(self.vol_tau.float()) + 1e-3
                k = self.vol_trim if ell.shape[1] > 2 * self.vol_trim + 2 else 0
                core = ell[:, k:ell.shape[1] - k]
                T = core.shape[1]
                smax = tau * (torch.logsumexp(core / tau, 1) - math.log(T))
                smin = -tau * (torch.logsumexp(-core / tau, 1) - math.log(T))
                ef = 100 * (1 - torch.exp(-(smax - smin).clamp_min(0)))             # 1 - ESV/EDV
                e = (ef - config.EF_MEAN) / config.EF_STD
            self.aux = e
            self.last_ell = ell.detach()
            z = z + (e[:, None, None] * self.vol_inject.float()).to(z.dtype)
        return z
