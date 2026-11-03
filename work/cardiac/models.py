"""EF model = per-frame 2D encoder -> temporal model -> mean pool -> regression head.

Only the temporal block differs between models; encoder, pooling, head (and, in train.py, loss,
optimiser, schedule, sampling, seeds) are shared.

Temporal blocks (all map (B, T, 512) -> (B, T, d_model)):
  "identity"  linear projection only, no frame mixing (ablation: is temporal modelling used at all?)
  "uvt"       UVT-style temporal Transformer (Reynaud et al. 2021), the brief's optional extra baseline
  "bigru"     bidirectional GRU baseline
  "mamba"     pure-PyTorch Mamba (Gu & Dao 2023) selective-SSM blocks, optionally bidirectional
              (Vim/VideoMamba-style forward + backward scan with separate parameters)
  "sync"      proposed: heart-rate-tuned bidirectional SSM (cardiac/sync.py)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.checkpoint import checkpoint

from .dataset import Normalize


# ----------------------------------------------------------------------------- encoder

class ResNet18Encoder(nn.Module):
    """ImageNet ResNet-18 trunk, 112x112 input -> 512-d per frame (global average pool)."""

    def __init__(self, pretrained: bool = True, grad_ckpt: bool = False):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.stages = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])
        self.grad_ckpt = grad_ckpt
        self.out_dim = 512

    def forward(self, x, return_maps: bool = False):  # (N, 3, H, W)
        x = self.stem(x)
        maps = []
        for s in self.stages:
            if self.grad_ckpt and self.training and x.requires_grad:
                x = checkpoint(s, x, use_reentrant=False)
            else:
                x = s(x)
            maps.append(x)
        f = x.mean((2, 3))
        return (f, maps) if return_maps else f


# ----------------------------------------------------------------------------- selective scan

def selective_scan(u, delta, A, Bm, Cm, D, method: str = "parallel"):
    """Mamba S6 recurrence, computed in float32.

    u, delta: (B, L, Din); A: (Din, N); Bm, Cm: (B, L, N); D: (Din,)
      h_t = exp(delta_t A) * h_{t-1} + delta_t B_t u_t      (h_t: (B, Din, N))
      y_t = <C_t, h_t> + D u_t
    method="parallel": Hillis-Steele inclusive scan, log2(L) vectorised steps (only multiplies by
    decays in (0,1], so it is numerically stable); "sequential": plain loop (reference).
    """
    with torch.autocast(device_type=u.device.type, enabled=False):
        u, delta, Bm, Cm = u.float(), delta.float(), Bm.float(), Cm.float()
        dA = torch.exp(delta.unsqueeze(-1) * A.float())                 # (B, L, Din, N)
        dBu = (delta * u).unsqueeze(-1) * Bm.unsqueeze(2)                # (B, L, Din, N)
        if method == "sequential":
            h = torch.zeros_like(dBu[:, 0])
            hs = []
            for t in range(u.shape[1]):
                h = dA[:, t] * h + dBu[:, t]
                hs.append(h)
            h = torch.stack(hs, 1)
        elif method == "parallel":
            a, h = dA, dBu
            L = u.shape[1]
            off = 1
            while off < L:
                h_prev = F.pad(h[:, :-off], (0, 0, 0, 0, off, 0))           # h[t-off], 0 for t<off
                a_prev = F.pad(a[:, :-off], (0, 0, 0, 0, off, 0), value=1.0)
                h = h + a * h_prev
                a = a * a_prev
                off *= 2
        else:
            raise ValueError(method)
        y = (h * Cm.unsqueeze(2)).sum(-1) + u * D.float()
    return y


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class MambaMixer(nn.Module):
    """One-direction Mamba mixer (Gu & Dao 2023, Fig. 3), pure PyTorch."""

    def __init__(self, d_model, d_state=16, expand=2, d_conv=4, dt_rank="auto",
                 dt_min=1e-3, dt_max=1e-1, scan="parallel"):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.scan = scan
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # dt init as in the reference implementation: dt ~ logUniform[dt_min, dt_max], bias = softplus^-1(dt)
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)  # S4D-real init
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):  # (B, L, d_model)
        L = x.shape[1]
        xi, z = self.in_proj(x).chunk(2, dim=-1)
        xi = F.silu(self.conv1d(xi.transpose(1, 2))[..., :L].transpose(1, 2))  # causal depthwise conv
        dt, Bm, Cm = self.x_proj(xi).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        y = selective_scan(xi, dt, A, Bm, Cm, self.D, self.scan)
        y = y.to(z.dtype) * F.silu(z)
        return self.out_proj(y)


class MambaLayer(nn.Module):
    """Pre-norm residual layer. bidirectional=True: forward and time-reversed mixers, summed."""

    def __init__(self, d_model, bidirectional=True, **kw):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.fwd = MambaMixer(d_model, **kw)
        self.bwd = MambaMixer(d_model, **kw) if bidirectional else None

    def forward(self, x):
        h = self.norm(x)
        y = self.fwd(h)
        if self.bwd is not None:
            y = y + self.bwd(h.flip(1)).flip(1)
        return x + y


class MambaTemporal(nn.Module):
    def __init__(self, d_in=512, d_model=256, n_layers=2, bidirectional=True, **kw):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([MambaLayer(d_model, bidirectional, **kw) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.out_dim = d_model

    def forward(self, f):
        x = self.inp(f)
        for l in self.layers:
            x = l(x)
        return self.norm(x)


class RangeReadout(nn.Module):
    """Soft temporal extremes of k learned projections, added back to every frame's output.

    EF is a ratio of temporal extremes (1 - ESV/EDV), but the shared head only sees a mean over time, and
    a linear selective SSM cannot compute a maximum. c_t = W z_t (k channels); per channel the soft max
    and soft min over the clip (log-sum-exp with a learned temperature, `trim` frames excluded at each
    end); [range, midrange] -> linear -> added to every frame. The output layer starts at zero, so the
    block starts out exactly as without the readout. k=16, d=256: 12,576 parameters."""

    def __init__(self, d_model, k=16, tau=0.1, trim=0):
        super().__init__()
        self.proj = nn.Linear(d_model, k)
        self.tau = nn.Parameter(torch.full((k,), math.log(math.expm1(tau))))   # softplus^-1(tau)
        self.inject = nn.Linear(2 * k, d_model)
        nn.init.zeros_(self.inject.weight)
        nn.init.zeros_(self.inject.bias)
        self.trim = trim

    def stats(self, z):
        """[soft range, soft midrange] of the k channels over the clip: (B, 2k), float32."""
        with torch.autocast(device_type=z.device.type, enabled=False):
            c = self.proj(z.float())
            T = c.shape[1]
            k = self.trim if T > 2 * self.trim + 2 else 0
            c = c[:, k:T - k]
            tau = F.softplus(self.tau.float()) + 1e-3
            n = math.log(c.shape[1])
            smax = tau * (torch.logsumexp(c / tau, 1) - n)
            smin = -tau * (torch.logsumexp(-c / tau, 1) - n)
            return torch.cat((smax - smin, 0.5 * (smax + smin)), -1)

    def forward(self, z):
        with torch.autocast(device_type=z.device.type, enabled=False):
            g = self.inject(self.stats(z))
        return z + g[:, None, :].to(z.dtype)


class HarmonicReadout(nn.Module):
    """Oscillation amplitude of k learned channels at the clip's own heart rate, added back to every frame.

    c_t = W z_t (k channels). With the clip's cardiac frequency w (rad per sampled frame, from the
    label-free clock cardiac.sync.cardiac_clock), each channel is fitted by least squares with
    a0 + sum_h [a_h cos(h w t) + b_h sin(h w t)], h in `harmonics` (ridge `ridge`; exact for clips that
    hold a non-integer number of beats, where a plain DFT leaks). The amplitudes sqrt(a_h^2 + b_h^2) do
    not depend on where in the cycle the clip starts and use every frame (noise averages out), unlike
    the extremes; for the log-volume curve the fundamental amplitude tracks log(EDV/ESV).
    [amplitudes] -> linear (zero-initialised) -> added to every frame. k=16, 2 harmonics, d=256: 12,560
    parameters."""

    def __init__(self, d_model, k=16, harmonics=(1, 2), ridge=1e-3):
        super().__init__()
        self.proj = nn.Linear(d_model, k)
        self.register_buffer("harm", torch.tensor([float(h) for h in harmonics]), persistent=False)
        self.inject = nn.Linear(k * len(harmonics), d_model)
        nn.init.zeros_(self.inject.weight)
        nn.init.zeros_(self.inject.bias)
        self.ridge = ridge
        self.last_amp = None

    def forward(self, z, omega):
        with torch.autocast(device_type=z.device.type, enabled=False):
            g = self.inject(self.amplitudes(z, omega))
        return z + g[:, None, :].to(z.dtype)

    def amplitudes(self, z, omega):
        """Least-squares oscillation amplitudes of the k channels at harmonics of omega: (B, H * k), float32."""
        with torch.autocast(device_type=z.device.type, enabled=False):
            c = self.proj(z.to(self.proj.weight.dtype))                                 # (B, T, k)
            B, T, _ = c.shape
            t = torch.arange(T, device=c.device, dtype=c.dtype) - (T - 1) / 2
            ang = omega.to(c.dtype)[:, None, None] * t[None, :, None] * self.harm.to(c.dtype)[None, None, :]
            X = torch.cat((torch.ones_like(ang[..., :1]), torch.cos(ang), torch.sin(ang)), -1)  # (B, T, 1+2H)
            G = X.transpose(1, 2) @ X / T + self.ridge * torch.eye(X.shape[-1], device=c.device, dtype=c.dtype)
            # solve_ex: no singularity check (G is ridge-regularised), so no GPU->CPU sync per forward
            coef = torch.linalg.solve_ex(G, X.transpose(1, 2) @ c / T)[0]              # (B, 1+2H, k)
            nh = self.harm.numel()
            amp = (coef[:, 1:1 + nh].square() + coef[:, 1 + nh:].square() + 1e-8).sqrt()  # (B, H, k)
            self.last_amp = amp.detach()
            return amp.flatten(1)


def clip_omega(f, kind="hr", prior_period=22.5, min_T=0):
    """Cardiac frequency (rad per sampled frame) of each clip for a HarmonicReadout: "hr" = the label-free
    clock (cardiac.sync.cardiac_clock on the input features; clips shorter than min_T frames fall back to
    the prior), "prior" = the same fixed frequency 2 pi / prior_period for every clip (ablation)."""
    from .sync import TWO_PI, cardiac_clock
    if kind == "prior" or f.shape[1] < min_T:
        return torch.full((f.shape[0],), TWO_PI / prior_period, device=f.device)
    return cardiac_clock(f.detach())[0]


# ----------------------------------------------------------------------------- baselines

class BiGRUTemporal(nn.Module):
    def __init__(self, d_in=512, d_model=256, n_layers=2, dropout=0.0, range_k=0, range_trim=2, harm_k=0,
                 harm_clock="hr"):
        """range_k > 0 adds a RangeReadout, harm_k > 0 a HarmonicReadout driven by the same label-free clock
        as the sync block (ablations: the readouts on the baseline)."""
        super().__init__()
        assert d_model % 2 == 0
        self.inp = nn.Linear(d_in, d_model)
        self.gru = nn.GRU(d_model, d_model // 2, num_layers=n_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(d_model)
        self.range = RangeReadout(d_model, range_k, trim=range_trim) if range_k else None
        self.harm = HarmonicReadout(d_model, harm_k) if harm_k else None
        self.harm_clock = harm_clock
        self.out_dim = d_model

    def forward(self, f):
        x = self.inp(f)
        # cuDNN GRU: run in float32 so the result does not depend on bf16 RNN support
        with torch.autocast(device_type=x.device.type, enabled=False):
            y, _ = self.gru(x.float())
        y = self.norm(y)
        if self.range is not None:
            y = self.range(y)
        if self.harm is not None:
            y = self.harm(y, clip_omega(f, self.harm_clock))
        return y


class UVTTemporal(nn.Module):
    """Temporal Transformer baseline in the style of UVT (Reynaud et al., MICCAI 2021): post-LN BERT
    layers over per-frame features, output = mean of all hidden states (input + every layer), as UVT.
    Unlike the original UVT (which uses no positional information), learned absolute positions are added
    (BERT embedding style: + pos, LayerNorm, dropout) so the block is a temporal model; pos_emb=False
    reproduces the original's order-blind behaviour. Trained with EF loss only, like the other blocks."""

    # defaults: 2 layers, FFN 320, 256 positions -> 1,055,104 params (BiGRU 1,021,184; Mamba 1,007,616)
    def __init__(self, d_in=512, d_model=256, n_layers=2, heads=4, intermediate=320, dropout=0.1,
                 max_len=256, pos_emb=True):
        super().__init__()
        from .uvt import BertEncoder
        self.inp = nn.Linear(d_in, d_model)
        self.pos = nn.Embedding(max_len, d_model) if pos_emb else None
        if self.pos is not None:
            nn.init.normal_(self.pos.weight, 0.0, 0.02)
        self.emb_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.drop = nn.Dropout(dropout)
        self.T = BertEncoder(d_model, n_layers, heads, intermediate, dropout, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, f):
        x = self.inp(f)
        if self.pos is not None:
            x = x + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        x = self.drop(self.emb_norm(x))
        return self.norm(torch.stack(self.T(x)).mean(0))


class IdentityTemporal(nn.Module):
    def __init__(self, d_in=512, d_model=256):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, f):
        return self.norm(self.inp(f))


def make_temporal(kind: str, d_in=512, d_model=256, n_layers=2, **kw) -> nn.Module:
    if kind == "bigru":
        return BiGRUTemporal(d_in, d_model, n_layers, **kw)
    if kind == "mamba":
        return MambaTemporal(d_in, d_model, n_layers, bidirectional=True, **kw)
    if kind == "mamba_uni":
        return MambaTemporal(d_in, d_model, n_layers, bidirectional=False, **kw)
    if kind == "uvt":
        return UVTTemporal(d_in, d_model, n_layers, **kw)
    if kind == "sync":
        from .sync import SyncTemporal
        return SyncTemporal(d_in, d_model, n_layers, **kw)
    if kind == "identity":
        return IdentityTemporal(d_in, d_model)
    raise ValueError(kind)


# ----------------------------------------------------------------------------- full model

class EFModel(nn.Module):
    def __init__(self, temporal: str, mean, std, d_model=256, n_layers=2, pretrained=True,
                 grad_ckpt=False, temporal_kw=None):
        super().__init__()
        self.normalize = Normalize(mean, std)
        self.encoder = ResNet18Encoder(pretrained, grad_ckpt)
        self.temporal = make_temporal(temporal, self.encoder.out_dim, d_model, n_layers, **(temporal_kw or {}))
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, video, valid=None):
        """video: uint8 (B, T, 3, H, W). valid: optional (B, T) bool; pooling then averages valid frames.
        Returns dict(ef=(B,) standardised EF, Z=(B,T,d), F=(B,T,512))."""
        B, T = video.shape[:2]
        x = self.normalize(video).flatten(0, 1).contiguous(memory_format=torch.channels_last)
        if getattr(self.temporal, "wants_maps", False):
            # the block also reads the encoder's spatial maps (same encoder, same pooled F_t)
            f, maps = self.encoder(x, return_maps=True)
            f = f.view(B, T, -1)
            z = self.temporal(f, maps=[m.view(B, T, *m.shape[1:]) for m in maps])
        else:
            f = self.encoder(x).view(B, T, -1)
            z = self.temporal(f)
        if valid is None:
            pooled = z.mean(1)
        else:
            w = valid.to(z.dtype).unsqueeze(-1)
            pooled = (z * w).sum(1) / w.sum(1).clamp_min(1)
        return {"ef": self.head(pooled).squeeze(-1), "Z": z, "F": f}


class GridAdapter(nn.Module):
    """Spatial grid features (B, T, G, d_in) -> (B, T, G * d_tok): a shared Linear per grid position, plus a
    learned embedding of the position, GELU, then the positions concatenated (keeps where in the image each
    feature came from, which a global average pool discards)."""

    def __init__(self, d_in, n_tokens, d_tok=96, drop=0.0):
        super().__init__()
        self.proj = nn.Linear(d_in, d_tok)
        self.pos = nn.Parameter(torch.randn(n_tokens, d_tok) * 0.02)
        self.drop = nn.Dropout(drop)

    def forward(self, f):
        return self.drop(F.gelu(self.proj(f) + self.pos)).flatten(2)


class PhasePool(nn.Module):
    """Read-out at the cardiac frequency: least-squares amplitudes of k learned channels at w and 2w
    (HarmonicReadout) and their soft temporal range/midrange (RangeReadout), (B, 4k), appended to the mean
    pool so the head sees the oscillation EF is made of, not only its average. w = the block's own clock
    (sync: last_omega; for a per-pair clock, its last pair = the clip's clock) or, for blocks without one,
    the label-free clock of the input features."""

    def __init__(self, d_model, k=16):
        super().__init__()
        self.harm = HarmonicReadout(d_model, k)
        self.range = RangeReadout(d_model, k, trim=2)
        del self.harm.inject, self.range.inject            # only their statistics are used
        self.out_dim = 4 * k

    def forward(self, z, f, temporal):
        w = getattr(temporal, "last_omega", None)
        if w is None:
            w = clip_omega(f, "hr")
        elif w.dim() == 2:
            w = w[:, -1]
        return torch.cat((self.harm.amplitudes(z, w), self.range.stats(z)), -1)


class FeatModel(nn.Module):
    """Frozen-encoder variant: the same temporal block, mean pool and head on cached per-frame features
    (scripts/cache_features.py); returns the same dict as EFModel (F = the cached features).
    temporal_kw grid_tokens=G*G (grid_dim=96): input is a spatial grid cache (B, T, G*G, d_in), read through a
    GridAdapter before the temporal block."""

    def __init__(self, temporal: str, d_in: int = 768, d_model=256, n_layers=2, temporal_kw=None):
        super().__init__()
        kw = dict(temporal_kw or {})
        g, gd, gp = kw.pop("grid_tokens", 0), kw.pop("grid_dim", 96), kw.pop("grid_drop", 0.0)
        pool, pool_k = kw.pop("pool", "mean"), kw.pop("pool_k", 16)
        self.grid = GridAdapter(d_in, g, gd, gp) if g else None
        self.temporal = make_temporal(temporal, g * gd if g else d_in, d_model, n_layers, **kw)
        self.phase = PhasePool(d_model, pool_k) if pool == "phase" else None
        d_head = d_model + (self.phase.out_dim if self.phase is not None else 0)
        self.head = nn.Sequential(nn.Linear(d_head, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, video, valid=None):
        f = video.float()
        if self.grid is not None:
            f = self.grid(f)
        z = self.temporal(f)
        if valid is None:
            pooled = z.mean(1)
        else:
            w = valid.to(z.dtype).unsqueeze(-1)
            pooled = (z * w).sum(1) / w.sum(1).clamp_min(1)
        if self.phase is not None:
            pooled = torch.cat((pooled, self.phase(z, f, self.temporal).to(pooled.dtype)), -1)
        return {"ef": self.head(pooled).squeeze(-1), "Z": z, "F": f}


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
