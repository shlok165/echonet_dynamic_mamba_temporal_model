"""Ultrasound Video Transformer (UVT), re-implemented in plain PyTorch.

Reference: H. Reynaud, A. Vlontzos, B. Hou, A. Beqiri, P. Leeson, B. Kainz, "Ultrasound Video
Transformers for Cardiac Ejection Fraction Estimation", MICCAI 2021, arXiv:2107.00977.
Original code: https://github.com/HReynaud/UVT (MIT licence, (c) the UVT authors), which in turn uses
ResNetAE from https://github.com/farrell236/ResNetAE. This file follows Network/model.py and
ResNetAE/ResNetAE.py of that repo; module and parameter names are kept identical so the released
checkpoints (best.pt, saved from nn.DataParallel) load with `load_released_checkpoint`.

Pipeline (per frame, then over time):
  frame 3x128x128 (112x112 zero-padded by 8, intensities /255)
    -> ResNetAE encoder (4 levels x 8 residual blocks, multi-resolution skips) -> 128x8x8
    -> fc1 + tanh -> embedding (emb_dim)
  embeddings (B, F, emb_dim) -> BERT encoder (post-LN, GELU), all hidden states (input + every layer)
    averaged -> (B, F, emb_dim)
    -> "extremas" MLP -> per-frame ES/ED signal: tanh in [-1, 1] (SDmode='reg': ES=-1, ED=+1)
       or 3-class logits (SDmode='cla': other / ES / ED)
    -> "ejection" MLP -> per-frame scalar, mean over frames, sigmoid -> EF / 100

Differences from the original, all opt-in:
  * pos_emb (default False). The original passes embeddings straight into BertModel(...).encoder, which
    skips BERT's embedding layer, so NO positional information is added and attention is
    permutation-equivariant. Set pos_emb=True for learned absolute positions.
  * key_padding_mask: optional; the original never masks frames.
  * Only the MLP heads (use_conv=False, the configuration of the released weights) are implemented;
    the repo's Inception-conv heads were reported worse and are omitted.
  * BERT attention uses torch's scaled_dot_product_attention (same maths as HF BertSelfAttention).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# Configuration of the released checkpoints (UVT README "Test a network" table + main.py).
PAPER_CONFIG = dict(emb_dim=1024, num_hidden_layers=16, intermediate_size=8192, attention_heads=16)
# Released weights (download_weights.sh of the UVT repo): (sampling, SD branch) -> URL.
# M = mirror ('repeat' data mode), R = random ('sample' data mode).
RELEASED_WEIGHTS = {
    ("R", "CLA"): "https://imperialcollegelondon.box.com/shared/static/k04ye6l67mcef6jsna934qa2q6aw2sn2",
    ("M", "REG"): "https://imperialcollegelondon.box.com/shared/static/1qw28blyyda5q7lvnfxk86ycms2qaweh",
    ("R", "REG"): "https://imperialcollegelondon.box.com/shared/static/qbxklno9dch1peowcfyh8e31ix65eqcy",
    ("M", "CLA"): "https://imperialcollegelondon.box.com/shared/static/qywn066gwklxdanas97rz03by984xczh",
}


# ----------------------------------------------------------------------------- ResNetAE encoder

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1),
            nn.BatchNorm2d(in_channels),  # as in the original (in == out everywhere it is used)
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.residual_block(x)


class ResNetEncoder(nn.Module):
    def __init__(self, n_ResidualBlock=8, n_levels=4, input_ch=3, z_dim=128, bUseMultiResSkips=True):
        super().__init__()
        self.max_filters = 2 ** (n_levels + 3)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips
        self.conv_list = nn.ModuleList()
        self.res_blk_list = nn.ModuleList()
        self.multi_res_skip_list = nn.ModuleList()
        self.input_conv = nn.Sequential(nn.Conv2d(input_ch, 8, 3, 1, padding=1), nn.BatchNorm2d(8), nn.ReLU(inplace=True))
        for i in range(n_levels):
            n1, n2 = 2 ** (i + 3), 2 ** (i + 4)
            ks = 2 ** (n_levels - i)
            self.res_blk_list.append(nn.Sequential(*[ResidualBlock(n1, n1) for _ in range(n_ResidualBlock)]))
            self.conv_list.append(nn.Sequential(nn.Conv2d(n1, n2, 2, 2, padding=0), nn.BatchNorm2d(n2), nn.ReLU(inplace=True)))
            if bUseMultiResSkips:
                self.multi_res_skip_list.append(nn.Sequential(
                    nn.Conv2d(n1, self.max_filters, ks, ks, padding=0), nn.BatchNorm2d(self.max_filters),
                    nn.ReLU(inplace=True)))
        self.output_conv = nn.Sequential(nn.Conv2d(self.max_filters, z_dim, 3, 1, padding=1), nn.BatchNorm2d(z_dim),
                                         nn.ReLU(inplace=True))

    grad_ckpt = False  # recompute residual stacks in backward (memory saving; set by UVT)

    def forward(self, x):
        x = self.input_conv(x)
        skips = []
        for i in range(self.n_levels):
            if self.grad_ckpt and self.training and x.requires_grad:
                # note: BatchNorm running stats get updated twice per step under recomputation
                x = torch.utils.checkpoint.checkpoint(self.res_blk_list[i], x, use_reentrant=False)
            else:
                x = self.res_blk_list[i](x)
            if self.bUseMultiResSkips:
                skips.append(self.multi_res_skip_list[i](x))
            x = self.conv_list[i](x)
        if self.bUseMultiResSkips:
            x = sum([x] + skips)
        return self.output_conv(x)


class ResNetAE(nn.Module):
    """Encoder half of ResNetAE (the original sets decoder and fc2 to None, so they are not built)."""

    def __init__(self, input_shape=(128, 128, 3), n_ResidualBlock=8, n_levels=4, z_dim=128, bottleneck_dim=1024,
                 bUseMultiResSkips=True):
        super().__init__()
        assert input_shape[0] == input_shape[1]
        self.z_dim = z_dim
        self.img_latent_dim = input_shape[0] // (2 ** n_levels)
        self.encoder = ResNetEncoder(n_ResidualBlock, n_levels, input_shape[2], z_dim, bUseMultiResSkips)
        self.fc1 = nn.Linear(z_dim * self.img_latent_dim ** 2, bottleneck_dim)

    def encode(self, x):
        h = self.encoder(x)
        return torch.tanh(self.fc1(h.reshape(x.shape[0], -1)))


# ----------------------------------------------------------------------------- BERT encoder (HF-compatible names)

class BertSelfAttention(nn.Module):
    def __init__(self, hidden, heads, p_drop):
        super().__init__()
        assert hidden % heads == 0
        self.h, self.dh, self.p = heads, hidden // heads, p_drop
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)

    def forward(self, x, attn_mask=None):
        B, T, _ = x.shape
        q, k, v = (m(x).view(B, T, self.h, self.dh).transpose(1, 2) for m in (self.query, self.key, self.value))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.p if self.training else 0.0)
        return o.transpose(1, 2).reshape(B, T, self.h * self.dh)


class BertSelfOutput(nn.Module):
    def __init__(self, hidden, p_drop, eps):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=eps)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, h, x):
        return self.LayerNorm(self.dropout(self.dense(h)) + x)


class BertAttention(nn.Module):
    def __init__(self, hidden, heads, p_hidden, p_attn, eps):
        super().__init__()
        self.self = BertSelfAttention(hidden, heads, p_attn)  # attribute name "self" as in HF
        self.output = BertSelfOutput(hidden, p_hidden, eps)

    def forward(self, x, attn_mask=None):
        return self.output(self.self(x, attn_mask), x)


class BertIntermediate(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.dense = nn.Linear(hidden, inter)

    def forward(self, x):
        return F.gelu(self.dense(x))  # HF 'gelu' = exact erf GELU


class BertOutput(nn.Module):
    def __init__(self, hidden, inter, p_drop, eps):
        super().__init__()
        self.dense = nn.Linear(inter, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=eps)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, h, x):
        return self.LayerNorm(self.dropout(self.dense(h)) + x)


class BertLayer(nn.Module):
    def __init__(self, hidden, heads, inter, p_hidden=0.1, p_attn=0.1, eps=1e-12):
        super().__init__()
        self.attention = BertAttention(hidden, heads, p_hidden, p_attn, eps)
        self.intermediate = BertIntermediate(hidden, inter)
        self.output = BertOutput(hidden, inter, p_hidden, eps)

    def forward(self, x, attn_mask=None):
        a = self.attention(x, attn_mask)
        return self.output(self.intermediate(a), a)


class BertEncoder(nn.Module):
    """Stack of post-LN BERT layers; returns all hidden states [input, layer1, ..., layerL]."""

    def __init__(self, hidden, n_layers, heads, inter, p_hidden=0.1, p_attn=0.1, eps=1e-12, init_range=0.02):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(hidden, heads, inter, p_hidden, p_attn, eps) for _ in range(n_layers)])
        self.apply(lambda m: _bert_init(m, init_range))

    def forward(self, x, key_padding_mask=None):
        """key_padding_mask: optional (B, T) bool, True = real frame."""
        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(torch.bool)  # broadcast over heads and queries
        hs = [x]
        for l in self.layer:
            x = l(x, mask)
            hs.append(x)
        return hs


def _bert_init(m, std):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ----------------------------------------------------------------------------- UVT

class Reduce(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1).mean(dim=1)


class UVT(nn.Module):
    def __init__(self, emb_dim=1024, num_hidden_layers=16, intermediate_size=8192, attention_heads=16,
                 SDmode="reg", rm_branch=None, input_size=128, pos_emb=False, max_frames=1024,
                 hidden_dropout=0.1, attn_dropout=0.1, encode_chunk=256, grad_ckpt=False):
        super().__init__()
        if SDmode not in ("reg", "cla"):
            raise ValueError(SDmode)
        self.SDmode, self.rm_branch, self.em = SDmode, rm_branch, emb_dim
        self.encode_chunk = encode_chunk
        self.AE = ResNetAE(input_shape=(input_size, input_size, 3), n_ResidualBlock=8, n_levels=4,
                           bottleneck_dim=emb_dim)
        self.AE.encoder.grad_ckpt = grad_ckpt
        self.T = BertEncoder(emb_dim, num_hidden_layers, attention_heads, intermediate_size, hidden_dropout,
                             attn_dropout)
        self.pos = nn.Embedding(max_frames, emb_dim) if pos_emb else None
        if self.pos is not None:
            nn.init.normal_(self.pos.weight, 0.0, 0.02)
        last = 3 if SDmode == "cla" else 1
        if rm_branch != "SD":
            self.extremas = nn.Sequential(
                nn.Linear(emb_dim, emb_dim // 2), nn.LayerNorm(emb_dim // 2), nn.LeakyReLU(0.05, inplace=True),
                nn.Linear(emb_dim // 2, emb_dim // 4), nn.LayerNorm(emb_dim // 4), nn.LeakyReLU(0.05, inplace=True),
                nn.Linear(emb_dim // 4, last))
        if rm_branch != "EF":
            self.ejection = nn.Sequential(
                nn.Linear(emb_dim, emb_dim // 2), nn.LayerNorm(emb_dim // 2), nn.LeakyReLU(0.05, inplace=True),
                nn.Linear(emb_dim // 2, 1), Reduce(), nn.Sigmoid())

    def encode_frames(self, x):
        """(N, 3, H, W) -> (N, emb). Chunked so full-length videos fit in memory at inference."""
        if self.training or x.shape[0] <= self.encode_chunk:
            return self.AE.encode(x)
        return torch.cat([self.AE.encode(c) for c in x.split(self.encode_chunk)], 0)

    def forward(self, video, key_padding_mask=None):
        """video: float (B, F, 3, H, W) in [0, 1] (already zero-padded to input_size).
        Returns dict: sd (B, F) tanh signal ['reg'] or (B, F, 3) logits ['cla'] or None;
                      ef (B,) EF/100 in (0, 1) or None; Z (B, F, emb) averaged hidden states."""
        B, Fn = video.shape[:2]
        emb = self.encode_frames(video.flatten(0, 1)).view(B, Fn, self.em)
        if self.pos is not None:
            emb = emb + self.pos(torch.arange(Fn, device=emb.device))[None]
        z = torch.stack(self.T(emb, key_padding_mask)).mean(0)
        sd = ef = None
        if self.rm_branch != "SD":
            sd = self.extremas(z)
            sd = torch.tanh(sd).squeeze(-1) if self.SDmode == "reg" else sd
        if self.rm_branch != "EF":
            if key_padding_mask is None:
                ef = self.ejection(z)
            else:  # mean only over real frames (the original never pads, so this path is an extension)
                h = self.ejection[:4](z).squeeze(-1)
                w = key_padding_mask.to(h.dtype)
                ef = torch.sigmoid((h * w).sum(1) / w.sum(1).clamp_min(1))
        return {"sd": sd, "ef": ef, "Z": z}


def get_uvt(emb_dim=1024, num_hidden_layers=16, intermediate_size=8192, attention_heads=16, SDmode="reg", **kw):
    return UVT(emb_dim, num_hidden_layers, intermediate_size, attention_heads, SDmode, **kw)


def uvt_loss(out, sd_label, ef, SDmode="reg", attention=None, ef_weight=5.0):
    """Training loss of Network/train.py: SD loss + 5 * MSE(EF/100).
    reg: MSE on the per-frame signal (only frames with attention==1 if given, as in 'sample' mode).
    cla: cross-entropy with class weights [1, 5, 5] (other, ES, ED); frames outside attention ignored."""
    loss_sd = torch.zeros((), device=ef.device)
    if out["sd"] is not None:
        if SDmode == "reg":
            p, y = out["sd"].float(), sd_label.float()
            if attention is not None:
                m = attention.bool()
                p, y = p[m], y[m]
            loss_sd = F.mse_loss(p, y)
        else:
            y = sd_label.long().clone()
            if attention is not None:
                y[~attention.bool()] = -100
            w = torch.tensor([1.0, 5.0, 5.0], device=ef.device)
            loss_sd = F.cross_entropy(out["sd"].float().reshape(-1, 3), y.reshape(-1), weight=w, ignore_index=-100)
    loss_ef = torch.zeros((), device=ef.device)
    if out["ef"] is not None:
        loss_ef = F.mse_loss(out["ef"].float(), ef.float() / 100.0) * ef_weight
    return loss_sd + loss_ef, loss_sd.detach(), loss_ef.detach()


# ----------------------------------------------------------------------------- ED/ES decoding (Network/test.py)

def smooth(vec, window=5, rep=1):
    """Box filter applied `rep` times with zero padding (test.py `smooth`)."""
    vec = torch.as_tensor(vec, dtype=torch.float32).flatten()
    weight = torch.ones((1, 1, window)) / window
    for _ in range(rep):
        vec = F.conv1d(vec[None, None], weight, padding=(window - 1) // 2).flatten()
    return vec


def decode_sd_peaks(class_vec, mean_dist_coef=0.3):
    """ED/ES detection on a full video, 'reg' branch, as in test.py (SDmode='reg', full videos).

    Smooth the signal (window 5, 3 passes), take sign changes as phase boundaries, merge boundaries
    closer than mean_dist_coef * mean spacing, then within each segment take the extremum in the
    direction of the segment's mean sign. Returns list of (cls, index) with cls = +1 (ED) / -1 (ES),
    or None when fewer than 3 boundaries are found (the original's 'breaker' case)."""
    v = torch.as_tensor(class_vec, dtype=torch.float32).flatten()
    s = smooth(v, 5, 3)
    zc = ((s.sign().roll(1) - s.sign()) != 0).long()
    zc[0] = 0
    idx = torch.where(zc == 1)[0]
    if idx.numel() < 3:
        return None
    dist = idx[1:] - idx[:-1]
    mean_dist = dist.float().mean()
    while ((dist < mean_dist * mean_dist_coef) & (dist > 0)).any():
        b = int(torch.where(dist < mean_dist * mean_dist_coef)[0][0])
        new = int((idx[b] + idx[b + 1]) // 2)
        idx = torch.cat((idx[:b], torch.tensor([new]), idx[b + 2:]))
        dist = idx[1:] - idx[:-1]
    peaks = []
    for i in range(idx.numel() - 1):
        a, b = int(idx[i]), int(idx[i + 1])
        seg = v[a:b].numpy()
        cls = int(math.copysign(1, seg.mean())) if seg.mean() != 0 else 0
        peaks.append((cls, a + int((seg * cls).argmax())))
    return peaks


def peak_errors(peaks, es_frame, ed_frame):
    """Signed error of the nearest predicted ES (-1) / ED (+1) peak to the labelled frame (test.py).
    Returns (es_err, ed_err); NaN when no peak of that class exists (the original used 999)."""
    out = []
    for cls, tgt in ((-1, es_frame), (1, ed_frame)):
        cand = [i for c, i in (peaks or []) if c == cls]
        out.append(float(min((i - tgt for i in cand), key=abs)) if cand else float("nan"))
    return tuple(out)


# ----------------------------------------------------------------------------- released checkpoints

def load_released_checkpoint(model: UVT, path: str | Path, strict: bool = True):
    """Load a UVT repo best.pt (saved from nn.DataParallel: keys prefixed 'module.')."""
    # mmap: the released files are ~4 GB (weights + optimizer state); avoid reading them fully into RAM
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = ck.get("state_dict", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=strict)
    return res, {k: ck[k] for k in ("epoch", "best_loss", "loss") if k in ck}


def count_params(m):
    return sum(p.numel() for p in m.parameters())
