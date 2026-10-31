"""PanEcho frame encoder (Holste et al., JAMA 2025), frozen, as a per-frame 2D feature extractor.

PanEcho's image encoder is torchvision ConvNeXt-Tiny with the last linear layer removed (768-d output
after the classifier LayerNorm), trained on >1M Yale echocardiography videos for 39 reporting tasks;
EchoNet-Dynamic was not used for pretraining. Weights: CC BY-NC-SA 4.0 (non-commercial; attribution;
share-alike for adapted material) from https://github.com/CarDS-Yale/PanEcho/releases/v1.0.
We do not run the repository's hubconf (it unpickles a task table from a URL); the encoder is rebuilt
here and only its tensors are loaded with torch.load(weights_only=True).

Preprocessing follows PanEcho's EchoNet-Dynamic loader: each 112x112 frame is resized to 256x256
(cv2.INTER_AREA), centre-cropped to 224x224, scaled to [0, 1] and ImageNet-normalised.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models.convnext import CNBlock

from .models import FeatModel

WEIGHTS = os.path.join(os.environ.get("ECHO_DATA", r"E:\echo_project"), "third_party", "panecho_weights",
                       "panecho.pt")
PREFIX = "encoder.encoder.model."
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def panecho_image_encoder(weights: str = WEIGHTS) -> nn.Module:
    m = torchvision.models.convnext_tiny(weights=None)
    m.classifier[-1] = nn.Identity()
    sd = torch.load(weights, map_location="cpu", weights_only=True)["weights"]
    enc = {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX)}
    missing, unexpected = m.load_state_dict(enc, strict=True), None
    return m.eval()


class PanEchoEFModel(FeatModel):
    """PanEcho encoder with its last `n_tune` ConvNeXt blocks trainable, then FeatModel's temporal block,
    mean pool and head (same keys and outputs, so a frozen-feature run's checkpoint warm-starts it).
    n_tune=3 tunes stage 4 (with its downsampling layer) and the final LayerNorm; the frozen front always
    runs in eval mode without gradients. Input: StackClips uint8 (B, T, 3, 224, 224). The frozen weights
    are reloaded from the PanEcho release and are left out of state_dict(), so checkpoints hold only what
    training changes."""

    def __init__(self, temporal: str, d_model=256, n_layers=2, temporal_kw=None, n_tune=3, chunk=128,
                 weights: str = WEIGHTS):
        super().__init__(temporal, 768, d_model, n_layers, temporal_kw)
        enc = panecho_image_encoder(weights)
        # stem and downsampling layers are Sequentials too, so this lists every layer in order
        mods = [m for stage in enc.features for m in stage]
        blocks = [i for i, m in enumerate(mods) if isinstance(m, CNBlock)]
        assert 0 < n_tune <= len(blocks)
        cut = blocks[-n_tune]
        while cut > 0 and not isinstance(mods[cut - 1], CNBlock):
            cut -= 1                                    # the first tuned block takes its stage's downsampling
        self.frozen = nn.Sequential(*mods[:cut]).requires_grad_(False)
        self.tuned = nn.Sequential(*mods[cut:], enc.avgpool, enc.classifier)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)
        self.chunk = chunk
        self._register_state_dict_hook(_drop_frozen)
        self.register_load_state_dict_post_hook(_allow_missing_frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen.eval()
        return self

    def forward(self, video, valid=None):
        B, T = video.shape[:2]
        x = video.flatten(0, 1)
        with torch.no_grad():
            h = torch.cat([self.frozen(((xi.float() / 255.0 - self.mean) / self.std)
                                       .contiguous(memory_format=torch.channels_last))
                           for xi in x.split(self.chunk)])
        f = self.tuned(h).view(B, T, -1)
        return super().forward(f, valid)


def _drop_frozen(module, state_dict, prefix, local_metadata):
    for k in [k for k in state_dict if k.startswith(prefix + "frozen.")]:
        del state_dict[k]


def _allow_missing_frozen(module, incompatible):
    # keys are full names, so a wrapped model (e.g. AveragedModel) sees "module.frozen.*"
    incompatible.missing_keys[:] = [k for k in incompatible.missing_keys if ".frozen." not in "." + k]


def preprocess(frames_uint8: np.ndarray) -> torch.Tensor:
    """(T, 112, 112, 3) uint8 RGB -> (T, 3, 224, 224) float32, ImageNet-normalised (PanEcho EchoNet loader)."""
    import cv2
    out = np.empty((len(frames_uint8), 224, 224, 3), np.uint8)
    for i, f in enumerate(frames_uint8):
        out[i] = cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA)[16:240, 16:240]
    x = torch.from_numpy(out).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std
