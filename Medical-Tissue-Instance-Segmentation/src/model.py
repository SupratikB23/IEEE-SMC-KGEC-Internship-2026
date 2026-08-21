"""The model: a frozen foundation-model backbone, small trainable adapters, and
a decoder that turns features back into object outlines.

The backbone is never modified. Only the adapters (under 3% of parameters) and
the decoder are trained, which is what keeps the whole thing inside a 16 GB GPU.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


# --------------------------------------------------------------------------- #
# Adapters                                                                     #
# --------------------------------------------------------------------------- #
class DoRALinear(nn.Module):
    """Weight-decomposed low-rank adaptation of a frozen linear layer.

        W = m * V / ||V||,   V = W0 + (alpha / r) * B @ A

    The magnitude ``m`` and the direction ``V / ||V||`` are learned separately,
    which is what distinguishes this from plain low-rank adaptation. ``B`` is
    zero-initialised, so at the start W == W0 exactly and training departs
    smoothly from the pretrained weights rather than jumping away from them.

    Because W is an ordinary weight matrix, the adapter folds back into W0 after
    training and costs nothing at inference.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.scale = alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        # one magnitude per output unit, initialised to the pretrained norm
        self.m = nn.Parameter(base.weight.detach().norm(dim=1, keepdim=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        V = self.base.weight + self.scale * (self.B @ self.A)
        V = V / (V.norm(dim=1, keepdim=True) + 1e-6)
        return F.linear(x, self.m * V, self.base.bias)


def _is_attention_linear(name: str) -> bool:
    """The query, key, value and output projections of an attention block.

    The MLP layers are excluded: their names carry no '.attention.' component.
    """
    leaf = name.rsplit(".", 1)[-1]
    return ".attention." in name and leaf in ("query", "key", "value", "dense")


def apply_adapters(model: nn.Module, r: int, alpha: float) -> int:
    """Freeze the backbone, then wrap every attention projection. Returns the
    number of layers adapted (96 for a 24-layer ViT: 4 per block)."""
    for p in model.parameters():
        p.requires_grad_(False)

    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and _is_attention_linear(n)]
    for name, module in targets:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[-1], DoRALinear(module, r, alpha))
    return len(targets)


def adapter_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# --------------------------------------------------------------------------- #
# Backbone                                                                     #
# --------------------------------------------------------------------------- #
class FrozenBackbone(nn.Module):
    """Phikon-v2 (ViT-L/16), frozen, with adapters injected.

    Hidden states are tapped at four depths so the decoder receives features at
    several levels of abstraction rather than only the last one.
    """

    def __init__(self, name: str | None = None, adapt: bool = True) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(name or cfg.backbone)
        self.n_adapted = apply_adapters(self.encoder, cfg.adapter_rank,
                                        cfg.adapter_alpha) if adapt else 0
        self.tap_layers = cfg.tap_layers

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """x: B x 3 x 224 x 224  ->  list of B x 1024 x 14 x 14 feature maps."""
        out = self.encoder(pixel_values=x, output_hidden_states=True)
        maps = []
        for layer in self.tap_layers:
            h = out.hidden_states[layer][:, 1:, :]        # drop the CLS token
            b, n, d = h.shape
            g = int(round(n ** 0.5))
            maps.append(h.transpose(1, 2).reshape(b, d, g, g))
        return maps


# --------------------------------------------------------------------------- #
# Decoder                                                                      #
# --------------------------------------------------------------------------- #
def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNetDecoder(nn.Module):
    """Progressive upsampling, 14 -> 28 -> 56 -> 112 -> 224.

    A convolutional stem runs over the raw image and feeds skip connections at
    each resolution. Those skips are what restore the fine detail needed to tell
    two touching objects apart — a transformer's 14x14 grid alone cannot.
    """

    def __init__(self, in_dim: int = 1024, n_taps: int = 4, width: int = 64) -> None:
        super().__init__()
        self.fuse = nn.Conv2d(in_dim * n_taps, width * 8, 1)

        # stem over the raw image, one level per upsampling stage
        self.stem1 = _conv_block(3, width)              # 224
        self.stem2 = _conv_block(width, width)          # 112
        self.stem3 = _conv_block(width, width)          # 56
        self.stem4 = _conv_block(width, width)          # 28

        self.up4 = _conv_block(width * 8 + width, width * 4)   # 28
        self.up3 = _conv_block(width * 4 + width, width * 2)   # 56
        self.up2 = _conv_block(width * 2 + width, width)       # 112
        self.up1 = _conv_block(width + width, width)           # 224

    def forward(self, taps: List[torch.Tensor], image: torch.Tensor) -> torch.Tensor:
        s1 = self.stem1(image)                                  # 224
        s2 = self.stem2(F.avg_pool2d(s1, 2))                    # 112
        s3 = self.stem3(F.avg_pool2d(s2, 2))                    # 56
        s4 = self.stem4(F.avg_pool2d(s3, 2))                    # 28

        x = self.fuse(torch.cat(taps, dim=1))                   # 14
        x = F.interpolate(x, size=s4.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up4(torch.cat([x, s4], 1))
        x = F.interpolate(x, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, s3], 1))
        x = F.interpolate(x, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, s2], 1))
        x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        return self.up1(torch.cat([x, s1], 1))


class SegFormerDecoder(nn.Module):
    """All-MLP head with a single bilinear upsample — the low-capacity read-out,
    kept so the effect of decoder capacity can be measured."""

    def __init__(self, in_dim: int = 1024, n_taps: int = 4, width: int = 64) -> None:
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(in_dim, width, 1) for _ in range(n_taps)])
        self.fuse = nn.Sequential(nn.Conv2d(width * n_taps, width, 1),
                                  nn.BatchNorm2d(width), nn.ReLU(inplace=True))

    def forward(self, taps: List[torch.Tensor], image: torch.Tensor) -> torch.Tensor:
        x = self.fuse(torch.cat([p(t) for p, t in zip(self.proj, taps)], dim=1))
        return F.interpolate(x, size=image.shape[-2:], mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------- #
# Full segmentation model                                                      #
# --------------------------------------------------------------------------- #
class SegModel(nn.Module):
    """Frozen backbone + adapters + decoder + prediction heads.

    Always predicts a semantic map and a boundary map. On nuclei it additionally
    predicts horizontal and vertical distance maps, because contours alone are
    too thin to separate objects at several hundred per image.
    """

    def __init__(self, kind: str = "gland", decoder: str | None = None,
                 use_hv: bool | None = None, backbone: FrozenBackbone | None = None) -> None:
        super().__init__()
        self.kind = kind
        self.use_hv = (kind == "nuclei") if use_hv is None else use_hv

        self.backbone = backbone or FrozenBackbone()
        dec = (decoder or cfg.decoder).lower()
        Decoder = SegFormerDecoder if dec == "segformer" else UNetDecoder
        self.decoder = Decoder(cfg.embed_dim, len(cfg.tap_layers))

        width = 64
        self.sem_head = nn.Conv2d(width, 1, 1)
        self.bnd_head = nn.Conv2d(width, 1, 1)
        self.hv_head = nn.Conv2d(width, 2, 1) if self.use_hv else None

    def forward(self, image: torch.Tensor):
        """Returns (semantic logits, boundary logits, hv maps or None)."""
        feats = self.decoder(self.backbone(image), image)
        sem = self.sem_head(feats)
        bnd = self.bnd_head(feats)
        hv = torch.tanh(self.hv_head(feats)) if self.hv_head is not None else None
        return sem, bnd, hv

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# --------------------------------------------------------------------------- #
# Stage-1 predictor                                                            #
# --------------------------------------------------------------------------- #
class Predictor(nn.Module):
    """Small transformer used only during self-supervised adaptation.

    Two heads: one predicts the frozen teacher's token embedding (the latent
    anchor), the other predicts the hematoxylin boundary target. Both are
    discarded once Stage 1 finishes — only the adapter is kept.
    """

    def __init__(self, enc_dim: int = 1024, dim: int = 384, depth: int = 6,
                 heads: int = 6, n_tokens: int = 196) -> None:
        super().__init__()
        self.embed = nn.Linear(enc_dim, dim)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True,
                                           norm_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)

        self.emb_head = nn.Linear(dim, enc_dim)   # latent anchor
        self.bnd_head = nn.Linear(dim, 1)         # hematoxylin boundary

    def forward(self, tokens: torch.Tensor):
        x = self.norm(self.blocks(self.embed(tokens) + self.pos))
        return self.emb_head(x), self.bnd_head(x).squeeze(-1)


def build_model(kind: str = "gland", **kw) -> SegModel:
    return SegModel(kind=kind, **kw)
