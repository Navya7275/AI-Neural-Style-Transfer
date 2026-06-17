"""VGG-19 feature extractor with loss layers injected at selected depths."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from nst.config import StyleTransferConfig
from nst.losses import ContentLoss, StyleLoss, TVLoss
from nst.model.normalization import Normalization

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_style_model(
    cnn: nn.Module,
    normalization_mean: torch.Tensor,
    normalization_std: torch.Tensor,
    style_img: torch.Tensor,
    content_img: torch.Tensor,
    device: torch.device,
    content_layers: Sequence[str] | None = None,
    style_layers: Sequence[str] | None = None,
    tv_weight: float = 2e-6,
) -> tuple[nn.Sequential, list[StyleLoss], list[ContentLoss], TVLoss]:
    """Build a truncated VGG-19 with content / style loss layers inserted.

    The network is cloned so the original weights are never modified.
    MaxPool layers are replaced with AvgPool for smoother gradients.
    """
    cfg = StyleTransferConfig()
    if content_layers is None:
        content_layers = cfg.content_layers
    if style_layers is None:
        style_layers = cfg.style_layers

    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses: list[ContentLoss] = []
    style_losses: list[StyleLoss] = []
    tv_loss = TVLoss(weight=tv_weight).to(device)

    model = nn.Sequential(normalization)
    block, conv_in_block = 1, 0

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            conv_in_block += 1
            name = f"conv{block}_{conv_in_block}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu{block}_{conv_in_block}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool{block}"
            layer = nn.AvgPool2d(kernel_size=2, stride=2)
            block += 1
            conv_in_block = 0
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn{block}_{conv_in_block}"
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{name}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{name}", style_loss)
            style_losses.append(style_loss)

    # Trim layers after the last loss node — no point computing deeper features
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], (ContentLoss, StyleLoss)):
            break
    model = model[: j + 1]

    return model, style_losses, content_losses, tv_loss
