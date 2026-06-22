"""High-level stylisation pipeline — bridges the UI sliders with the engine."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

import gradio as gr

from nst.config import StyleTransferConfig, IMAGENET_MEAN, IMAGENET_STD, get_device
from nst.preprocessing import image_loader, match_histogram
from nst.postprocessing import post_process
from nst.engine import run_style_transfer
from PIL import Image

# ── Module-level singletons (loaded once, reused across calls) ──────────

_device = get_device()

_cnn = (
    models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    .features.to(_device)
    .eval()
)

_norm_mean = torch.tensor(IMAGENET_MEAN).to(_device)
_norm_std = torch.tensor(IMAGENET_STD).to(_device)


def stylize(
    content_path: str | None,
    style_path: str | None,
    style_strength: float,
    content_preservation: float,
    smoothing: float,
    steps: float,
    color_match: bool,
    sharpening: bool,
    contrast: float,
    resolution: float,
    progress: gr.Progress = gr.Progress(),
) -> Image.Image:
    """Run the full style-transfer pipeline and return a PIL image.

    Slider values are converted to optimiser weights via exponential
    scaling so that the UI feels linear to the user.
    """
    if content_path is None or style_path is None:
        raise gr.Error("Please upload both images to begin.")

    progress(0, desc="Loading images...")
    content_img = image_loader(content_path, max_size=int(resolution), device=_device)
    style_img = image_loader(style_path, max_size=int(resolution), device=_device)

    _, _, h, w = content_img.shape
    style_img = nn.functional.interpolate(
        style_img, size=(h, w), mode="bilinear", align_corners=False,
    )

    progress(0.08, desc="Preparing canvas...")

    style_weight = 5e5 * (10 ** (style_strength / 10.0))
    content_weight = 0.5 * (10 ** (content_preservation / 10.0))
    tv_weight = smoothing * 1e-5

    if color_match:
        input_img = match_histogram(content_img, style_img).to(_device)
    else:
        input_img = content_img.clone()

    progress(0.12, desc="Transferring style...")

    output = run_style_transfer(
        _cnn, _norm_mean, _norm_std,
        content_img, style_img, input_img, _device,
        num_steps=int(steps),
        style_weight=style_weight,
        content_weight=content_weight,
        tv_weight=tv_weight,
        progress_callback=progress,
    )

    progress(0.95, desc="Finishing touches...")
    return post_process(output, sharpen=sharpening, contrast_boost=contrast)
