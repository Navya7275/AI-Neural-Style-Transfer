"""Hyperparameters, layer configurations, and device utilities."""

from dataclasses import dataclass, field

import torch


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class StyleTransferConfig:
    """All tuneable knobs for the Gatys optimisation loop."""

    content_layers: list[str] = field(
        default_factory=lambda: ["relu4_2"],
    )
    style_layers: list[str] = field(
        default_factory=lambda: [
            "relu1_1",
            "relu2_1",
            "relu3_1",
            "relu4_1",
            "relu5_1",
        ],
    )

    num_steps: int = 400
    style_weight: float = 1e6
    content_weight: float = 1e1
    tv_weight: float = 2e-6
    max_size: int = 512


def get_device() -> torch.device:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
