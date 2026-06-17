"""ImageNet normalisation layer inserted at the head of VGG-19."""

import torch
import torch.nn as nn


class Normalization(nn.Module):
    """Shift and scale input images to match ImageNet statistics."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.mean = mean.clone().view(-1, 1, 1)
        self.std = std.clone().view(-1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
