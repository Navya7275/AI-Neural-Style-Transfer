"""Total Variation loss — encourages spatial smoothness."""

import torch
import torch.nn as nn


class TVLoss(nn.Module):
    """Penalise high-frequency noise by measuring neighbouring-pixel differences."""

    def __init__(self, weight: float = 2e-6) -> None:
        super().__init__()
        self.weight = weight
        self.loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        self.loss = self.weight * (diff_h.abs().mean() + diff_w.abs().mean())
        return x
