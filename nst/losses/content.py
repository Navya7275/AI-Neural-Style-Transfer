"""Content loss — feature-space MSE between generated and content images."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    """Compute MSE between activations of the generated image and the content target."""

    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        self.target = target.detach()
        self.loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.loss = F.mse_loss(x, self.target)
        return x
