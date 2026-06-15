"""Style loss — Gram-matrix MSE between generated and style images."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """Compute the Gram matrix for a batch of feature maps.

    The Gram matrix captures feature correlations and serves as
    a texture descriptor independent of spatial arrangement.
    """
    b, c, h, w = x.size()
    features = x.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)


class StyleLoss(nn.Module):
    """Compute MSE between Gram matrices of the generated image and the style target."""

    def __init__(self, target_feature: torch.Tensor) -> None:
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        G = gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x
