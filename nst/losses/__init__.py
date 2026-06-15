"""Loss functions for Gatys neural style transfer."""

from nst.losses.content import ContentLoss
from nst.losses.style import StyleLoss, gram_matrix
from nst.losses.regularization import TVLoss

__all__ = ["ContentLoss", "StyleLoss", "TVLoss", "gram_matrix"]
