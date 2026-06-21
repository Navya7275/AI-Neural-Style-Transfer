"""L-BFGS optimisation loop for Gatys-style neural style transfer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.optim as optim

from nst.model import build_style_model

if TYPE_CHECKING:
    from collections.abc import Callable


def run_style_transfer(
    cnn: torch.nn.Module,
    normalization_mean: torch.Tensor,
    normalization_std: torch.Tensor,
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    input_img: torch.Tensor,
    device: torch.device,
    *,
    num_steps: int = 400,
    style_weight: float = 1e6,
    content_weight: float = 1e1,
    tv_weight: float = 2e-6,
    progress_callback: Callable[[float, str], None] | None = None,
) -> torch.Tensor:
    """Run the iterative optimisation that blends content and style.

    Parameters
    ----------
    cnn : pretrained VGG-19 features module
    input_img : the canvas tensor (initialised from content image)
    num_steps : L-BFGS iterations
    style_weight / content_weight : relative loss weighting
    tv_weight : total-variation regularisation strength
    progress_callback : optional ``(fraction, description)`` reporter
    """

    model, style_losses, content_losses, tv_loss = build_style_model(
        cnn, normalization_mean, normalization_std,
        style_img, content_img, device,
        tv_weight=tv_weight,
    )

    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_img])
    step = [0]

    while step[0] <= num_steps:

        def closure() -> torch.Tensor:
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)

            style_score = sum(sl.loss for sl in style_losses) / len(style_losses)
            content_score = sum(cl.loss for cl in content_losses) / len(content_losses)
            tv_loss(input_img)

            loss = (
                style_weight * style_score
                + content_weight * content_score
                + tv_loss.loss
            )
            loss.backward()
            step[0] += 1

            if progress_callback and step[0] % 25 == 0:
                pct = min(0.15 + 0.75 * (step[0] / num_steps), 0.92)
                progress_callback(pct, desc=f"Step {step[0]}/{num_steps}")

            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img
