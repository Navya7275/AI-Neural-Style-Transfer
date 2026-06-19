"""Histogram-based colour transfer between content and style images."""

import torch
import torch.nn as nn


def match_histogram(
    content_tensor: torch.Tensor,
    style_tensor: torch.Tensor,
) -> torch.Tensor:
    """Re-map the per-channel intensity distribution of *content* to match *style*.

    This reduces colour clashing when the style palette differs
    dramatically from the content photograph.
    """
    c_t = content_tensor.squeeze(0)
    s_t = style_tensor.squeeze(0)
    result = torch.zeros_like(c_t)

    for ch in range(3):
        c_ch = c_t[ch].flatten()
        s_ch = s_t[ch].flatten()

        c_sorted, c_indices = c_ch.sort()
        s_sorted, _ = s_ch.sort()

        s_interp = nn.functional.interpolate(
            s_sorted.unsqueeze(0).unsqueeze(0),
            size=c_sorted.shape[0],
            mode="linear",
            align_corners=False,
        ).squeeze()

        output = torch.zeros_like(c_ch)
        output[c_indices] = s_interp
        result[ch] = output.view(c_t[ch].shape)

    return result.unsqueeze(0).clamp(0, 1)
