"""Image loading with aspect-ratio preservation and edge-artifact prevention."""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


def image_loader(
    image_path: str,
    max_size: int = 512,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Load an image as a normalised (0-1) tensor, longest edge ≤ *max_size*.

    Dimensions are rounded to multiples of 8 for clean pooling.
    A reflect-pad-then-crop pass prevents dark border artifacts.
    """
    if device is None:
        device = torch.device("cpu")

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    scale = max_size / max(w, h)
    new_w = (int(w * scale) // 8) * 8
    new_h = (int(h * scale) // 8) * 8

    loader = transforms.Compose([
        transforms.Resize((new_h, new_w)),
        transforms.ToTensor(),
    ])
    tensor = loader(image).unsqueeze(0).to(device, torch.float)

    pad = 8
    tensor = nn.functional.pad(tensor, [pad, pad, pad, pad], mode="reflect")
    tensor = tensor[:, :, pad:-pad, pad:-pad]

    return tensor
