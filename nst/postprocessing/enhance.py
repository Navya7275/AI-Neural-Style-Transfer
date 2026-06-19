"""Post-processing: edge cropping, sharpening, and contrast adjustment."""

from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import torch


def post_process(
    output_tensor: torch.Tensor,
    *,
    sharpen: bool = True,
    contrast_boost: float = 1.05,
    edge_crop: int = 4,
) -> Image.Image:
    """Convert an optimised tensor to a polished PIL image.

    Applies optional edge cropping (to remove pooling-boundary artifacts),
    contrast enhancement, and unsharp-mask sharpening.
    """
    image = output_tensor.squeeze(0).cpu().detach().clamp(0, 1)

    if edge_crop > 0:
        image = image[:, edge_crop:-edge_crop, edge_crop:-edge_crop]

    pil_img = transforms.ToPILImage()(image)

    if contrast_boost != 1.0:
        pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast_boost)

    if sharpen:
        pil_img = pil_img.filter(ImageFilter.SHARPEN)

    return pil_img


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a (1, 3, H, W) tensor to a PIL Image without enhancement."""
    image = tensor.squeeze(0).cpu().detach().clamp(0, 1)
    return transforms.ToPILImage()(image)
