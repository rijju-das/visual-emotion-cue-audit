"""Base types and image operations for interventions."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from ..schema import CueFamily


@dataclass
class GeneratedTwin:
    image: Image.Image
    mask: Image.Image
    cue_family: CueFamily
    operation: str
    target_region: Optional[Tuple[int, int, int, int]] = None
    expected_direction: str = "attenuate_source_affect"
    metadata: Dict[str, Any] = field(default_factory=dict)


def luminance_preserving_desaturate(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    gray = np.repeat(luminance[..., None], 3, axis=2)
    alpha = np.asarray(mask.convert("L"), dtype=np.float32)[..., None] / 255.0
    output = rgb * (1.0 - alpha) + gray * alpha
    return Image.fromarray(np.clip(output * 255.0, 0, 255).astype(np.uint8))


def attenuate_illumination(image: Image.Image, mask: Image.Image, exposure: float = 0.65) -> Image.Image:
    """Reduce exposure inside a mask in linear RGB without blurring scene content."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    shifted = np.clip(linear * exposure, 0.0, 1.0)
    shifted = np.where(shifted <= 0.0031308, shifted * 12.92, 1.055 * shifted ** (1.0 / 2.4) - 0.055)
    alpha = np.asarray(mask.convert("L"), dtype=np.float32)[..., None] / 255.0
    output = rgb * (1.0 - alpha) + shifted * alpha
    return Image.fromarray(np.clip(output * 255.0, 0, 255).astype(np.uint8))


def blurred_region(image: Image.Image, mask: Image.Image, radius: float = 12.0) -> Image.Image:
    blurred = image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))
    feathered = mask.convert("L").filter(ImageFilter.GaussianBlur(radius=max(1.0, radius / 3.0)))
    return Image.composite(blurred, image.convert("RGB"), feathered)


def mask_bbox(mask: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    return mask.convert("L").getbbox()
