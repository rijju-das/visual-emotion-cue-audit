"""Subject-preserving scene-context attenuation."""

from typing import List

import numpy as np
from PIL import Image, ImageFilter

from ..schema import CueFamily
from .base import GeneratedTwin, luminance_preserving_desaturate, mask_bbox


class ContextIntervention:
    cue_family = CueFamily.CONTEXT

    @staticmethod
    def context_mask(image: Image.Image) -> Image.Image:
        width, height = image.size
        small = image.convert("L").resize((160, 160), Image.Resampling.BILINEAR)
        edges = np.asarray(small.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
        yy, xx = np.mgrid[0:160, 0:160]
        centre = np.exp(-(((xx - 79.5) / 60.0) ** 2 + ((yy - 79.5) / 60.0) ** 2))
        foreground_score = 0.60 * edges + 0.40 * centre
        threshold = float(np.quantile(foreground_score, 0.58))
        foreground = (foreground_score >= threshold).astype(np.uint8) * 255
        foreground_image = Image.fromarray(foreground, mode="L").filter(ImageFilter.GaussianBlur(7))
        foreground_image = foreground_image.resize((width, height), Image.Resampling.BILINEAR)
        context = 255 - np.asarray(foreground_image, dtype=np.uint8)
        return Image.fromarray(context, mode="L")

    def generate(self, image: Image.Image, foreground_mask: Image.Image = None) -> List[GeneratedTwin]:
        if foreground_mask is not None and foreground_mask.getbbox() is not None:
            foreground = np.asarray(foreground_mask.convert("L"), dtype=np.uint8)
            mask = Image.fromarray(255 - foreground, mode="L")
            estimator = "mask2former_panoptic_thing_union"
        else:
            mask = self.context_mask(image)
            estimator = "edge_plus_centre_prior_fallback"
        desaturated = luminance_preserving_desaturate(image, mask)
        blur_radius = max(5.0, min(image.size) / 50.0)
        blurred_background = desaturated.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        blurred = Image.composite(blurred_background, image.convert("RGB"), mask.convert("L"))
        return [
            GeneratedTwin(
                image=blurred,
                mask=mask,
                cue_family=self.cue_family,
                operation="background_chroma_and_detail_attenuation",
                target_region=mask_bbox(mask),
                metadata={
                    "foreground_estimator": estimator,
                    "foreground_preserved": foreground_mask is not None,
                    "context_operation": "desaturate_and_blur_background_only",
                },
            )
        ]
