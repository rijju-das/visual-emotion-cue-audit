"""Luminance-preserving local colour interventions."""

from typing import List

from PIL import Image, ImageDraw

from ..schema import CueFamily
from .base import GeneratedTwin, luminance_preserving_desaturate


class ColorIntervention:
    cue_family = CueFamily.COLOR

    def __init__(self, grid_size: int = 4):
        self.grid_size = grid_size

    def generate(self, image: Image.Image) -> List[GeneratedTwin]:
        width, height = image.size
        twins = []
        for index in range(self.grid_size * self.grid_size):
            row, column = divmod(index, self.grid_size)
            x0 = round(column * width / self.grid_size)
            x1 = round((column + 1) * width / self.grid_size)
            y0 = round(row * height / self.grid_size)
            y1 = round((row + 1) * height / self.grid_size)
            mask = Image.new("L", image.size, 0)
            ImageDraw.Draw(mask).rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
            twins.append(
                GeneratedTwin(
                    image=luminance_preserving_desaturate(image, mask),
                    mask=mask,
                    cue_family=self.cue_family,
                    operation="local_chroma_removal",
                    target_region=(x0, y0, x1, y1),
                    metadata={"grid_cell": index, "grid_size": self.grid_size},
                )
            )
        return twins

