"""OCR-localised embedded-text removal and affect-conflict insertion."""

import csv
import io
import subprocess
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

from ..schema import CueFamily
from .base import GeneratedTwin, blurred_region


CONFLICT_WORD = {
    "anger": "JOY",
    "disgust": "DELIGHT",
    "fear": "SAFE",
    "joy": "SADNESS",
    "sadness": "JOY",
    "surprise": "CALM",
}


class TextIntervention:
    cue_family = CueFamily.TEXT

    def __init__(self, executable: str, add_conflict: bool = True):
        self.executable = Path(executable)
        self.add_conflict = add_conflict
        self.unavailable_reason = ""

    def _boxes(self, image_path: Path) -> List[Tuple[int, int, int, int, str, float]]:
        if not self.executable.exists():
            self.unavailable_reason = "tesseract_missing"
            return []
        completed = subprocess.run(
            [str(self.executable), str(image_path), "stdout", "tsv"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if completed.returncode != 0:
            self.unavailable_reason = "tesseract_failed"
            return []
        boxes = []
        for row in csv.DictReader(io.StringIO(completed.stdout), delimiter="\t"):
            text = (row.get("text") or "").strip()
            try:
                confidence = float(row.get("conf", -1))
            except ValueError:
                confidence = -1
            if text and confidence >= 40:
                left, top = int(row["left"]), int(row["top"])
                width, height = int(row["width"]), int(row["height"])
                boxes.append((left, top, left + width, top + height, text, confidence))
        return boxes

    @staticmethod
    def _font(size: int):
        for path in [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]:
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
        return ImageFont.load_default()

    def generate(self, image: Image.Image, image_path: Path, emotion: str) -> List[GeneratedTwin]:
        boxes = self._boxes(image_path)
        twins = []
        for box_index, (left, top, right, bottom, token, confidence) in enumerate(boxes):
            mask = Image.new("L", image.size, 0)
            ImageDraw.Draw(mask).rectangle((left, top, right, bottom), fill=255)
            twins.append(
                GeneratedTwin(
                    image=blurred_region(image, mask, radius=max(5.0, min(image.size) / 55.0)),
                    mask=mask,
                    cue_family=self.cue_family,
                    operation="remove_detected_text",
                    target_region=mask.getbbox(),
                    metadata={
                        "ocr_tokens": [token],
                        "ocr_confidences": [confidence],
                        "ocr_box_index": box_index,
                        "text_intervention_scope": "single_reported_ocr_token_candidate",
                    },
                )
            )
        if self.add_conflict:
            conflict = image.convert("RGB").copy()
            overlay = ImageDraw.Draw(conflict)
            word = CONFLICT_WORD.get(emotion, "CALM")
            size = max(18, min(image.size) // 10)
            font = self._font(size)
            bounds = overlay.textbbox((0, 0), word, font=font, stroke_width=1)
            text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
            x = max(4, (image.width - text_width) // 2)
            y = max(4, image.height - text_height - size // 2)
            box = (x - 6, y - 4, x + text_width + 6, y + text_height + 4)
            overlay.rounded_rectangle(box, radius=5, fill=(245, 245, 245), outline=(20, 20, 20), width=2)
            overlay.text((x, y), word, font=font, fill=(15, 15, 15), stroke_width=1, stroke_fill="white")
            conflict_mask = Image.new("L", image.size, 0)
            ImageDraw.Draw(conflict_mask).rectangle(box, fill=255)
            twins.append(
                GeneratedTwin(
                    image=conflict,
                    mask=conflict_mask,
                    cue_family=self.cue_family,
                    operation="insert_affect_conflict_text",
                    target_region=box,
                    expected_direction="increase_uncertainty_or_follow_text",
                    metadata={"inserted_text": word, "source_emotion": emotion, "ocr_text_present": bool(boxes)},
                )
            )
        return twins
