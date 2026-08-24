"""Build the image-centred framework figure used in the WiML abstract."""

from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "abstract" / "figures" / "counterfactual_twins_examples.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu") / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def fit_image(
    path: Path,
    size: Tuple[int, int],
    crop: Optional[Tuple[int, int, int, int]] = None,
) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if crop:
        image = image.crop(crop)
    target_w, target_h = size
    scale = min(target_w / image.width, target_h / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    panel = Image.new("RGB", size, "#f7f7f7")
    panel.paste(resized, ((target_w - resized.width) // 2, (target_h - resized.height) // 2))
    return panel


def centred(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, text_font, fill: str) -> None:
    x, y = xy
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((x - (box[2] - box[0]) / 2, y), text, font=text_font, fill=fill)


def main() -> None:
    original_scene = ROOT / "assets" / "abstract_examples" / "disgust-224-original.jpg"
    original_face = ROOT / "assets" / "abstract_examples" / "sadness-210-original.jpg"
    examples = [
        (
            "FACIAL AU REGION",
            "neutralise brow evidence",
            original_face,
            ROOT / "runs" / "emotion6_full" / "twins" / "facial_action_region" / "sadness-210--facial_action_region--00.png",
            (245, 0, 520, 275),
            "#b54545",
        ),
        (
            "COLOUR + LIGHTING",
            "remove local chroma",
            original_scene,
            ROOT / "runs" / "emotion6_full" / "twins" / "color_lighting" / "disgust-224--color_lighting--00.png",
            None,
            "#c87922",
        ),
        (
            "SCENE CONTEXT",
            "attenuate background",
            original_scene,
            ROOT / "runs" / "emotion6_full" / "twins" / "scene_context" / "disgust-224--scene_context--00.png",
            None,
            "#3e7c59",
        ),
        (
            "EMBEDDED TEXT",
            "insert conflicting affect",
            original_scene,
            ROOT / "runs" / "emotion6_full" / "twins" / "embedded_text" / "disgust-224--embedded_text--00.png",
            None,
            "#3f68a8",
        ),
    ]

    width, height = 2000, 520
    margin, gap = 30, 18
    card_w = (width - 2 * margin - 3 * gap) // 4
    image_w, image_h = 202, 172
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(27, bold=True)
    label_font = font(22, bold=True)
    small_font = font(20)
    footer_font = font(23, bold=True)

    for index, (title, operation, before_path, after_path, crop, accent) in enumerate(examples):
        x0 = margin + index * (card_w + gap)
        x1 = x0 + card_w
        draw.rounded_rectangle((x0, 10, x1, 373), radius=12, fill="#fbfbfb", outline=accent, width=3)
        centred(draw, ((x0 + x1) // 2, 25), title, title_font, accent)

        left_x = x0 + 17
        right_x = x1 - 17 - image_w
        image_y = 98
        centred(draw, (left_x + image_w // 2, 65), "ORIGINAL", label_font, "#444444")
        centred(draw, (right_x + image_w // 2, 65), "TWIN", label_font, accent)

        before = fit_image(before_path, (image_w, image_h), crop)
        after = fit_image(after_path, (image_w, image_h), crop)
        canvas.paste(before, (left_x, image_y))
        canvas.paste(after, (right_x, image_y))
        draw.rectangle((left_x, image_y, left_x + image_w, image_y + image_h), outline="#777777", width=2)
        draw.rectangle((right_x, image_y, right_x + image_w, image_y + image_h), outline=accent, width=4)

        arrow_left = left_x + image_w + 7
        arrow_right = right_x - 7
        arrow_y = image_y + image_h // 2
        draw.line((arrow_left, arrow_y, arrow_right, arrow_y), fill=accent, width=5)
        draw.polygon(
            [(arrow_right, arrow_y), (arrow_right - 12, arrow_y - 9), (arrow_right - 12, arrow_y + 9)],
            fill=accent,
        )
        centred(draw, ((x0 + x1) // 2, 294), operation, small_font, "#333333")
        centred(draw, ((x0 + x1) // 2, 327), "one cue changed; other content preserved", font(16), "#666666")

    draw.rounded_rectangle((margin, 399, width - margin, 504), radius=12, fill="#f0f5f5", outline="#4a7777", width=2)
    centred(draw, (width // 2, 414), "PAIRED MODEL AUDIT", footer_font, "#285d5d")
    centred(
        draw,
        (width // 2, 455),
        "emotion + valence--arousal   |   cue responsiveness   |   selective invariance   |   evidence grounding   |   conflict calibration",
        small_font,
        "#222222",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, dpi=(300, 300), optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
