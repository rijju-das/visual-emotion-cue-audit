"""Dataset adapters and deterministic sampling."""

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from .schema import AffectSample, EMOTIONS


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _normalise_va(value: str) -> Optional[float]:
    if value is None or value.strip() == "":
        return None
    raw = float(value)
    return max(-1.0, min(1.0, (raw - 5.0) / 4.0))


def _ground_truth(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    rows = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            clean = {key.strip("[]"): value for key, value in row.items()}
            rows[Path(clean["image_filename"]).name] = clean
    return rows


def load_emotion6(image_root: Path, ground_truth_csv: Path) -> List[AffectSample]:
    annotations = _ground_truth(ground_truth_csv)
    samples = []
    for emotion in EMOTIONS:
        folder = image_root / emotion
        if not folder.is_dir():
            raise FileNotFoundError("Missing Emotion6 class directory: {}".format(folder))
        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            row = annotations.get(image_path.name, {})
            distribution = {}
            for label in EMOTIONS:
                value = row.get("prob. {}".format(label), "")
                if value not in (None, ""):
                    distribution[label] = float(value)
            samples.append(
                AffectSample(
                    sample_id="{}-{}".format(emotion, image_path.stem),
                    image_path=str(image_path.resolve()),
                    emotion=emotion,
                    emotion_distribution=distribution,
                    valence=_normalise_va(row.get("valence")),
                    arousal=_normalise_va(row.get("arousal")),
                )
            )
    if not samples:
        raise RuntimeError("No Emotion6 images found at {}".format(image_root))
    return samples


def deterministic_split(samples: List[AffectSample], seed: int = 42) -> List[AffectSample]:
    rng = random.Random(seed)
    by_class = {label: [] for label in EMOTIONS}
    for sample in samples:
        by_class[sample.emotion].append(sample)
    result = []
    for label in EMOTIONS:
        group = list(by_class[label])
        rng.shuffle(group)
        train_end = int(0.70 * len(group))
        val_end = train_end + int(0.10 * len(group))
        for index, sample in enumerate(group):
            sample.split = "train" if index < train_end else "validation" if index < val_end else "test"
            result.append(sample)
    return result


def balanced_subset(samples: List[AffectSample], split: str, per_class: int, seed: int) -> List[AffectSample]:
    rng = random.Random(seed)
    selected = []
    for label in EMOTIONS:
        group = [sample for sample in samples if sample.split == split and sample.emotion == label]
        rng.shuffle(group)
        selected.extend(group[:per_class])
    rng.shuffle(selected)
    return selected


def load_sample_manifest(path: Path) -> List[AffectSample]:
    """Load a project-local JSONL sample manifest."""
    if not path.is_file():
        raise FileNotFoundError("Missing auxiliary sample manifest: {}".format(path))
    samples = []
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            payload = json.loads(line)
            image_path = Path(payload["image_path"])
            if not image_path.is_absolute():
                payload["image_path"] = str((path.parent / image_path).resolve())
            samples.append(AffectSample(**payload))
    return samples
