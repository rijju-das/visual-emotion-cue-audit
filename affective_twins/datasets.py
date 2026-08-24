"""Dataset adapters and deterministic sampling."""

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from .schema import AffectSample, EMOTION6_ANNOTATION_LABELS, EMOTIONS


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
            key = clean["image_filename"].strip().replace("\\", "/").lstrip("./")
            if key.startswith("images/"):
                key = key[len("images/"):]
            if key in rows:
                raise ValueError("Duplicate Emotion6 annotation key: {}".format(key))
            rows[key] = clean
    return rows


def load_emotion6(image_root: Path, ground_truth_csv: Path) -> List[AffectSample]:
    annotations = _ground_truth(ground_truth_csv)
    samples = []
    missing_annotations = []
    for nominal_emotion in EMOTIONS:
        folder = image_root / nominal_emotion
        if not folder.is_dir():
            raise FileNotFoundError("Missing Emotion6 class directory: {}".format(folder))
        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            annotation_key = "{}/{}".format(nominal_emotion, image_path.name)
            row = annotations.get(annotation_key, {})
            if not row:
                missing_annotations.append(annotation_key)
                continue
            distribution = {}
            for label in EMOTION6_ANNOTATION_LABELS:
                value = row.get("prob. {}".format(label), "")
                if value not in (None, ""):
                    distribution[label] = float(value)
            if not distribution:
                raise ValueError("Emotion6 annotation has no emotion distribution: {}".format(annotation_key))
            ranked = sorted(
                EMOTION6_ANNOTATION_LABELS,
                key=lambda label: (-distribution.get(label, 0.0), EMOTION6_ANNOTATION_LABELS.index(label)),
            )
            human_emotion = ranked[0]
            human_probability = distribution.get(human_emotion, 0.0)
            human_margin = human_probability - distribution.get(ranked[1], 0.0)
            samples.append(
                AffectSample(
                    sample_id="{}-{}".format(nominal_emotion, image_path.stem),
                    image_path=str(image_path.resolve()),
                    emotion=human_emotion,
                    emotion_distribution=distribution,
                    valence=_normalise_va(row.get("valence")),
                    arousal=_normalise_va(row.get("arousal")),
                    nominal_emotion=nominal_emotion,
                    human_plurality_emotion=human_emotion,
                    human_plurality_probability=human_probability,
                    metadata={
                        "annotation_key": annotation_key,
                        "annotation_label_count": len(EMOTION6_ANNOTATION_LABELS),
                        "folder_human_agreement": nominal_emotion == human_emotion,
                        "human_plurality_margin": human_margin,
                        "human_plurality_tied": human_margin <= 1e-12,
                        "label_source": "human_plurality",
                        "source_dataset": "Emotion6",
                    },
                )
            )
    if missing_annotations:
        preview = ", ".join(missing_annotations[:10])
        raise ValueError(
            "Missing Emotion6 annotations for {} image(s): {}".format(len(missing_annotations), preview)
        )
    if not samples:
        raise RuntimeError("No Emotion6 images found at {}".format(image_root))
    return samples


def deterministic_split(samples: List[AffectSample], seed: int = 42) -> List[AffectSample]:
    rng = random.Random(seed)
    by_class = {label: [] for label in EMOTIONS}
    for sample in samples:
        if sample.emotion in by_class:
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


def high_confidence_human_subset(samples: List[AffectSample], per_class: int) -> List[AffectSample]:
    """Select the strongest human-plurality examples for every modeled emotion."""
    selected = []
    for label in EMOTIONS:
        group = [
            sample for sample in samples
            if sample.human_plurality_emotion == label
            and sample.human_plurality_probability is not None
            and float(sample.metadata.get("human_plurality_margin", 0.0)) > 1e-12
        ]
        group.sort(key=lambda sample: (
            -sample.human_plurality_probability,
            -float(sample.metadata.get("human_plurality_margin", 0.0)),
            sample.sample_id,
        ))
        if len(group) < per_class:
            raise ValueError(
                "Only {} human-plurality samples available for {}; requested {}"
                .format(len(group), label, per_class)
            )
        for rank, sample in enumerate(group[:per_class], start=1):
            sample.split = "held_out_audit"
            sample.metadata.update({
                "audit_selection": "top_unique_human_plurality_probability_per_emotion",
                "human_confidence_rank_within_emotion": rank,
            })
            selected.append(sample)
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
