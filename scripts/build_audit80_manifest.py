#!/usr/bin/env python3
"""Build a portable audit set from high-confidence human Emotion6 labels and ABAW AUs."""

import json
import shutil
from collections import Counter
from pathlib import Path

from affective_twins.datasets import high_confidence_human_subset, load_emotion6, load_sample_manifest


PROJECT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = PROJECT.parent / "Emotional_colorTransfer" / "implementation" / "Emotion6"
GROUND_TRUTH = PROJECT.parent / "Emotional_colorTransfer" / "Emotion61" / "ground_truth.csv"
ABAW_MANIFEST = PROJECT / "data" / "abaw_au20" / "manifest.jsonl"
OUTPUT = PROJECT / "data" / "audit80"
PER_EMOTION = 10


def main():
    emotion6 = high_confidence_human_subset(
        load_emotion6(IMAGE_ROOT, GROUND_TRUTH),
        per_class=PER_EMOTION,
    )
    abaw = load_sample_manifest(ABAW_MANIFEST)
    samples = emotion6 + abaw
    if len(samples) != 80:
        raise RuntimeError("Expected 80 source samples, found {}".format(len(samples)))
    image_dir = OUTPUT / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    portable = []
    expected_image_names = set()
    for sample in samples:
        row = sample.to_dict()
        source = Path(sample.image_path)
        suffix = source.suffix.lower() or ".jpg"
        destination = image_dir / "{}{}".format(row["sample_id"], suffix)
        shutil.copy2(str(source), str(destination))
        expected_image_names.add(destination.name)
        row["image_path"] = str(Path("images") / destination.name)
        portable.append(row)
    for stale in image_dir.iterdir():
        if stale.is_file() and stale.name not in expected_image_names:
            stale.unlink()
    manifest = OUTPUT / "manifest.jsonl"
    with manifest.open("w") as stream:
        for row in portable:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "n": len(portable),
        "emotion6": len(emotion6),
        "abaw_au": sum(row.get("metadata", {}).get("source_dataset") == "Aff-Wild2/ABAW" for row in portable),
        "emotion6_by_human_plurality": dict(Counter(sample.emotion for sample in emotion6)),
        "emotion6_by_folder_label": dict(Counter(sample.nominal_emotion for sample in emotion6)),
        "folder_human_agreement_count": sum(
            sample.nominal_emotion == sample.human_plurality_emotion for sample in emotion6
        ),
        "minimum_selected_probability_by_emotion": {
            label: min(
                sample.human_plurality_probability
                for sample in emotion6 if sample.human_plurality_emotion == label
            )
            for label in sorted({sample.human_plurality_emotion for sample in emotion6})
        },
        "selection": "top_{}_unique_human_plurality_probability_per_emotion".format(PER_EMOTION),
        "manifest": str(manifest.relative_to(PROJECT)),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
