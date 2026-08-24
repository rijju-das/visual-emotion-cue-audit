"""Human-validation template and simple agreement summaries."""

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

from .io import write_csv


def write_annotation_template(path: Path, interventions: List[Dict]) -> None:
    rows = []
    for intervention in interventions:
        if not intervention["eligible"]:
            continue
        rows.append({
            "intervention_id": intervention["intervention_id"],
            "sample_id": intervention["sample_id"],
            "cue_family_ground_truth": intervention["cue_family"],
            "operation": intervention["operation"],
            "annotator_id": "",
            "original_emotion": "",
            "twin_emotion": "",
            "original_valence_-1_to_1": "",
            "twin_valence_-1_to_1": "",
            "original_arousal_-1_to_1": "",
            "twin_arousal_-1_to_1": "",
            "identified_changed_cue": "",
            "unchanged_content_preserved_1_to_5": "",
            "edit_artifact_1_to_5": "",
            "caption_faithful_1_to_5": "",
            "notes": "",
        })
    write_csv(path, rows)


def summarise_annotations(path: Path) -> Dict:
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("annotator_id", "").strip()]
    if not rows:
        return {"n_annotations": 0, "status": "template_has_no_completed_annotations"}
    numeric = lambda key: [float(row[key]) for row in rows if row.get(key, "").strip()]
    cue_accuracy = np.mean([row["identified_changed_cue"].strip() == row["cue_family_ground_truth"].strip() for row in rows])
    return {
        "n_annotations": len(rows),
        "n_annotators": len({row["annotator_id"] for row in rows}),
        "n_interventions": len({row["intervention_id"] for row in rows}),
        "cue_identification_accuracy": float(cue_accuracy),
        "emotion_change_rate": float(np.mean([row["original_emotion"].strip() != row["twin_emotion"].strip() for row in rows])),
        "content_preservation_mean": float(np.mean(numeric("unchanged_content_preserved_1_to_5"))),
        "artifact_mean": float(np.mean(numeric("edit_artifact_1_to_5"))),
        "annotations_by_cue": dict(Counter(row["cue_family_ground_truth"] for row in rows)),
    }

