#!/usr/bin/env python3
"""Create a deterministic 20-frame, AU-annotated Aff-Wild2 audit subset.

The selected images are copied into this project for a self-contained local
experiment. Aff-Wild2 redistribution terms still apply; the manifest records
that these frames must not be released independently of the source dataset.
"""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

from affective_twins.interventions.face import AU_GROUPS, FaceActionRegionIntervention


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT.parent / "VA_new_work"
ANNOTATIONS = SOURCE / "ABAW Annotations"
IMAGE_ROOT = SOURCE / "cropped_aligned"
OUTPUT = PROJECT / "data" / "abaw_au20"
AU_NAMES = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"]
EXPRESSION = {1: "anger", 2: "disgust", 3: "fear", 4: "joy", 5: "sadness", 6: "surprise"}
TARGET = {"anger": 4, "disgust": 3, "fear": 3, "joy": 4, "sadness": 3, "surprise": 3}


def annotation_files(challenge: str):
    return {
        path.stem: path
        for path in (ANNOTATIONS / challenge / "Train_Set").glob("*.txt")
    }


def candidates():
    files = {
        "au": annotation_files("AU_Detection_Challenge"),
        "expr": annotation_files("EXPR_Recognition_Challenge"),
        "va": annotation_files("VA_Estimation_Challenge"),
    }
    videos = sorted(set(files["au"]) & set(files["expr"]) & set(files["va"]))
    result = defaultdict(list)
    for video in videos:
        folder = IMAGE_ROOT / video
        if not folder.is_dir():
            continue
        rows = {key: path.read_text().splitlines()[1:] for key, path in ((key, files[key][video]) for key in files)}
        length = min(len(values) for values in rows.values())
        for index in range(0, length, 5):
            try:
                expression_id = int(rows["expr"][index].strip())
                au_vector = [int(value) for value in rows["au"][index].split(",")[: len(AU_NAMES)]]
                valence, arousal = [float(value) for value in rows["va"][index].split(",")[:2]]
            except (ValueError, IndexError):
                continue
            image_path = folder / "{:05d}.jpg".format(index + 1)
            if (
                expression_id not in EXPRESSION
                or len(au_vector) != len(AU_NAMES)
                or min(au_vector) < 0
                or not any(au_vector)
                or min(valence, arousal) < -1.0
                or not image_path.is_file()
            ):
                continue
            active_aus = [name for name, active in zip(AU_NAMES, au_vector) if active]
            active_groups = [name for name, aus in AU_GROUPS.items() if set(aus) & set(active_aus)]
            if not active_groups:
                continue
            result[EXPRESSION[expression_id]].append({
                "video": video,
                "frame": index + 1,
                "image_path": image_path,
                "active_aus": active_aus,
                "active_groups": active_groups,
                "au_vector": au_vector,
                "valence": valence,
                "arousal": arousal,
            })
    return result


def main():
    rng = random.Random(42)
    pool = candidates()
    detector = FaceActionRegionIntervention(
        str(PROJECT.parent / "AU-Routing-VA" / "models" / "face_landmarker.task"),
        backend="opencv",
        dnn_prototxt=str(PROJECT / "assets" / "opencv_face_detector" / "deploy.prototxt"),
        dnn_model=str(PROJECT / "assets" / "opencv_face_detector" / "res10_300x300_ssd_iter_140000_fp16.caffemodel"),
    )
    output_images = OUTPUT / "images"
    output_images.mkdir(parents=True, exist_ok=True)
    chosen = []
    used_videos = set()
    used_groups = defaultdict(int)
    for emotion, target_count in TARGET.items():
        items = list(pool[emotion])
        rng.shuffle(items)
        items.sort(key=lambda item: (item["video"] in used_videos, sum(used_groups[group] for group in item["active_groups"])))
        for item in items:
            if sum(row["emotion"] == emotion for row in chosen) >= target_count:
                break
            if item["video"] in used_videos:
                continue
            with Image.open(item["image_path"]) as source:
                twins = detector.generate(
                    source.convert("RGB"),
                    active_aus=item["active_aus"],
                    trusted_face_crop=True,
                )
            if not twins:
                continue
            sample_id = "abaw-{}-{:05d}".format(item["video"].replace("_", "-"), item["frame"])
            destination = output_images / "{}.jpg".format(sample_id)
            shutil.copy2(str(item["image_path"]), str(destination))
            chosen.append({
                "sample_id": sample_id,
                "image_path": str(Path("images") / destination.name),
                "emotion": emotion,
                "emotion_distribution": {emotion: 1.0},
                "valence": item["valence"],
                "arousal": item["arousal"],
                "split": "external_test",
                "metadata": {
                    "source_dataset": "Aff-Wild2/ABAW",
                    "source_video": item["video"],
                    "frame_index": item["frame"],
                    "active_aus": item["active_aus"],
                    "au_vector": item["au_vector"],
                    "active_au_regions": item["active_groups"],
                    "allowed_cues": ["facial_action_region"],
                    "trusted_face_crop": True,
                    "annotation_backed": True,
                    "redistribution_restricted": True,
                },
            })
            used_videos.add(item["video"])
            for group in item["active_groups"]:
                used_groups[group] += 1
    if len(chosen) != sum(TARGET.values()):
        raise RuntimeError("Selected {} of {} requested AU frames".format(len(chosen), sum(TARGET.values())))
    manifest = OUTPUT / "manifest.jsonl"
    with manifest.open("w") as stream:
        for row in chosen:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "n": len(chosen),
        "by_emotion": {emotion: sum(row["emotion"] == emotion for row in chosen) for emotion in TARGET},
        "by_au": {name: sum(name in row["metadata"]["active_aus"] for row in chosen) for name in AU_NAMES},
        "by_region": dict(used_groups),
        "manifest": str(manifest),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
