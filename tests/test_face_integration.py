import math
from pathlib import Path

import numpy as np
from PIL import Image

from affective_twins.interventions.face import LANDMARK_COMPONENTS, FaceActionRegionIntervention


PROJECT = Path(__file__).resolve().parents[1]


def test_face_localizer_finds_at_least_one_emotion6_face():
    generator = FaceActionRegionIntervention(
        str(PROJECT.parent / "AU-Routing-VA" / "models" / "face_landmarker.task"),
        dnn_prototxt=str(PROJECT / "assets" / "opencv_face_detector" / "deploy.prototxt"),
        dnn_model=str(PROJECT / "assets" / "opencv_face_detector" / "res10_300x300_ssd_iter_140000_fp16.caffemodel"),
    )
    paths = list((PROJECT.parent / "Emotional_colorTransfer" / "implementation" / "Emotion6").glob("*/*"))[:120]
    found = []
    for path in paths:
        with Image.open(path) as source:
            found = generator.generate(source.convert("RGB"))
        if found:
            break
    assert found
    assert {twin.metadata["au_region"] for twin in found} >= {
        "brow_AU1_2_4",
        "eye_AU6_7",
        "mouth_AU10_12_15_23_24_25_26",
    }


def test_au_annotated_face_targets_only_active_regions_and_marks_controls():
    generator = FaceActionRegionIntervention(
        str(PROJECT / "assets" / "mediapipe" / "face_landmarker.task"),
        backend="opencv",
        dnn_prototxt=str(PROJECT / "assets" / "opencv_face_detector" / "deploy.prototxt"),
        dnn_model=str(PROJECT / "assets" / "opencv_face_detector" / "res10_300x300_ssd_iter_140000_fp16.caffemodel"),
    )
    path = next((PROJECT / "data" / "abaw_au20" / "images").glob("*.jpg"))
    with Image.open(path) as source:
        twins = generator.generate(source.convert("RGB"), active_aus=["AU1"], trusted_face_crop=True)
    targets = [twin for twin in twins if not twin.metadata["is_control"]]
    controls = [twin for twin in twins if twin.metadata["is_control"]]
    assert len(targets) == 1
    assert targets[0].metadata["target_active_aus"] == ["AU1"]
    assert controls
    assert all(not twin.metadata["target_active_aus"] for twin in controls)
    with Image.open(path) as source:
        original = np.asarray(source.convert("RGB"))
    target_output = np.asarray(targets[0].image)
    target_mask = np.asarray(targets[0].mask) > 0
    assert np.abs(target_output.astype(float)[target_mask] - original.astype(float)[target_mask]).mean() > 3.0
    assert targets[0].metadata["ablation_method"] == "expanded_landmark_mask_strong_blur_and_pixelation"


def test_landmark_masks_trace_separate_brows_eyes_and_mouth():
    image = Image.new("RGB", (200, 160), "gray")
    face = [(0.5, 0.5)] * 478
    centres = {
        "brow_AU1_2_4": [(0.34, 0.34), (0.66, 0.34)],
        "eye_AU6_7": [(0.35, 0.46), (0.65, 0.46)],
        "mouth_AU10_12_15_23_24_25_26": [(0.50, 0.70)],
    }
    for group_name, components in LANDMARK_COMPONENTS.items():
        mutable = list(face)
        for centre, indices in zip(centres[group_name], components):
            for position, index in enumerate(indices):
                angle = 2.0 * math.pi * position / len(indices)
                mutable[index] = (centre[0] + 0.07 * math.cos(angle), centre[1] + 0.025 * math.sin(angle))
        face = mutable

    masks = {
        group_name: np.asarray(FaceActionRegionIntervention._group_mask(image, face, components))
        for group_name, components in LANDMARK_COMPONENTS.items()
    }
    assert all(mask.max() == 255 and mask.mean() > 0 for mask in masks.values())
    # The midpoint remains untouched: paired features are two contours, not one band.
    assert masks["brow_AU1_2_4"][54, 100] == 0
    assert masks["eye_AU6_7"][74, 100] == 0
