from pathlib import Path

from PIL import Image

from affective_twins.interventions.face import FaceActionRegionIntervention


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
        "eye_AU5_6_7",
        "mouth_AU10_12_15_20_23_24_25_26",
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
