from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from affective_twins.datasets import deterministic_split, load_emotion6, load_sample_manifest
from affective_twins.interventions.color import ColorIntervention
from affective_twins.interventions.context import ContextIntervention
from affective_twins.interventions.text import TextIntervention
from affective_twins.metrics import aggregate, bootstrap_mean_ci, js_divergence, pair_metrics
from affective_twins.models.smolvlm import SmolVLMAdapter
from affective_twins.schema import AffectPrediction


PROJECT = Path(__file__).resolve().parents[1]


def prediction(label="joy", confidence=0.7, valence=0.5, arousal=0.3):
    probabilities = {name: 0.06 for name in ["anger", "disgust", "fear", "joy", "sadness", "surprise"]}
    probabilities[label] = confidence
    total = sum(probabilities.values())
    probabilities = {key: value / total for key, value in probabilities.items()}
    return AffectPrediction(probabilities, valence, arousal, probabilities[label], label)


def test_emotion6_loader_and_split():
    samples = load_emotion6(
        PROJECT.parent / "Emotional_colorTransfer" / "implementation" / "Emotion6",
        PROJECT.parent / "Emotional_colorTransfer" / "Emotion61" / "ground_truth.csv",
    )
    assert len(samples) == 1980
    assert all(sample.valence is None or -1 <= sample.valence <= 1 for sample in samples)
    counts = {name: 0 for name in ["train", "validation", "test"]}
    for sample in deterministic_split(samples):
        counts[sample.split] += 1
    assert counts == {"train": 1380, "validation": 198, "test": 402}


def test_abaw_manifest_has_20_annotation_backed_face_samples():
    samples = load_sample_manifest(PROJECT / "data" / "abaw_au20" / "manifest.jsonl")
    assert len(samples) == 20
    assert {sample.emotion for sample in samples} == {"anger", "disgust", "fear", "joy", "sadness", "surprise"}
    assert all(sample.metadata["annotation_backed"] for sample in samples)
    assert all(sample.metadata["active_aus"] for sample in samples)
    assert all(Path(sample.image_path).is_file() for sample in samples)


def test_colour_intervention_preserves_unmasked_pixels_and_luminance():
    array = np.zeros((80, 80, 3), dtype=np.uint8)
    array[..., 0], array[..., 1], array[..., 2] = 220, 80, 30
    image = Image.fromarray(array)
    irregular_object = np.zeros((80, 80), dtype=np.float32)
    irregular_object[10:55, 12:42] = 1.0
    irregular_object[42:70, 30:65] = 1.0
    twins = ColorIntervention(
        mask_provider=lambda _: [{"mask": irregular_object, "label": "chair", "score": 0.98}]
    ).generate(image)
    twin = twins[0]
    output, mask = np.asarray(twin.image), np.asarray(twin.mask) > 0
    assert [item.operation for item in twins] == [
        "complete_subject_chroma_removal",
        "background_exposure_reduction",
    ]
    assert twin.metadata["subject_labels"] == ["chair"]
    assert np.array_equal(output[~mask], array[~mask])
    before = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
    after = 0.2126 * output[..., 0] + 0.7152 * output[..., 1] + 0.0722 * output[..., 2]
    assert np.abs(before[mask] - after[mask]).mean() < 1.0


def test_colour_intervention_merges_people_and_preserves_subject_in_background_twin():
    array = np.full((80, 100, 3), (180, 120, 60), dtype=np.uint8)
    image = Image.fromarray(array)
    left_person = np.zeros((80, 100), dtype=np.float32)
    right_person = np.zeros((80, 100), dtype=np.float32)
    yy, xx = np.mgrid[:80, :100]
    left_person[((xx - 20) / 12) ** 2 + ((yy - 42) / 34) ** 2 <= 1.0] = 1.0
    right_person[((xx - 78) / 13) ** 2 + ((yy - 45) / 30) ** 2 <= 1.0] = 1.0
    twins = ColorIntervention(mask_provider=lambda _: [
        {"mask": left_person, "label": "person", "score": 0.99},
        {"mask": right_person, "label": "person", "score": 0.97},
    ]).generate(image)
    subject_mask = np.asarray(twins[0].mask) > 0
    background_mask = np.asarray(twins[1].mask) > 0
    expected_subject = np.logical_or(left_person > 0, right_person > 0)
    assert np.array_equal(subject_mask, expected_subject)
    assert np.array_equal(background_mask, ~expected_subject)
    assert twins[0].metadata["subject_type"] == "all_detected_people"
    assert twins[0].metadata["subject_instance_count"] == 2
    background_output = np.asarray(twins[1].image)
    assert np.array_equal(background_output[subject_mask], array[subject_mask])
    assert background_output[background_mask].mean() < array[background_mask].mean()


def test_colour_intervention_uses_nonrectangular_superpixel_fallback():
    yy, xx = np.mgrid[:96, :96]
    array = np.full((96, 96, 3), 30, dtype=np.uint8)
    array[(xx - 48) ** 2 + (yy - 48) ** 2 < 28 ** 2] = (230, 60, 30)
    twins = ColorIntervention(mask_provider=lambda _: [], superpixel_count=36).generate(Image.fromarray(array))
    mask = np.asarray(twins[0].mask) > 0
    rows, columns = np.where(mask)
    bounding_box_area = (rows.max() - rows.min() + 1) * (columns.max() - columns.min() + 1)
    assert twins[0].metadata["used_superpixel_fallback"]
    assert twins[0].metadata["subject_type"] == "salient_superpixel_region"
    assert twins[0].metadata["superpixel_component_count"] >= 2
    assert twins[0].metadata["subject_mask_rectangularity"] < 0.86
    assert "grid_cell" not in twins[0].metadata
    assert mask.sum() < 0.9 * bounding_box_area


def test_colour_intervention_keeps_person_wall_and_floor_as_separate_entities():
    image = Image.new("RGB", (100, 80), (120, 120, 120))
    yy, xx = np.mgrid[:80, :100]
    person = ((xx - 50) / 15) ** 2 + ((yy - 42) / 30) ** 2 <= 1.0
    wall = (yy < 52) & ~person
    floor = (yy >= 52) & ~person
    generator = ColorIntervention(semantic_provider=lambda _: [
        {"mask": person, "label": "person", "score": 0.99},
        {"mask": wall, "label": "wall", "score": 0.97},
        {"mask": floor, "label": "floor", "score": 0.96},
    ])
    twins = generator.generate(image)
    by_label = {twin.metadata["semantic_label"]: twin for twin in twins}
    assert {"person", "wall", "floor"} <= set(by_label)
    assert all(twin.operation == "semantic_region_chroma_removal" for twin in twins)
    assert np.array_equal(np.asarray(by_label["person"].mask) > 0, person)
    assert by_label["person"].metadata["is_complete_person_region"]
    assert "grid_cell" not in by_label["wall"].metadata


def test_context_mask_and_twin_are_well_formed():
    image = Image.new("RGB", (120, 90), (50, 120, 200))
    twins = ContextIntervention().generate(image)
    assert len(twins) == 1
    mask = np.asarray(twins[0].mask)
    assert mask.shape == (90, 120)
    assert 0 < mask.mean() < 255


def test_text_removal_and_conflict_with_stubbed_ocr(tmp_path):
    class StubText(TextIntervention):
        def _boxes(self, image_path):
            return [(10, 10, 80, 35, "happy", 96.0)]

    image = Image.new("RGB", (160, 100), "navy")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "happy", fill="white")
    twins = StubText("/unused", add_conflict=True).generate(image, tmp_path / "x.png", "joy")
    assert [twin.operation for twin in twins] == ["remove_detected_text", "insert_affect_conflict_text"]
    assert twins[1].metadata["inserted_text"] == "SADNESS"


def test_metrics_and_cause_summary_are_bounded():
    original, twin = prediction("joy", 0.75), prediction("sadness", 0.65, -0.4, 0.2)
    assert 0 <= js_divergence(original.emotion_probabilities, twin.emotion_probabilities) <= 1
    metrics = pair_metrics(original, twin, "joy", 0.93, 0.1, "attenuate_source_affect")
    row = {"sample_id": "synthetic", "cue_family": "color_lighting", "operation": "local_chroma_removal", "eligible": True, "original_confidence": original.confidence, "original_correct": 1.0, "original_brier": 0.1, "original_nll": 0.2, "original_valence_absolute_error": 0.1, "original_arousal_absolute_error": 0.1, **metrics}
    summary = aggregate([row])
    assert 0 <= summary["cause_diagnostic_score"] <= 1
    assert summary["directional_success_rate"] == 1.0
    low, high = bootstrap_mean_ci([0.1, 0.2, 0.3])
    assert low <= 0.2 <= high


def test_vlm_json_parser():
    raw = '{"emotion":"joy","emotion_probabilities":{"anger":0.02,"disgust":0.02,"fear":0.02,"joy":0.86,"sadness":0.04,"surprise":0.04},"valence":0.8,"arousal":0.5,"confidence":0.86,"evidence_cue":"color_lighting","evidence":"bright colours","caption":"A bright scene."}'
    parsed = SmolVLMAdapter._parse(raw)
    assert parsed.predicted_emotion == "joy"
    assert abs(sum(parsed.emotion_probabilities.values()) - 1.0) < 1e-8
    assert parsed.evidence_cue == "color_lighting"


def test_vlm_free_text_fallback_is_explicit():
    parsed = SmolVLMAdapter._parse("A bright scene with a smiling face.")
    assert parsed.predicted_emotion == "surprise"
    assert parsed.raw["parse_status"] == "fallback_free_text"
    assert parsed.confidence <= 0.35


def test_vlm_constrained_choice_parser():
    assert SmolVLMAdapter._choice(" Joy.", ["anger", "joy"]) == "joy"
    assert SmolVLMAdapter._choice(" Color lighting.", ["color_lighting", "scene_context"]) == "color_lighting"
    assert SmolVLMAdapter._choice("unknown", ["low", "high"]) is None
