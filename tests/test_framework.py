from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from affective_twins.datasets import (
    deterministic_split,
    high_confidence_human_subset,
    load_emotion6,
    load_sample_manifest,
)
from affective_twins.interventions.color import ColorIntervention
from affective_twins.interventions.context import ContextIntervention
from affective_twins.interventions.base import GeneratedTwin
from affective_twins.interventions.text import TextIntervention
from affective_twins.metrics import aggregate, bootstrap_mean_ci, js_divergence, pair_metrics
from affective_twins.models.smolvlm import SmolVLMAdapter
from affective_twins.models.resnet import SampleDataset
from affective_twins.runner import _report_condition_candidates, _require_sample_images
from affective_twins.schema import AffectPrediction, AffectSample, CueFamily


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
    by_id = {sample.sample_id: sample for sample in samples}
    assert by_id["anger-55"].nominal_emotion == "anger"
    assert by_id["anger-55"].human_plurality_emotion == "sadness"
    assert by_id["anger-55"].emotion == "sadness"
    assert by_id["anger-55"].human_plurality_probability == pytest.approx(0.8)
    assert by_id["anger-55"].emotion_distribution["neutral"] == pytest.approx(0.0333333)
    assert by_id["disgust-55"].human_plurality_emotion == "disgust"
    assert by_id["disgust-55"].human_plurality_probability == pytest.approx(0.5)
    counts = {name: 0 for name in ["train", "validation", "test"]}
    for sample in deterministic_split(samples):
        counts[sample.split] += 1
    assert counts == {"train": 1155, "validation": 162, "test": 338}


def test_high_confidence_audit_selection_uses_human_labels():
    samples = load_emotion6(
        PROJECT.parent / "Emotional_colorTransfer" / "implementation" / "Emotion6",
        PROJECT.parent / "Emotional_colorTransfer" / "Emotion61" / "ground_truth.csv",
    )
    selected = high_confidence_human_subset(samples, per_class=10)
    assert len(selected) == 60
    assert {label: sum(sample.emotion == label for sample in selected) for label in [
        "anger", "disgust", "fear", "joy", "sadness", "surprise"
    ]} == {
        "anger": 10, "disgust": 10, "fear": 10,
        "joy": 10, "sadness": 10, "surprise": 10,
    }
    assert all(sample.emotion == sample.human_plurality_emotion for sample in selected)
    assert all(sample.nominal_emotion in {
        "anger", "disgust", "fear", "joy", "sadness", "surprise"
    } for sample in selected)
    assert all(sample.split == "held_out_audit" for sample in selected)
    assert all(sample.metadata["human_plurality_margin"] > 0 for sample in selected)


def test_resnet_training_target_uses_human_distribution(tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (4, 4), "white").save(image_path)
    sample = AffectSample(
        sample_id="sample",
        image_path=str(image_path),
        emotion="sadness",
        emotion_distribution={"anger": 0.1, "sadness": 0.6, "neutral": 0.3},
    )
    _, target, _, _ = SampleDataset([sample], lambda image: np.asarray(image))[0]
    assert target.tolist() == pytest.approx([1 / 7, 0, 0, 0, 6 / 7, 0])


def test_portable_audit_manifest_preserves_human_and_folder_labels():
    samples = load_sample_manifest(PROJECT / "data" / "audit80" / "manifest.jsonl")
    emotion6 = [sample for sample in samples if sample.metadata.get("source_dataset") == "Emotion6"]
    assert len(samples) == 80
    assert len(emotion6) == 60
    assert all(sample.human_plurality_emotion == sample.emotion for sample in emotion6)
    assert all(sample.nominal_emotion for sample in emotion6)
    assert all("neutral" in sample.emotion_distribution for sample in emotion6)
    assert all(Path(sample.image_path).is_file() for sample in samples)


def test_abaw_manifest_has_20_annotation_backed_face_samples():
    samples = load_sample_manifest(PROJECT / "data" / "abaw_au20" / "manifest.jsonl")
    assert len(samples) == 20
    assert {sample.emotion for sample in samples} == {"anger", "disgust", "fear", "joy", "sadness", "surprise"}
    assert all(sample.metadata["annotation_backed"] for sample in samples)
    assert all(sample.metadata["active_aus"] for sample in samples)
    assert all(Path(sample.image_path).is_file() for sample in samples)


def test_generation_preflight_lists_missing_manifest_images(tmp_path):
    samples = [
        AffectSample("missing-a", str(tmp_path / "a.jpg"), "joy"),
        AffectSample("missing-b", str(tmp_path / "b.jpg"), "sadness"),
    ]
    with pytest.raises(FileNotFoundError, match="2 missing image file") as error:
        _require_sample_images(samples)
    assert "a.jpg" in str(error.value)
    assert "b.jpg" in str(error.value)


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
    twins = ColorIntervention(
        mask_provider=lambda _: [], superpixel_fallback=True, superpixel_count=36
    ).generate(Image.fromarray(array))
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


def test_panoptic_colour_intervention_keeps_each_scene_entity():
    image = Image.new("RGB", (100, 80), (120, 120, 120))
    yy, xx = np.mgrid[:80, :100]
    person = ((xx - 50) / 15) ** 2 + ((yy - 42) / 30) ** 2 <= 1.0
    wall = (yy < 52) & ~person
    floor = (yy >= 52) & ~person
    generator = ColorIntervention(panoptic_provider=lambda _: [
        {"mask": person, "label": "person", "score": 0.99, "segment_id": 7, "label_id": 0},
        {"mask": wall, "label": "wall-other", "score": 0.97, "segment_id": 8, "label_id": 116},
        {"mask": floor, "label": "floor-wood", "score": 0.96, "segment_id": 9, "label_id": 104},
    ])
    twins = generator.generate(image)
    by_label = {twin.metadata["panoptic_label"]: twin for twin in twins}
    assert set(by_label) == {"person", "wall-other", "floor-wood"}
    assert all(twin.operation == "panoptic_entity_chroma_removal" for twin in twins)
    assert np.array_equal(np.asarray(by_label["person"].mask) > 0, person)
    assert by_label["person"].metadata["panoptic_segment_id"] == 7
    assert by_label["wall-other"].metadata["intervention_scope"] == "complete_panoptic_entity"


def test_context_mask_and_twin_are_well_formed():
    image = Image.new("RGB", (120, 90), (50, 120, 200))
    twins = ContextIntervention().generate(image)
    assert len(twins) == 1
    mask = np.asarray(twins[0].mask)
    assert mask.shape == (90, 120)
    assert 0 < mask.mean() < 255


def test_panoptic_context_preserves_foreground_and_changes_only_background():
    array = np.zeros((90, 120, 3), dtype=np.uint8)
    array[..., 0], array[..., 1], array[..., 2] = 40, 150, 220
    image = Image.fromarray(array)
    foreground = np.zeros((90, 120), dtype=np.uint8)
    foreground[15:80, 35:85] = 255
    twin = ContextIntervention().generate(image, Image.fromarray(foreground))[0]
    output = np.asarray(twin.image)
    subject = foreground > 0
    assert np.array_equal(output[subject], array[subject])
    assert not np.array_equal(output[~subject], array[~subject])
    assert twin.metadata["foreground_estimator"] == "mask2former_panoptic_thing_union"


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


def test_same_vlm_grounds_reported_cue_to_candidate_region():
    class StubVLM(SmolVLMAdapter):
        def __init__(self):
            self.model_name = "stub"
            self.cache_dir = None

        def _ask(self, image, question, max_new_tokens=12):
            return "option_1"

    class StubLocator:
        def predict(self, images):
            return [prediction("joy") for _ in images], None

    source = Image.new("RGB", (20, 20), "orange")
    first_mask = Image.new("L", (20, 20), 0)
    second_mask = Image.new("L", (20, 20), 0)
    ImageDraw.Draw(first_mask).rectangle((0, 0, 7, 19), fill=255)
    ImageDraw.Draw(second_mask).rectangle((10, 0, 19, 19), fill=255)
    candidates = [
        GeneratedTwin(source, first_mask, CueFamily.COLOR, "desaturate", metadata={"panoptic_label": "wall"}),
        GeneratedTwin(source, second_mask, CueFamily.COLOR, "desaturate", metadata={"panoptic_label": "person"}),
    ]
    report = prediction("joy")
    report.evidence_cue = "color_lighting"
    report.evidence = "the person's bright clothing"
    report.raw["parse_status"] = "valid_constrained"
    selected, status = _report_condition_candidates(
        StubVLM(), StubLocator(), source, candidates,
        AffectSample("sample", "unused.jpg", "joy"), report, include_control=True,
    )
    assert status == "reported_cue_target"
    assert selected[0].metadata["selected_candidate_label"].startswith("person (right")
    assert selected[0].metadata["report_condition_role"] == "reported_cue_target"
    assert selected[0].metadata["selection_model"] == "exact_reported_evidence_grounding"
    assert selected[0].metadata["reported_evidence_region_match"]
    assert not selected[0].metadata["is_control"]
    assert selected[1].metadata["report_condition_role"] == "same_cue_matched_region_control"
    assert selected[1].metadata["is_control"]


def test_vlm_evidence_phrase_is_conditioned_on_its_emotion_and_cue_report():
    class StubVLM(SmolVLMAdapter):
        def __init__(self):
            self.questions = []
            self.allowed_cues = ["color_lighting", "facial_action_region", "scene_context"]

        def _ask(self, image, question, max_new_tokens=12):
            self.questions.append(question)
            if question.startswith("Classify the apparent emotion"):
                return "joy"
            if question.startswith("Choose the strongest visible affect cue"):
                return "color_lighting"
            if question.startswith("Classify apparent valence"):
                return "positive"
            if question.startswith("Classify apparent arousal"):
                return "high"
            if question.startswith("How confident"):
                return "high"
            if question.startswith("You classified"):
                return "bright yellow clothing"
            return "A person wears bright clothing."

    vlm = StubVLM()
    report = vlm._predict_one(Image.new("RGB", (20, 20), "yellow"))
    cue_question = next(question for question in vlm.questions if question.startswith("Choose the strongest"))
    evidence_question = next(question for question in vlm.questions if question.startswith("You classified"))
    assert "embedded_text" not in cue_question
    assert "joy" in evidence_question
    assert "color or lighting" in evidence_question
    assert report.evidence == "bright yellow clothing"


def test_reported_mouth_evidence_intervenes_on_mouth_not_brows():
    class StubLocator:
        def predict(self, images):
            return [prediction("anger") for _ in images], None

    source = Image.new("RGB", (20, 20), "white")
    candidates = [
        GeneratedTwin(source, Image.new("L", (20, 20), 255), CueFamily.FACE, "ablate", metadata={"au_region": region})
        for region in ["brow_AU1_2_4", "eye_AU5_6_7", "mouth_AU10_12_15_20_23_24_25_26"]
    ]
    report = prediction("anger")
    report.evidence_cue = "facial_action_region"
    report.evidence = "mouth"
    report.raw["parse_status"] = "valid_constrained"
    selected, status = _report_condition_candidates(
        None, StubLocator(), source, candidates,
        AffectSample("sample", "unused.jpg", "anger"), report, include_control=False,
    )
    assert status == "reported_cue_target"
    assert selected[0].metadata["au_region"].startswith("mouth")
    assert selected[0].metadata["reported_evidence_matched_terms"] == ["mouth"]


def test_unmatched_object_evidence_is_rejected_instead_of_substituted():
    class StubLocator:
        def predict(self, images):
            return [prediction("disgust") for _ in images], None

    source = Image.new("RGB", (20, 20), "green")
    candidates = [
        GeneratedTwin(source, Image.new("L", (20, 20), 255), CueFamily.COLOR, "desaturate", metadata={"panoptic_label": label})
        for label in ["river", "boat"]
    ]
    report = prediction("disgust")
    report.evidence_cue = "color_lighting"
    report.evidence = "plastic bottles"
    report.raw["parse_status"] = "valid_constrained"
    selected, status = _report_condition_candidates(
        None, StubLocator(), source, candidates,
        AffectSample("sample", "unused.jpg", "disgust"), report, include_control=False,
    )
    assert selected == []
    assert status == "reported_evidence_absent_from_candidate_labels"


def test_vlm_region_choice_rejects_option_order_bias():
    class FirstOptionVLM(SmolVLMAdapter):
        def __init__(self):
            self.model_name = "stub"
            self.cache_dir = None

        def _ask(self, image, question, max_new_tokens=12):
            return "option_0"

    result = FirstOptionVLM().select_evidence_region(
        Image.new("RGB", (20, 20), "white"),
        "color_lighting",
        ["wall (left)", "person (right)"],
        evidence="person",
        cache_key="sample",
    )
    assert result["index"] is None
    assert result["status"] == "invalid_or_order_sensitive"


def test_reported_text_requires_ocr_grounding_in_original():
    class StubLocator:
        def predict(self, images):
            return [prediction("sadness") for _ in images], None

    source = Image.new("RGB", (20, 20), "black")
    conflict = GeneratedTwin(
        source,
        Image.new("L", (20, 20), 255),
        CueFamily.TEXT,
        "insert_affect_conflict_text",
        metadata={"inserted_text": "JOY"},
    )
    report = prediction("sadness")
    report.evidence_cue = "embedded_text"
    report.evidence = "a sad word"
    report.raw["parse_status"] = "valid_constrained"
    selected, status = _report_condition_candidates(
        None, StubLocator(), source, [conflict],
        AffectSample("sample", "unused.jpg", "sadness"), report, include_control=True,
    )
    assert selected == []
    assert status == "reported_text_not_groundable_by_ocr"


def test_report_conditioned_metrics_compare_reported_target_to_controls():
    def row(cue, role, vlm_drop, source_drop, is_control=0.0):
        return {
            "sample_id": "sample",
            "cue_family": cue,
            "operation": "test_edit",
            "eligible": True,
            "is_control": is_control,
            "original_confidence": 0.78,
            "original_correct": 1.0,
            "original_folder_correct": float("nan"),
            "folder_human_agreement": float("nan"),
            "original_brier": 0.1,
            "original_nll": 0.2,
            "original_valence_absolute_error": 0.1,
            "original_arousal_absolute_error": 0.1,
            "directional_success": float(source_drop > 0),
            "source_probability_drop": source_drop,
            "emotion_js_divergence": 0.05,
            "va_distance": 0.1,
            "feature_cosine": 0.95,
            "entropy_change": 0.04,
            "report_condition_role": role,
            "vlm_valid": 1.0,
            "vlm_cue_grounded": 1.0,
            "vlm_caption_jaccard": 0.8,
            "vlm_original_class_probability_drop": vlm_drop,
            "vlm_original_prediction_flip": float(vlm_drop > 0.1),
            "vlm_entropy_change": 0.05,
            "vlm_reported_cue_retained": 0.0,
        }

    summary = aggregate([
        row("color_lighting", "reported_cue_target", 0.20, 0.12),
        row("scene_context", "unreported_cue_comparator", 0.03, 0.01),
        row("color_lighting", "same_cue_matched_region_control", 0.04, 0.02, is_control=1.0),
    ])
    faithfulness = summary["report_conditioned_faithfulness"]
    assert faithfulness["original_class_probability_drop_mean"] == pytest.approx(0.20)
    assert faithfulness["reported_minus_unreported_drop_mean"] == pytest.approx(0.17)
    assert faithfulness["reported_minus_same_cue_control_drop_mean"] == pytest.approx(0.16)
    assert faithfulness["prediction_flip_rate"] == 1.0


def test_three_cue_audit_does_not_penalize_absent_text_family():
    row = {
        "sample_id": "sample",
        "cue_family": "color_lighting",
        "operation": "test_edit",
        "eligible": True,
        "is_control": 0.0,
        "original_confidence": 0.7,
        "original_correct": 1.0,
        "original_folder_correct": float("nan"),
        "folder_human_agreement": float("nan"),
        "original_brier": 0.1,
        "original_nll": 0.2,
        "original_valence_absolute_error": 0.1,
        "original_arousal_absolute_error": 0.1,
        "directional_success": 1.0,
        "source_probability_drop": 0.1,
        "emotion_js_divergence": 0.02,
        "va_distance": 0.1,
        "feature_cosine": 0.95,
        "entropy_change": 0.01,
        "report_condition_role": "unreported_cue_comparator",
    }
    summary = aggregate(
        [row],
        expected_cues=["color_lighting", "facial_action_region", "scene_context"],
    )
    assert summary["cue_coverage"] == pytest.approx(1 / 3)
    assert np.isnan(summary["conflict_uncertainty_success_rate"])
    assert "conflict uncertainty" not in summary["cause_note"]
