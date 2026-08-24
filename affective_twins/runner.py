"""End-to-end generation and evaluation orchestration."""

import platform
import sys
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from .datasets import balanced_subset, deterministic_split, load_emotion6, load_sample_manifest
from .interventions import ColorIntervention, ContextIntervention, FaceActionRegionIntervention, TextIntervention
from .io import read_jsonl, write_csv, write_json, write_jsonl
from .human import write_annotation_template
from .metrics import aggregate, pair_metrics
from .models import ResNetAffectModel, SmolVLMAdapter, train_independent_models
from .schema import AffectPrediction, AffectSample, CueFamily, Intervention


def load_samples(config: Dict) -> List[AffectSample]:
    dataset = config["dataset"]
    return deterministic_split(load_emotion6(Path(dataset["image_root"]), Path(dataset["ground_truth_csv"])), config["seed"])


def train(config: Dict) -> Dict[str, float]:
    return train_independent_models(load_samples(config), Path(config["model"]["checkpoint"]), config["seed"])


def load_audit_samples(config: Dict) -> List[AffectSample]:
    audit_manifest = config["dataset"].get("audit_manifest")
    if audit_manifest:
        return load_sample_manifest(Path(audit_manifest))
    samples = balanced_subset(
        load_samples(config),
        "test",
        int(config["run"]["limit_per_class"]),
        config["seed"] + 303,
    )
    manifest = config["dataset"].get("auxiliary_manifest")
    if manifest:
        samples.extend(load_sample_manifest(Path(manifest)))
    return samples


def _require_sample_images(samples: List[AffectSample]) -> None:
    missing = [Path(sample.image_path) for sample in samples if not Path(sample.image_path).is_file()]
    if missing:
        preview = "\n".join("  - {}".format(path) for path in missing[:20])
        remainder = "\n  ... and {} more".format(len(missing) - 20) if len(missing) > 20 else ""
        raise FileNotFoundError(
            "Audit manifest references {} missing image file(s):\n{}{}\n"
            "Restore data/audit80/images from the licensed dataset copy before running generation."
            .format(len(missing), preview, remainder)
        )


def _select(locator: ResNetAffectModel, source: Image.Image, candidates, emotion: str, include_control: bool = False):
    if not candidates:
        return []
    predictions, _ = locator.predict([source] + [candidate.image for candidate in candidates])
    base = predictions[0].emotion_probabilities[emotion]
    drops = [base - prediction.emotion_probabilities[emotion] for prediction in predictions[1:]]
    target_indices = [index for index, candidate in enumerate(candidates) if not candidate.metadata.get("is_control", False)]
    if not target_indices:
        return []
    selected = max(target_indices, key=lambda index: drops[index])
    candidates[selected].metadata.update({
        "candidate_count": len(candidates),
        "selection_model": "independent_locator",
        "locator_source_probability_drop": float(drops[selected]),
        "locator_all_probability_drops": [float(value) for value in drops],
    })
    result = [candidates[selected]]
    if include_control:
        explicit_controls = [index for index, candidate in enumerate(candidates) if candidate.metadata.get("is_control", False)]
        if explicit_controls:
            control_index = min(explicit_controls, key=lambda index: abs(drops[index]))
        elif candidates[selected].cue_family == CueFamily.COLOR:
            remaining = [index for index in range(len(candidates)) if index != selected]
            selected_area = float(candidates[selected].metadata.get("object_mask_area_fraction", 0.0))
            control_index = min(
                remaining,
                key=lambda index: abs(
                    float(candidates[index].metadata.get("object_mask_area_fraction", 0.0)) - selected_area
                ),
            ) if remaining else None
            if control_index is not None:
                candidates[control_index].metadata.update({
                    "is_control": True,
                    "control_matching": "nearest_object_mask_area",
                    "target_object_mask_area_fraction": selected_area,
                })
        else:
            control_index = None
        if control_index is not None:
            candidates[control_index].operation = "matched_region_control"
            candidates[control_index].expected_direction = "control"
            candidates[control_index].metadata.update({
                "is_control": True,
                "control_for_candidate": selected,
                "locator_source_probability_drop": float(drops[control_index]),
                "selection_model": "matched_control",
            })
            result.append(candidates[control_index])
    return result


def _score_without_discarding(locator: ResNetAffectModel, source: Image.Image, candidates, emotion: str):
    """Record locator sensitivity while retaining each distinct intervention scope."""
    if not candidates:
        return []
    predictions, _ = locator.predict([source] + [candidate.image for candidate in candidates])
    base = predictions[0].emotion_probabilities[emotion]
    for candidate, prediction in zip(candidates, predictions[1:]):
        candidate.metadata.update({
            "candidate_count": len(candidates),
            "selection_model": "predefined_semantic_scope",
            "locator_source_probability_drop": float(base - prediction.emotion_probabilities[emotion]),
        })
    return candidates


def _retain_face_regions(
    locator: ResNetAffectModel,
    source: Image.Image,
    candidates,
    emotion: str,
    include_control: bool = False,
):
    """Retain every annotation-active AU group and at most one inactive control."""
    if not candidates:
        return []
    predictions, _ = locator.predict([source] + [candidate.image for candidate in candidates])
    base = predictions[0].emotion_probabilities[emotion]
    drops = [base - prediction.emotion_probabilities[emotion] for prediction in predictions[1:]]
    target_indices = [
        index for index, candidate in enumerate(candidates)
        if not candidate.metadata.get("is_control", False)
    ]
    for index in target_indices:
        candidates[index].metadata.update({
            "candidate_count": len(candidates),
            "selection_model": "retain_all_annotation_active_au_regions",
            "locator_source_probability_drop": float(drops[index]),
        })
    result = [candidates[index] for index in target_indices]
    controls = [
        index for index, candidate in enumerate(candidates)
        if candidate.metadata.get("is_control", False)
    ]
    if include_control and controls:
        control_index = min(controls, key=lambda index: abs(drops[index]))
        candidates[control_index].operation = "matched_region_control"
        candidates[control_index].expected_direction = "control"
        candidates[control_index].metadata.update({
            "is_control": True,
            "control_for_regions": [candidates[index].metadata.get("au_region") for index in target_indices],
            "locator_source_probability_drop": float(drops[control_index]),
            "selection_model": "single_inactive_au_region_control",
        })
        result.append(candidates[control_index])
    return result


def generate(config: Dict) -> Dict:
    output_dir = Path(config["run"]["output_dir"])
    checkpoint = Path(config["model"]["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError("Missing model checkpoint. Run `affective-twins train` first.")
    samples = load_audit_samples(config)
    _require_sample_images(samples)
    locator = ResNetAffectModel(checkpoint, role="locator")
    generators = [
        ColorIntervention(
            int(config["run"].get("grid_size", 4)),
            backend=config["run"].get("color_backend", "maskrcnn"),
            score_threshold=float(config["run"].get("object_score_threshold", 0.65)),
            mask_threshold=float(config["run"].get("object_mask_threshold", 0.5)),
            min_area_fraction=float(config["run"].get("object_min_area_fraction", 0.01)),
            max_area_fraction=float(config["run"].get("object_max_area_fraction", 0.65)),
            max_candidates=int(config["run"].get("object_max_candidates", 8)),
            superpixel_fallback=bool(config["run"].get("superpixel_fallback", False)),
            superpixel_count=int(config["run"].get("superpixel_count", 48)),
            panoptic_model=config["model"].get(
                "panoptic_model", "facebook/mask2former-swin-small-coco-panoptic"
            ),
            panoptic_score_threshold=float(config["run"].get("panoptic_score_threshold", 0.55)),
            panoptic_min_area_fraction=float(config["run"].get("panoptic_min_area_fraction", 0.03)),
            panoptic_local_files_only=bool(config["model"].get("panoptic_local_files_only", False)),
            semantic_model=config["model"].get(
                "semantic_model", "nvidia/segformer-b0-finetuned-ade-512-512"
            ),
            semantic_score_threshold=float(config["run"].get("semantic_score_threshold", 0.45)),
            semantic_local_files_only=bool(config["model"].get("semantic_local_files_only", False)),
        ),
        FaceActionRegionIntervention(
            config["assets"]["face_landmarker"],
            backend=config["run"].get("face_backend", "auto"),
            dnn_prototxt=config["assets"].get("face_detector_prototxt", ""),
            dnn_model=config["assets"].get("face_detector_model", ""),
        ),
        ContextIntervention(),
        TextIntervention(config["assets"]["tesseract"], bool(config["run"].get("text_conflict", True))),
    ]
    intervention_rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "samples.jsonl", [sample.to_dict() for sample in samples])
    for sample in samples:
        with Image.open(sample.image_path) as source_file:
            source = source_file.convert("RGB")
        context_foreground_mask = None
        for generator in generators:
            allowed_cues = sample.metadata.get("allowed_cues")
            if allowed_cues and generator.cue_family.value not in allowed_cues:
                continue
            if generator.cue_family == CueFamily.TEXT:
                candidates = generator.generate(source, Path(sample.image_path), sample.emotion)
            elif generator.cue_family == CueFamily.FACE:
                candidates = generator.generate(
                    source,
                    active_aus=sample.metadata.get("active_aus", []),
                    trusted_face_crop=bool(sample.metadata.get("trusted_face_crop", False)),
                )
            elif generator.cue_family == CueFamily.CONTEXT:
                candidates = generator.generate(source, foreground_mask=context_foreground_mask)
            else:
                candidates = generator.generate(source)
            if generator.cue_family == CueFamily.COLOR:
                candidates = _score_without_discarding(locator, source, candidates, sample.emotion)
                foreground_candidates = [
                    candidate for candidate in candidates
                    if candidate.metadata.get("panoptic_is_thing", False)
                    or candidate.metadata.get("semantic_label") == "person"
                    or candidate.metadata.get("intervention_scope") == "complete_subject"
                ]
                if foreground_candidates:
                    union = np.logical_or.reduce([
                        np.asarray(candidate.mask.convert("L")) > 0
                        for candidate in foreground_candidates
                    ])
                    context_foreground_mask = Image.fromarray(union.astype(np.uint8) * 255, mode="L")
            elif generator.cue_family == CueFamily.FACE:
                candidates = _retain_face_regions(
                    locator,
                    source,
                    candidates,
                    sample.emotion,
                    include_control=bool(config["run"].get("matched_controls", False)),
                )
            if not candidates:
                reason = getattr(generator, "unavailable_reason", "") or "no_eligible_region_detected"
                intervention_rows.append(Intervention(
                    intervention_id="{}--{}--skipped".format(sample.sample_id, generator.cue_family.value),
                    sample_id=sample.sample_id,
                    cue_family=generator.cue_family,
                    operation="skipped",
                    image_path="",
                    mask_path="",
                    eligible=False,
                    skip_reason=reason,
                ).to_dict())
                continue
            for index, candidate in enumerate(candidates):
                identifier = "{}--{}--{:02d}".format(sample.sample_id, candidate.cue_family.value, index)
                image_path = output_dir / "twins" / candidate.cue_family.value / "{}.png".format(identifier)
                mask_path = output_dir / "masks" / candidate.cue_family.value / "{}.png".format(identifier)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                candidate.image.save(image_path)
                candidate.mask.save(mask_path)
                intervention_rows.append(Intervention(
                    intervention_id=identifier,
                    sample_id=sample.sample_id,
                    cue_family=candidate.cue_family,
                    operation=candidate.operation,
                    image_path=str(image_path.resolve()),
                    mask_path=str(mask_path.resolve()),
                    target_region=candidate.target_region,
                    expected_direction=candidate.expected_direction,
                    metadata=candidate.metadata,
                ).to_dict())
    write_jsonl(output_dir / "interventions.jsonl", intervention_rows)
    write_annotation_template(output_dir / "human_validation_template.csv", intervention_rows)
    counts = Counter(row["cue_family"] for row in intervention_rows if row["eligible"])
    control_counts = Counter(
        row["cue_family"]
        for row in intervention_rows
        if row["eligible"] and row.get("metadata", {}).get("is_control", False)
    )
    skipped = Counter(row["cue_family"] for row in intervention_rows if not row["eligible"])
    provenance = {
        "framework_version": "0.3.0",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "seed": config["seed"],
        "sample_count": len(samples),
        "eligible_pairs_by_cue": dict(counts),
        "matched_controls_by_cue": dict(control_counts),
        "skipped_samples_by_cue": dict(skipped),
        "config": config,
    }
    write_json(output_dir / "provenance.json", provenance)
    return provenance


def _prediction_from_dict(payload: Dict) -> AffectPrediction:
    return AffectPrediction(**payload)


def evaluate(config: Dict) -> Dict:
    output_dir = Path(config["run"]["output_dir"])
    sample_rows = read_jsonl(output_dir / "samples.jsonl")
    intervention_rows = read_jsonl(output_dir / "interventions.jsonl")
    samples = {row["sample_id"]: AffectSample(**row) for row in sample_rows}
    eligible = [row for row in intervention_rows if row["eligible"]]
    evaluator = ResNetAffectModel(Path(config["model"]["checkpoint"]), role="evaluator")
    original_ids = list(samples)
    original_images = []
    for sample_id in original_ids:
        with Image.open(samples[sample_id].image_path) as source:
            original_images.append(source.convert("RGB"))
    original_predictions, original_features = evaluator.predict(original_images)
    original_by_id = {sample_id: (prediction, original_features[index]) for index, (sample_id, prediction) in enumerate(zip(original_ids, original_predictions))}

    twin_images = []
    for intervention in eligible:
        with Image.open(intervention["image_path"]) as source:
            twin_images.append(source.convert("RGB"))
    twin_predictions, twin_features = evaluator.predict(twin_images)

    if config["model"].get("enable_vlm", False):
        per_cue_limit = int(config["model"].get("vlm_pair_limit_per_cue", 0))
        selected_indices = []
        for cue in CueFamily:
            cue_indices = [
                index for index, row in enumerate(eligible)
                if row["cue_family"] == cue.value and not row.get("metadata", {}).get("is_control", False)
            ]
            selected_indices.extend(cue_indices[:per_cue_limit] if per_cue_limit > 0 else cue_indices)
        selected_indices = sorted(set(selected_indices))
        selected_original_ids = list(dict.fromkeys(eligible[index]["sample_id"] for index in selected_indices))
        selected_original_images = [original_images[original_ids.index(sample_id)] for sample_id in selected_original_ids]
        vlm = SmolVLMAdapter(
            config["model"]["vlm_model"],
            local_files_only=bool(config["model"].get("vlm_local_files_only", False)),
            cache_dir=config["model"].get("vlm_cache_dir"),
        )
        vlm_original_by_id = dict(zip(
            selected_original_ids,
            vlm.predict(
                selected_original_images,
                cache_keys=["original::{}".format(sample_id) for sample_id in selected_original_ids],
            ),
        ))
        selected_twin_predictions = vlm.predict(
            [twin_images[index] for index in selected_indices],
            cache_keys=["twin::{}".format(eligible[index]["intervention_id"]) for index in selected_indices],
        )
        vlm_twins_by_id = {
            eligible[index]["intervention_id"]: prediction
            for index, prediction in zip(selected_indices, selected_twin_predictions)
        }
    else:
        vlm_original_by_id, vlm_twins_by_id = {}, {}

    rows = []
    for index, (intervention, twin_prediction) in enumerate(zip(eligible, twin_predictions)):
        sample = samples[intervention["sample_id"]]
        original_prediction, original_feature = original_by_id[sample.sample_id]
        cosine = float(torch.nn.functional.cosine_similarity(original_feature[None], twin_features[index][None]).item())
        with Image.open(intervention["mask_path"]) as mask_file:
            mask_fraction = float(np.asarray(mask_file.convert("L"), dtype=np.float32).mean() / 255.0)
        metrics = pair_metrics(original_prediction, twin_prediction, sample.emotion, cosine, mask_fraction, intervention["expected_direction"])
        row = {
            "sample_id": sample.sample_id,
            "intervention_id": intervention["intervention_id"],
            "cue_family": intervention["cue_family"],
            "operation": intervention["operation"],
            "is_control": float(intervention.get("metadata", {}).get("is_control", False)),
            "au_annotation_backed": float(intervention.get("metadata", {}).get("au_annotation_backed", False)),
            "target_active_aus": ";".join(intervention.get("metadata", {}).get("target_active_aus", [])),
            "eligible": True,
            "source_emotion": sample.emotion,
            "original_prediction": original_prediction.predicted_emotion,
            "twin_prediction": twin_prediction.predicted_emotion,
            "original_confidence": original_prediction.confidence,
            "twin_confidence": twin_prediction.confidence,
            "original_correct": float(original_prediction.predicted_emotion == sample.emotion),
            "original_brier": float(sum((probability - float(label == sample.emotion)) ** 2 for label, probability in original_prediction.emotion_probabilities.items())),
            "original_nll": float(-math.log(max(1e-12, original_prediction.emotion_probabilities[sample.emotion]))),
            "original_valence_absolute_error": abs(original_prediction.valence - sample.valence) if sample.valence is not None else float("nan"),
            "original_arousal_absolute_error": abs(original_prediction.arousal - sample.arousal) if sample.arousal is not None else float("nan"),
            "original_valence": original_prediction.valence,
            "original_arousal": original_prediction.arousal,
            "twin_valence": twin_prediction.valence,
            "twin_arousal": twin_prediction.arousal,
            "image_path": intervention["image_path"],
            "mask_path": intervention["mask_path"],
            **metrics,
        }
        if intervention["intervention_id"] in vlm_twins_by_id:
            vo, vt = vlm_original_by_id[sample.sample_id], vlm_twins_by_id[intervention["intervention_id"]]
            from .metrics import entropy, token_jaccard
            vlm_valid = vo.raw.get("parse_status") == "valid_constrained" and vt.raw.get("parse_status") == "valid_constrained"
            row.update({
                "vlm_valid": float(vlm_valid),
                "vlm_original_parse_status": vo.raw.get("parse_status", ""),
                "vlm_twin_parse_status": vt.raw.get("parse_status", ""),
                "vlm_original_prediction": vo.predicted_emotion,
                "vlm_twin_prediction": vt.predicted_emotion,
                "vlm_original_caption": vo.caption,
                "vlm_twin_caption": vt.caption,
                "vlm_original_evidence_cue": vo.evidence_cue,
                "vlm_twin_evidence_cue": vt.evidence_cue,
                "vlm_cue_grounded": float(vt.evidence_cue == intervention["cue_family"]) if vlm_valid else float("nan"),
                "vlm_caption_jaccard": token_jaccard(vo.caption, vt.caption),
                "vlm_entropy_change": entropy(vt.emotion_probabilities) - entropy(vo.emotion_probabilities),
                "vlm_source_probability_drop": vo.emotion_probabilities[sample.emotion] - vt.emotion_probabilities[sample.emotion],
                "vlm_va_distance": float(math.hypot(vt.valence - vo.valence, vt.arousal - vo.arousal)),
            })
        rows.append(row)
    summary = aggregate(rows)
    summary.update({
        "dataset": config["dataset"]["name"],
        "n_samples": len(samples),
        "n_manifest_entries": len(intervention_rows),
        "n_skipped": len(intervention_rows) - len(eligible),
        "skips_by_reason": dict(Counter(row["skip_reason"] for row in intervention_rows if not row["eligible"])),
        "vlm_enabled": bool(config["model"].get("enable_vlm", False)),
        "vlm_pairs_evaluated": len(vlm_twins_by_id),
    })
    write_csv(output_dir / "pair_metrics.csv", rows)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "predictions.jsonl", rows)
    return summary


def doctor(config: Dict) -> Dict:
    assets = config["assets"]
    checks = {
        "dataset": Path(config["dataset"]["image_root"]).is_dir(),
        "ground_truth": Path(config["dataset"]["ground_truth_csv"]).is_file(),
        "checkpoint": Path(config["model"]["checkpoint"]).is_file(),
        "face_landmarker_model": Path(assets["face_landmarker"]).is_file(),
        "opencv_face_detector": Path(assets.get("face_detector_prototxt", "")).is_file() and Path(assets.get("face_detector_model", "")).is_file(),
        "tesseract": Path(assets["tesseract"]).is_file(),
    }
    if config["dataset"].get("audit_manifest"):
        checks["audit_manifest"] = Path(config["dataset"]["audit_manifest"]).is_file()
        checks["dataset"] = checks["audit_manifest"]
        checks["ground_truth"] = True
        if checks["audit_manifest"]:
            audit_samples = load_sample_manifest(Path(config["dataset"]["audit_manifest"]))
            missing_images = [sample.image_path for sample in audit_samples if not Path(sample.image_path).is_file()]
            checks["audit_images_present"] = not missing_images
            checks["audit_image_count"] = len(audit_samples)
            checks["missing_audit_image_count"] = len(missing_images)
            checks["missing_audit_images"] = missing_images[:20]
    try:
        import mediapipe  # noqa: F401
        checks["mediapipe"] = True
    except ImportError:
        checks["mediapipe"] = False
    try:
        import transformers  # noqa: F401
        checks["transformers"] = True
    except ImportError:
        checks["transformers"] = False
    try:
        import torchvision  # noqa: F401
        checks["torchvision_object_segmenter"] = True
    except ImportError:
        checks["torchvision_object_segmenter"] = False
    try:
        import scipy  # noqa: F401
        checks["scipy_panoptic_dependency"] = True
    except ImportError:
        checks["scipy_panoptic_dependency"] = False
    return checks
