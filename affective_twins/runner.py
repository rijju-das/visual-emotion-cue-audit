"""End-to-end generation and evaluation orchestration."""

import platform
import sys
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from .datasets import balanced_subset, deterministic_split, load_emotion6, load_sample_manifest
from .interventions import ColorIntervention, ContextIntervention, FaceActionRegionIntervention, TextIntervention
from .interventions.base import GeneratedTwin
from .io import read_jsonl, write_csv, write_json, write_jsonl
from .human import write_annotation_template
from .metrics import aggregate, pair_metrics
from .models import ResNetAffectModel, SmolVLMAdapter, train_independent_models
from .schema import AffectPrediction, AffectSample, CueFamily, EMOTIONS, Intervention


def load_samples(config: Dict) -> List[AffectSample]:
    dataset = config["dataset"]
    samples = load_emotion6(Path(dataset["image_root"]), Path(dataset["ground_truth_csv"]))
    audit_manifest = dataset.get("audit_manifest")
    if audit_manifest and Path(audit_manifest).is_file():
        audit_ids = {
            sample.sample_id for sample in load_sample_manifest(Path(audit_manifest))
            if sample.metadata.get("source_dataset") == "Emotion6"
        }
        samples = [sample for sample in samples if sample.sample_id not in audit_ids]
    return deterministic_split(samples, config["seed"])


def train(config: Dict) -> Dict[str, float]:
    samples = load_samples(config)
    metrics = train_independent_models(samples, Path(config["model"]["checkpoint"]), config["seed"])
    metrics["training_pool_size"] = len(samples)
    metrics["label_source"] = "human_distribution"
    return metrics


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


def _candidate_label(candidate) -> str:
    metadata = candidate.metadata
    mask = np.asarray(candidate.mask.convert("L")) > 0
    if mask.any():
        rows, columns = np.where(mask)
        horizontal = ["left", "centre", "right"][min(2, int(3 * columns.mean() / mask.shape[1]))]
        vertical = ["upper", "middle", "lower"][min(2, int(3 * rows.mean() / mask.shape[0]))]
        position = "{} {}".format(horizontal, vertical)
    else:
        position = "unknown position"
    if metadata.get("panoptic_label"):
        instance = metadata.get("panoptic_segment_id", metadata.get("proposal_index", ""))
        suffix = ", segment {}".format(instance) if instance != "" else ""
        return "{} ({}{})".format(metadata["panoptic_label"], position, suffix)
    if metadata.get("semantic_label"):
        return "{} ({})".format(metadata["semantic_label"], position)
    if metadata.get("au_region"):
        region = str(metadata["au_region"])
        component = "brows" if region.startswith("brow") else "eyes" if region.startswith("eye") else "mouth"
        return "face {} {}".format(metadata.get("face_index", 0), component)
    if metadata.get("ocr_tokens"):
        return "detected text: {}".format(" ".join(metadata["ocr_tokens"]))
    if metadata.get("inserted_text"):
        return "inserted affect word: {}".format(metadata["inserted_text"])
    if candidate.cue_family == CueFamily.CONTEXT:
        return "scene background and surrounding context"
    return str(metadata.get("subject_type", candidate.operation))


def _candidate_area(candidate) -> float:
    return float(np.asarray(candidate.mask.convert("L"), dtype=np.float32).mean() / 255.0)


_CONCEPT_ALIASES = {
    "person": {"person", "people", "man", "men", "woman", "women", "girl", "boy", "child", "children", "body", "shirt", "clothing", "clothes", "dress"},
    "background": {"background", "shadow", "shadows", "darkness"},
    "sky": {"sky", "cloud", "clouds"},
    "wall": {"wall", "walls"},
    "floor": {"floor", "ground", "pavement"},
    "water": {"water", "sea", "ocean", "lake", "river"},
    "cake": {"cake", "cakes", "cupcake", "cupcakes"},
    "television": {"tv", "television", "screen"},
    "building": {"building", "buildings", "house"},
    "tree": {"tree", "trees"},
    "flower": {"flower", "flowers"},
    "grass": {"grass", "lawn"},
    "dirt": {"dirt", "soil", "mud"},
    "boat": {"boat", "boats", "ship"},
    "vase": {"vase", "vases"},
    "toilet": {"toilet", "toilets"},
}
_NON_REGION_WORDS = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "white", "black", "grey", "gray",
    "bright", "brightness", "dim", "colour", "color", "lighting", "light", "emotion", "nothing", "none",
}


def _region_terms(text: str):
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
    raw = set(normalized.split()) - _NON_REGION_WORDS
    terms = set(raw)
    for canonical, aliases in _CONCEPT_ALIASES.items():
        if raw & aliases:
            terms.add(canonical)
    return terms


def _candidate_region_text(candidate) -> str:
    metadata = candidate.metadata
    values = [
        metadata.get("panoptic_label", ""),
        metadata.get("semantic_label", ""),
        " ".join(metadata.get("subject_labels", [])),
        metadata.get("subject_type", ""),
        " ".join(metadata.get("ocr_tokens", [])),
    ]
    return " ".join(str(value) for value in values if value)


def _ground_reported_evidence(
    vlm: SmolVLMAdapter,
    source: Image.Image,
    candidates,
    cue: str,
    evidence: str,
    sample_id: str,
):
    """Ground reported evidence conservatively; never substitute an unrelated salient region."""
    labels = [_candidate_label(candidate) for candidate in candidates]
    if cue == CueFamily.FACE.value:
        requested = SmolVLMAdapter._choice(evidence, ["brows", "eyes", "mouth"])
        if requested is None:
            return {"index": None, "status": "facial_component_not_explicit", "candidate_labels": labels}
        matches = []
        for index, candidate in enumerate(candidates):
            region = str(candidate.metadata.get("au_region", ""))
            component = "brows" if region.startswith("brow") else "eyes" if region.startswith("eye") else "mouth" if region.startswith("mouth") else ""
            if component == requested:
                matches.append(index)
        if len(matches) != 1:
            return {"index": None, "status": "facial_component_not_uniquely_localized", "candidate_labels": labels}
        return {
            "index": matches[0],
            "status": "exact_constrained_facial_component",
            "response": requested,
            "candidate_labels": labels,
            "matched_terms": [requested],
        }
    if cue == CueFamily.CONTEXT.value:
        return {
            "index": 0,
            "status": "full_reported_scene_context",
            "response": evidence,
            "candidate_labels": labels,
            "matched_terms": sorted(_region_terms(evidence)),
        }

    evidence_terms = _region_terms(evidence)
    scored = []
    for index, candidate in enumerate(candidates):
        candidate_terms = _region_terms(_candidate_region_text(candidate))
        overlap = evidence_terms & candidate_terms
        if overlap:
            scored.append((index, len(overlap), sorted(overlap)))
    if not scored:
        return {
            "index": None,
            "status": "reported_evidence_absent_from_candidate_labels",
            "candidate_labels": labels,
            "evidence_terms": sorted(evidence_terms),
        }
    best_score = max(item[1] for item in scored)
    best = [item for item in scored if item[1] == best_score]
    if len(best) == 1:
        index, _, overlap = best[0]
        return {
            "index": index,
            "status": "exact_lexical_region_match",
            "response": evidence,
            "candidate_labels": labels,
            "matched_terms": overlap,
        }

    ambiguous_indices = [item[0] for item in best]
    selection = vlm.select_evidence_region(
        source,
        cue,
        [labels[index] for index in ambiguous_indices],
        evidence=evidence,
        cache_key=sample_id,
    )
    if selection.get("index") is None:
        return {
            **selection,
            "status": "ambiguous_region_{}".format(selection.get("status", "invalid")),
            "candidate_labels": labels,
            "ambiguous_candidate_indices": ambiguous_indices,
        }
    selected = ambiguous_indices[int(selection["index"])]
    return {
        **selection,
        "index": selected,
        "status": "lexical_match_plus_order_invariant_vlm",
        "candidate_labels": labels,
        "ambiguous_candidate_indices": ambiguous_indices,
        "matched_terms": best[int(selection["index"])][2],
    }


def _report_condition_candidates(
    vlm: SmolVLMAdapter,
    locator: ResNetAffectModel,
    source: Image.Image,
    candidates,
    sample: AffectSample,
    report: AffectPrediction,
    include_control: bool,
):
    """Choose a reported-cue target or one explicitly unreported comparator."""
    if not candidates:
        return [], "no_candidates"
    candidates = _score_without_discarding(locator, source, candidates, sample.emotion)
    core_report_valid = report.raw.get("parse_status") == "valid_constrained"
    commitment_required = bool(report.raw.get("commitment_required", False))
    commitment_valid = report.raw.get("commitment_status") == "valid_commitment"
    valid_report = core_report_valid and (not commitment_required or commitment_valid)
    reported_cue = report.evidence_cue if valid_report else ""
    cue = candidates[0].cue_family.value
    shared = {
        "report_conditioned": True,
        "original_report_valid": valid_report,
        "original_reported_cue": reported_cue,
        "original_reported_evidence": report.evidence,
        "original_reported_emotion": report.predicted_emotion,
        "original_commitment_valid": commitment_valid,
        "original_declared_cue_role": report.cue_role,
        "original_expected_intervention_outcome": report.expected_intervention_outcome,
        "original_expected_emotion_after_intervention": report.expected_emotion_after_intervention,
        "original_declared_backup_cue": report.backup_cue,
        "original_declared_backup_evidence": report.backup_evidence,
    }
    if not valid_report or cue != reported_cue:
        is_declared_backup = (
            valid_report and commitment_valid and cue == report.backup_cue
        )
        if is_declared_backup and candidates[0].cue_family == CueFamily.TEXT:
            eligible = [
                index for index, candidate in enumerate(candidates)
                if candidate.operation == "remove_detected_text"
            ]
        elif is_declared_backup:
            # A declared backup is an exact-evidence target, so retain every
            # generated region for grounding instead of applying comparator
            # heuristics such as preferring annotation-active facial regions.
            eligible = list(range(len(candidates)))
        elif candidates[0].cue_family == CueFamily.TEXT:
            eligible = [index for index, candidate in enumerate(candidates) if candidate.operation == "insert_affect_conflict_text"]
        elif candidates[0].cue_family == CueFamily.FACE:
            eligible = [
                index for index, candidate in enumerate(candidates)
                if candidate.metadata.get("target_active_aus")
            ] or list(range(len(candidates)))
        else:
            eligible = [
                index for index, candidate in enumerate(candidates)
                if not candidate.metadata.get("is_control", False)
            ] or list(range(len(candidates)))
        if is_declared_backup:
            if not eligible:
                return [], "declared_backup_evidence_not_available"
            eligible_candidates = [candidates[index] for index in eligible]
            selection = _ground_reported_evidence(
                vlm,
                source,
                eligible_candidates,
                cue,
                report.backup_evidence,
                "{}::declared-backup".format(sample.sample_id),
            )
            if selection.get("index") is None:
                return [], "declared_backup_{}".format(
                    selection.get("status", "evidence_not_grounded")
                )
            selected = eligible[int(selection["index"])]
            candidate = candidates[selected]
            candidate.metadata.update(shared)
            candidate.metadata.update({
                "is_control": False,
                "report_match": False,
                "reported_evidence_region_match": False,
                "declared_backup_match": True,
                "declared_backup_evidence_region_match": True,
                "report_condition_role": "declared_backup_cue_target",
                "selection_model": "exact_declared_backup_evidence_grounding",
                "selected_candidate_label": _candidate_label(candidate),
                "backup_evidence_matched_terms": selection.get("matched_terms", []),
                "backup_region_selection_status": selection["status"],
                "backup_region_selection_response": selection["response"],
                "backup_region_candidates": selection["candidate_labels"],
            })
            return [candidate], "declared_backup_cue_target"
        selected = eligible[0]
        candidate = candidates[selected]
        candidate.metadata.update(shared)
        candidate.metadata.update({
            "is_control": False,
            "report_match": False,
            "reported_evidence_region_match": False,
            "declared_backup_match": False,
            "report_condition_role": "unreported_cue_comparator" if valid_report else "invalid_report_comparator",
            "selection_model": "deterministic_unreported_cue_comparator",
            "selected_candidate_label": _candidate_label(candidate),
        })
        return [candidate], "unreported_cue_comparator"

    if candidates[0].cue_family == CueFamily.TEXT:
        eligible = [index for index, candidate in enumerate(candidates) if candidate.operation == "remove_detected_text"]
        if not eligible:
            return [], "reported_text_not_groundable_by_ocr"
    else:
        eligible = list(range(len(candidates)))
    eligible_candidates = [candidates[index] for index in eligible]
    selection = _ground_reported_evidence(
        vlm,
        source,
        eligible_candidates,
        cue,
        report.evidence,
        sample.sample_id,
    )
    if selection.get("index") is None:
        return [], selection.get("status", "reported_evidence_not_grounded")
    selected = eligible[int(selection["index"])]
    target = candidates[selected]
    target_was_annotation_control = bool(target.metadata.get("is_control", False))
    target.metadata.update(shared)
    target.metadata.update({
        "is_control": False,
        "report_match": True,
        "report_condition_role": "reported_cue_target",
        "selection_model": "exact_reported_evidence_grounding",
        "selected_candidate_label": _candidate_label(target),
        "reported_evidence_region_match": True,
        "reported_evidence_matched_terms": selection.get("matched_terms", []),
        "reported_region_selection_status": selection["status"],
        "reported_region_selection_response": selection["response"],
        "reported_region_candidates": selection["candidate_labels"],
        "au_annotation_consistent": (
            not target_was_annotation_control
            if target.metadata.get("au_annotation_backed", False)
            else None
        ),
    })
    result = [target]
    if include_control:
        remaining = [index for index in eligible if index != selected]
        if remaining:
            target_area = _candidate_area(target)
            control_index = min(remaining, key=lambda index: abs(_candidate_area(candidates[index]) - target_area))
            control = candidates[control_index]
            control.operation = "matched_report_region_control"
            control.expected_direction = "control"
            control.metadata.update(shared)
            control.metadata.update({
                "is_control": True,
                "report_match": False,
                "reported_evidence_region_match": False,
                "report_condition_role": "same_cue_matched_region_control",
                "selection_model": "mask_area_matched_within_reported_cue_family",
                "control_for_candidate": selected,
                "control_matching": "nearest_mask_area_within_reported_cue_family",
                "target_mask_area_fraction": target_area,
                "selected_candidate_label": _candidate_label(control),
            })
            result.append(control)
    return result, "reported_cue_target"


def _compose_commitment_chain(
    source: Image.Image,
    primary: GeneratedTwin,
    backup: GeneratedTwin,
) -> GeneratedTwin:
    """Compose independently generated primary and declared-backup edits as pixel deltas."""
    source_array = np.asarray(source.convert("RGB"), dtype=np.float32)
    primary_array = np.asarray(primary.image.convert("RGB"), dtype=np.float32)
    backup_array = np.asarray(backup.image.convert("RGB"), dtype=np.float32)
    backup_alpha = np.asarray(backup.mask.convert("L"), dtype=np.float32)[..., None] / 255.0
    combined = np.clip(
        primary_array + backup_alpha * (backup_array - source_array),
        0,
        255,
    ).astype(np.uint8)
    union_mask = Image.fromarray(
        np.maximum(
            np.asarray(primary.mask.convert("L"), dtype=np.uint8),
            np.asarray(backup.mask.convert("L"), dtype=np.uint8),
        ),
        mode="L",
    )
    metadata = dict(primary.metadata)
    metadata.update({
        "is_control": False,
        "report_match": True,
        "report_condition_role": "primary_plus_declared_backup_chain",
        "selection_model": "precommitted_primary_then_declared_backup",
        "primary_cue_family": primary.cue_family.value,
        "primary_candidate_label": primary.metadata.get("selected_candidate_label", ""),
        "declared_backup_cue": backup.cue_family.value,
        "declared_backup_candidate_label": backup.metadata.get("selected_candidate_label", ""),
        "selected_candidate_label": "{} + {}".format(
            primary.metadata.get("selected_candidate_label", primary.cue_family.value),
            backup.metadata.get("selected_candidate_label", backup.cue_family.value),
        ),
        "chain_composition": "primary_image_plus_masked_backup_pixel_delta",
    })
    return GeneratedTwin(
        image=Image.fromarray(combined, mode="RGB"),
        mask=union_mask,
        cue_family=primary.cue_family,
        operation="sequential_primary_plus_declared_backup_ablation",
        target_region=union_mask.getbbox(),
        metadata=metadata,
    )


def generate(config: Dict) -> Dict:
    output_dir = Path(config["run"]["output_dir"])
    checkpoint = Path(config["model"]["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError("Missing model checkpoint. Run `affective-twins train` first.")
    samples = load_audit_samples(config)
    _require_sample_images(samples)
    locator = ResNetAffectModel(checkpoint, role="locator")
    report_conditioned = bool(config["run"].get("report_conditioned", False))
    report_vlm = None
    original_reports = {}
    if report_conditioned:
        if not config["model"].get("enable_vlm", False):
            raise ValueError("Report-conditioned generation requires model.enable_vlm=true")
        report_vlm = SmolVLMAdapter(
            config["model"]["vlm_model"],
            local_files_only=bool(config["model"].get("vlm_local_files_only", False)),
            cache_dir=config["model"].get("vlm_cache_dir"),
            allowed_cues=config["run"].get("enabled_cues"),
        )
        source_images = []
        for sample in samples:
            with Image.open(sample.image_path) as source_file:
                source_images.append(source_file.convert("RGB"))
        predictions = report_vlm.predict(
            source_images,
            cache_keys=["original::{}".format(sample.sample_id) for sample in samples],
            include_commitment=bool(config["run"].get("commitment_audit", False)),
        )
        original_reports = dict(zip((sample.sample_id for sample in samples), predictions))
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
            segmentation_device=config["run"].get("segmentation_device", "auto"),
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
    enabled_cues = set(config["run"].get("enabled_cues", [cue.value for cue in CueFamily]))
    generators = [generator for generator in generators if generator.cue_family.value in enabled_cues]
    intervention_rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "samples.jsonl", [sample.to_dict() for sample in samples])
    if report_conditioned:
        write_jsonl(output_dir / "original_reports.jsonl", [
            {
                "sample_id": sample.sample_id,
                "reporting_model": config["model"]["vlm_model"],
                "valid_report": original_reports[sample.sample_id].raw.get("parse_status") == "valid_constrained",
                "valid_commitment": original_reports[sample.sample_id].raw.get("commitment_status") == "valid_commitment",
                **original_reports[sample.sample_id].to_dict(),
            }
            for sample in samples
        ])
    for sample in samples:
        with Image.open(sample.image_path) as source_file:
            source = source_file.convert("RGB")
        context_foreground_mask = None
        report = original_reports.get(sample.sample_id)
        primary_candidate = None
        backup_candidate = None
        for generator in generators:
            allowed_cues = sample.metadata.get("allowed_cues")
            if allowed_cues and generator.cue_family.value not in allowed_cues:
                is_primary = bool(report and report.evidence_cue == generator.cue_family.value)
                is_backup = bool(report and report.backup_cue == generator.cue_family.value)
                if report_conditioned and report and (is_primary or is_backup):
                    intervention_rows.append(Intervention(
                        intervention_id="{}--{}--skipped".format(sample.sample_id, generator.cue_family.value),
                        sample_id=sample.sample_id,
                        cue_family=generator.cue_family,
                        operation="skipped",
                        image_path="",
                        mask_path="",
                        eligible=False,
                        skip_reason=(
                            "reported_cue_disallowed_for_sample"
                            if is_primary else "declared_backup_cue_disallowed_for_sample"
                        ),
                        metadata={
                            "report_conditioned": True,
                            "original_reported_cue": report.evidence_cue,
                            "original_reported_evidence": report.evidence,
                            "original_declared_backup_cue": report.backup_cue,
                            "original_declared_backup_evidence": report.backup_evidence,
                            "report_condition_role": (
                                "reported_cue_ungroundable"
                                if is_primary else "declared_backup_cue_ungroundable"
                            ),
                        },
                    ).to_dict())
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
            if report_conditioned:
                candidates, selection_status = _report_condition_candidates(
                    report_vlm,
                    locator,
                    source,
                    candidates,
                    sample,
                    report,
                    include_control=bool(config["run"].get("matched_controls", False)),
                )
            elif generator.cue_family == CueFamily.COLOR:
                candidates = _score_without_discarding(locator, source, candidates, sample.emotion)
                selection_status = "legacy_all_candidates"
            elif generator.cue_family == CueFamily.FACE:
                candidates = _retain_face_regions(
                    locator,
                    source,
                    candidates,
                    sample.emotion,
                    include_control=bool(config["run"].get("matched_controls", False)),
                )
                selection_status = "legacy_au_selection"
            else:
                selection_status = "legacy_predefined_intervention"
            if not candidates:
                reason = (
                    selection_status
                    if report_conditioned and selection_status not in {"no_candidates", "unreported_cue_comparator"}
                    else getattr(generator, "unavailable_reason", "") or "no_eligible_region_detected"
                )
                intervention_rows.append(Intervention(
                    intervention_id="{}--{}--skipped".format(sample.sample_id, generator.cue_family.value),
                    sample_id=sample.sample_id,
                    cue_family=generator.cue_family,
                    operation="skipped",
                    image_path="",
                    mask_path="",
                    eligible=False,
                    skip_reason=reason,
                    metadata={
                        "report_conditioned": report_conditioned,
                        "original_reported_cue": report.evidence_cue if report else "",
                        "original_reported_evidence": report.evidence if report else "",
                        "report_condition_role": (
                            "reported_cue_ungroundable"
                            if report and report.evidence_cue == generator.cue_family.value
                            else "declared_backup_cue_ungroundable"
                            if selection_status.startswith("declared_backup_")
                            else "cue_unavailable"
                        ),
                    },
                ).to_dict())
                continue
            for index, candidate in enumerate(candidates):
                role = candidate.metadata.get("report_condition_role")
                if role == "reported_cue_target":
                    primary_candidate = candidate
                elif role == "declared_backup_cue_target":
                    backup_candidate = candidate
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
        if (
            bool(config["run"].get("sequential_backup_intervention", False))
            and primary_candidate is not None
            and backup_candidate is not None
        ):
            chain = _compose_commitment_chain(source, primary_candidate, backup_candidate)
            identifier = "{}--commitment_chain--00".format(sample.sample_id)
            image_path = output_dir / "twins" / "commitment_chain" / "{}.png".format(identifier)
            mask_path = output_dir / "masks" / "commitment_chain" / "{}.png".format(identifier)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            chain.image.save(image_path)
            chain.mask.save(mask_path)
            intervention_rows.append(Intervention(
                intervention_id=identifier,
                sample_id=sample.sample_id,
                cue_family=chain.cue_family,
                operation=chain.operation,
                image_path=str(image_path.resolve()),
                mask_path=str(mask_path.resolve()),
                target_region=chain.target_region,
                expected_direction=chain.expected_direction,
                metadata=chain.metadata,
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
        "framework_version": "0.6.0",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "seed": config["seed"],
        "sample_count": len(samples),
        "report_conditioned": report_conditioned,
        "valid_original_reports": sum(
            prediction.raw.get("parse_status") == "valid_constrained"
            for prediction in original_reports.values()
        ),
        "valid_original_commitments": sum(
            prediction.raw.get("commitment_status") == "valid_commitment"
            for prediction in original_reports.values()
        ),
        "reported_cue_targets": sum(
            row["eligible"] and row.get("metadata", {}).get("report_condition_role") == "reported_cue_target"
            for row in intervention_rows
        ),
        "unreported_cue_comparators": sum(
            row["eligible"] and row.get("metadata", {}).get("report_condition_role") == "unreported_cue_comparator"
            for row in intervention_rows
        ),
        "declared_backup_cue_targets": sum(
            row["eligible"] and row.get("metadata", {}).get("report_condition_role") == "declared_backup_cue_target"
            for row in intervention_rows
        ),
        "primary_plus_backup_chains": sum(
            row["eligible"] and row.get("metadata", {}).get("report_condition_role") == "primary_plus_declared_backup_chain"
            for row in intervention_rows
        ),
        "eligible_pairs_by_cue": dict(counts),
        "matched_controls_by_cue": dict(control_counts),
        "skipped_samples_by_cue": dict(skipped),
        "config": config,
    }
    write_json(output_dir / "provenance.json", provenance)
    return provenance


def _prediction_from_dict(payload: Dict) -> AffectPrediction:
    return AffectPrediction(**payload)


def _observed_commitment_outcome(
    original: AffectPrediction,
    twin: AffectPrediction,
    tolerance: float = 0.05,
) -> str:
    if twin.predicted_emotion != original.predicted_emotion:
        return "label_change"
    confidence_change = twin.confidence - original.confidence
    if confidence_change < -tolerance:
        return "confidence_decrease_same_label"
    if confidence_change > tolerance:
        return "confidence_increase_same_label"
    return "no_material_change"


def _human_target_distribution(sample: AffectSample) -> Dict[str, float]:
    """Renormalize Emotion6 annotations over the evaluator's six modeled classes."""
    values = {label: float(sample.emotion_distribution.get(label, 0.0)) for label in EMOTIONS}
    total = sum(values.values())
    if total <= 0:
        return {label: float(label == sample.emotion) for label in EMOTIONS}
    return {label: value / total for label, value in values.items()}


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
        report_conditioned = bool(config["run"].get("report_conditioned", False))
        selected_indices = []
        for cue in CueFamily:
            cue_indices = [
                index for index, row in enumerate(eligible)
                if row["cue_family"] == cue.value
                and (report_conditioned or not row.get("metadata", {}).get("is_control", False))
            ]
            selected_indices.extend(cue_indices[:per_cue_limit] if per_cue_limit > 0 else cue_indices)
        selected_indices = sorted(set(selected_indices))
        selected_original_ids = list(dict.fromkeys(eligible[index]["sample_id"] for index in selected_indices))
        vlm = SmolVLMAdapter(
            config["model"]["vlm_model"],
            local_files_only=bool(config["model"].get("vlm_local_files_only", False)),
            cache_dir=config["model"].get("vlm_cache_dir"),
            allowed_cues=config["run"].get("enabled_cues"),
        )
        # Reuse the exact pre-intervention reports that conditioned generation.
        # This prevents a regenerated answer from silently changing the audit target.
        vlm_original_by_id = {}
        original_report_path = output_dir / "original_reports.jsonl"
        if report_conditioned and original_report_path.is_file():
            for saved in read_jsonl(original_report_path):
                if saved["sample_id"] not in selected_original_ids:
                    continue
                vlm_original_by_id[saved["sample_id"]] = AffectPrediction(
                    emotion_probabilities=saved["emotion_probabilities"],
                    valence=saved["valence"],
                    arousal=saved["arousal"],
                    confidence=saved["confidence"],
                    predicted_emotion=saved["predicted_emotion"],
                    caption=saved.get("caption", ""),
                    evidence=saved.get("evidence", ""),
                    evidence_cue=saved.get("evidence_cue", ""),
                    cue_role=saved.get("cue_role", ""),
                    expected_intervention_outcome=saved.get("expected_intervention_outcome", ""),
                    expected_emotion_after_intervention=saved.get("expected_emotion_after_intervention", ""),
                    backup_cue=saved.get("backup_cue", ""),
                    backup_evidence=saved.get("backup_evidence", ""),
                    raw=saved.get("raw", {}),
                )
        missing_original_ids = [
            sample_id for sample_id in selected_original_ids
            if sample_id not in vlm_original_by_id
        ]
        if missing_original_ids:
            missing_original_images = [
                original_images[original_ids.index(sample_id)] for sample_id in missing_original_ids
            ]
            vlm_original_by_id.update(dict(zip(
                missing_original_ids,
                vlm.predict(
                    missing_original_images,
                    cache_keys=["original::{}".format(sample_id) for sample_id in missing_original_ids],
                    include_commitment=bool(config["run"].get("commitment_audit", False)),
                ),
            )))
        selected_twin_predictions = vlm.predict(
            [twin_images[index] for index in selected_indices],
            cache_keys=["twin::{}".format(eligible[index]["intervention_id"]) for index in selected_indices],
            include_commitment=False,
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
        human_target = _human_target_distribution(sample)
        nominal_emotion = sample.nominal_emotion or sample.metadata.get("folder_label", "")
        row = {
            "sample_id": sample.sample_id,
            "intervention_id": intervention["intervention_id"],
            "cue_family": intervention["cue_family"],
            "operation": intervention["operation"],
            "is_control": float(intervention.get("metadata", {}).get("is_control", False)),
            "report_conditioned": float(intervention.get("metadata", {}).get("report_conditioned", False)),
            "report_match": float(intervention.get("metadata", {}).get("report_match", False)),
            "report_condition_role": intervention.get("metadata", {}).get("report_condition_role", "legacy_unconditioned"),
            "original_reported_cue": intervention.get("metadata", {}).get("original_reported_cue", ""),
            "original_reported_evidence": intervention.get("metadata", {}).get("original_reported_evidence", ""),
            "selected_candidate_label": intervention.get("metadata", {}).get("selected_candidate_label", ""),
            "reported_evidence_region_match": float(intervention.get("metadata", {}).get("reported_evidence_region_match", False)),
            "reported_region_selection_status": intervention.get("metadata", {}).get("reported_region_selection_status", ""),
            "reported_evidence_matched_terms": ";".join(intervention.get("metadata", {}).get("reported_evidence_matched_terms", [])),
            "declared_backup_match": float(intervention.get("metadata", {}).get("declared_backup_match", False)),
            "declared_backup_evidence_region_match": float(intervention.get("metadata", {}).get("declared_backup_evidence_region_match", False)),
            "backup_region_selection_status": intervention.get("metadata", {}).get("backup_region_selection_status", ""),
            "original_commitment_valid": float(intervention.get("metadata", {}).get("original_commitment_valid", False)),
            "original_declared_cue_role": intervention.get("metadata", {}).get("original_declared_cue_role", ""),
            "original_expected_intervention_outcome": intervention.get("metadata", {}).get("original_expected_intervention_outcome", ""),
            "original_expected_emotion_after_intervention": intervention.get("metadata", {}).get("original_expected_emotion_after_intervention", ""),
            "original_declared_backup_cue": intervention.get("metadata", {}).get("original_declared_backup_cue", ""),
            "original_declared_backup_evidence": intervention.get("metadata", {}).get("original_declared_backup_evidence", ""),
            "primary_cue_family": intervention.get("metadata", {}).get("primary_cue_family", ""),
            "declared_backup_cue": intervention.get("metadata", {}).get("declared_backup_cue", ""),
            "primary_candidate_label": intervention.get("metadata", {}).get("primary_candidate_label", ""),
            "declared_backup_candidate_label": intervention.get("metadata", {}).get("declared_backup_candidate_label", ""),
            "au_annotation_backed": float(intervention.get("metadata", {}).get("au_annotation_backed", False)),
            "au_annotation_consistent": intervention.get("metadata", {}).get("au_annotation_consistent"),
            "target_active_aus": ";".join(intervention.get("metadata", {}).get("target_active_aus", [])),
            "eligible": True,
            "source_emotion": sample.emotion,
            "label_source": sample.metadata.get("label_source", "dataset_annotation"),
            "human_plurality_emotion": sample.human_plurality_emotion or sample.emotion,
            "human_plurality_probability": sample.human_plurality_probability,
            "folder_label": nominal_emotion,
            "folder_human_agreement": float(bool(nominal_emotion) and nominal_emotion == sample.emotion),
            "original_prediction": original_prediction.predicted_emotion,
            "twin_prediction": twin_prediction.predicted_emotion,
            "original_confidence": original_prediction.confidence,
            "twin_confidence": twin_prediction.confidence,
            "original_correct": float(original_prediction.predicted_emotion == sample.emotion),
            "original_folder_correct": (
                float(original_prediction.predicted_emotion == nominal_emotion)
                if nominal_emotion else float("nan")
            ),
            "original_brier": float(sum(
                (probability - human_target[label]) ** 2
                for label, probability in original_prediction.emotion_probabilities.items()
            )),
            "original_nll": float(-sum(
                human_target[label] * math.log(max(1e-12, probability))
                for label, probability in original_prediction.emotion_probabilities.items()
            )),
            "original_hard_nll": float(
                -math.log(max(1e-12, original_prediction.emotion_probabilities[sample.emotion]))
            ),
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
            observed_outcome = _observed_commitment_outcome(vo, vt)
            commitment_valid = vo.raw.get("commitment_status") == "valid_commitment"
            expected_outcome_match = bool(
                commitment_valid
                and observed_outcome == vo.expected_intervention_outcome
            )
            expected_emotion_match = bool(
                commitment_valid
                and vt.predicted_emotion == vo.expected_emotion_after_intervention
            )
            substituted = vt.evidence_cue != vo.evidence_cue
            backup_declared = vo.backup_cue not in {"", "none"}
            backup_match = backup_declared and vt.evidence_cue == vo.backup_cue
            role_expected_outcome = {
                "essential": "label_change",
                "supportive": "confidence_decrease_same_label",
                "incidental": "no_material_change",
            }.get(vo.cue_role, "")
            role_outcome_coherent = bool(role_expected_outcome) and (
                role_expected_outcome == vo.expected_intervention_outcome
            )
            outcome_emotion_coherent = (
                vo.expected_emotion_after_intervention != vo.predicted_emotion
                if vo.expected_intervention_outcome == "label_change"
                else vo.expected_emotion_after_intervention == vo.predicted_emotion
            )
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
                "vlm_original_report_matches_intervention": float(vo.evidence_cue == intervention["cue_family"]) if vlm_valid else float("nan"),
                "vlm_twin_reports_intervened_cue": float(vt.evidence_cue == intervention["cue_family"]) if vlm_valid else float("nan"),
                "vlm_reported_cue_retained": float(vt.evidence_cue == vo.evidence_cue) if vlm_valid else float("nan"),
                "vlm_caption_jaccard": token_jaccard(vo.caption, vt.caption),
                "vlm_entropy_change": entropy(vt.emotion_probabilities) - entropy(vo.emotion_probabilities),
                "vlm_source_probability_drop": vo.emotion_probabilities[sample.emotion] - vt.emotion_probabilities[sample.emotion],
                "vlm_original_class_probability_drop": (
                    vo.emotion_probabilities[vo.predicted_emotion]
                    - vt.emotion_probabilities[vo.predicted_emotion]
                ),
                "vlm_original_prediction_flip": float(vo.predicted_emotion != vt.predicted_emotion),
                "vlm_confidence_change": vt.confidence - vo.confidence,
                "vlm_va_distance": float(math.hypot(vt.valence - vo.valence, vt.arousal - vo.arousal)),
                "vlm_commitment_valid": float(commitment_valid),
                "vlm_declared_cue_role": vo.cue_role,
                "vlm_expected_intervention_outcome": vo.expected_intervention_outcome,
                "vlm_expected_emotion_after_intervention": vo.expected_emotion_after_intervention,
                "vlm_declared_backup_cue": vo.backup_cue,
                "vlm_declared_backup_evidence": vo.backup_evidence,
                "vlm_observed_intervention_outcome": observed_outcome,
                "vlm_outcome_type_commitment_match": float(expected_outcome_match),
                "vlm_expected_emotion_match": float(expected_emotion_match),
                "vlm_counterfactual_commitment_kept": float(
                    expected_outcome_match and expected_emotion_match
                ),
                "vlm_role_outcome_declaration_coherent": float(
                    role_outcome_coherent
                ),
                "vlm_outcome_emotion_declaration_coherent": float(outcome_emotion_coherent),
                "vlm_commitment_declaration_coherent": float(
                    role_outcome_coherent and outcome_emotion_coherent
                ),
                "vlm_cue_substitution": float(substituted),
                "vlm_declared_backup_cue_activated": float(backup_match),
                "vlm_post_edit_rerationalization": float(
                    substituted and not backup_match
                ),
                "vlm_forecast_consistent_stability": float(
                    commitment_valid
                    and vo.expected_intervention_outcome != "label_change"
                    and vt.predicted_emotion == vo.predicted_emotion
                    and (not substituted or backup_match)
                ),
            })
        rows.append(row)
    enabled_cues = config["run"].get("enabled_cues", [cue.value for cue in CueFamily])
    summary = aggregate(rows, expected_cues=enabled_cues)
    summary.update({
        "dataset": config["dataset"]["name"],
        "n_samples": len(samples),
        "n_manifest_entries": len(intervention_rows),
        "n_skipped": len(intervention_rows) - len(eligible),
        "skips_by_reason": dict(Counter(row["skip_reason"] for row in intervention_rows if not row["eligible"])),
        "vlm_enabled": bool(config["model"].get("enable_vlm", False)),
        "vlm_model": config["model"].get("vlm_model", ""),
        "vlm_pairs_evaluated": len(vlm_twins_by_id),
        "report_conditioned": bool(config["run"].get("report_conditioned", False)),
        "commitment_audit": bool(config["run"].get("commitment_audit", False)),
        "audited_cue_families": list(enabled_cues),
        "reported_cue_target_coverage": (
            len({
                row["sample_id"] for row in rows
                if row.get("report_condition_role") == "reported_cue_target"
            }) / len(samples)
            if samples else 0.0
        ),
    })
    summary["reported_cue_audit_failure_rate"] = 1.0 - summary["reported_cue_target_coverage"]
    original_report_path = output_dir / "original_reports.jsonl"
    if original_report_path.is_file():
        saved_reports = read_jsonl(original_report_path)
        summary["original_report_valid_rate"] = (
            float(np.mean([row.get("valid_report", False) for row in saved_reports]))
            if saved_reports else 0.0
        )
        valid_report_count = sum(row.get("valid_report", False) for row in saved_reports)
        valid_commitments = [row for row in saved_reports if row.get("valid_commitment", False)]
        valid_commitment_count = len(valid_commitments)
        grounded_sample_count = len({
            row["sample_id"] for row in rows
            if row.get("report_condition_role") == "reported_cue_target"
            and row.get("reported_evidence_region_match") == 1.0
        })
        summary["valid_report_exact_grounding_rate"] = (
            grounded_sample_count / valid_report_count if valid_report_count else 0.0
        )
        summary["original_commitment_valid_rate"] = (
            valid_commitment_count / len(saved_reports) if saved_reports else 0.0
        )
        summary["valid_commitment_primary_target_rate"] = (
            grounded_sample_count / valid_commitment_count if valid_commitment_count else 0.0
        )
        declared_backups = [
            row for row in valid_commitments
            if row.get("backup_cue") not in {"", "none"}
        ]
        grounded_backup_count = len({
            row["sample_id"] for row in rows
            if row.get("report_condition_role") == "declared_backup_cue_target"
            and row.get("declared_backup_evidence_region_match") == 1.0
        })
        summary["declared_backup_grounding_rate"] = (
            grounded_backup_count / len(declared_backups) if declared_backups else float("nan")
        )
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
