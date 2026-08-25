"""Pair-level and aggregate counterfactual affect metrics."""

import math
from typing import Dict, Iterable, List

import numpy as np

from .schema import AffectPrediction, EMOTIONS


def bootstrap_mean_ci(values, seed: int = 42, replicates: int = 2000):
    values = np.asarray(list(values), dtype=np.float64)
    if not len(values):
        return [float("nan"), float("nan")]
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return [float(low), float(high)]


def entropy(probabilities: Dict[str, float]) -> float:
    values = np.asarray([probabilities[label] for label in EMOTIONS], dtype=np.float64)
    return float(-(values * np.log(np.clip(values, 1e-12, 1.0))).sum())


def js_divergence(first: Dict[str, float], second: Dict[str, float]) -> float:
    p = np.asarray([first[label] for label in EMOTIONS], dtype=np.float64)
    q = np.asarray([second[label] for label in EMOTIONS], dtype=np.float64)
    midpoint = 0.5 * (p + q)
    kl_p = (p * np.log(np.clip(p, 1e-12, 1.0) / np.clip(midpoint, 1e-12, 1.0))).sum()
    kl_q = (q * np.log(np.clip(q, 1e-12, 1.0) / np.clip(midpoint, 1e-12, 1.0))).sum()
    return float(0.5 * (kl_p + kl_q) / math.log(2.0))


def token_jaccard(first: str, second: str) -> float:
    a, b = set(first.lower().split()), set(second.lower().split())
    return float(len(a & b) / len(a | b)) if a or b else 1.0


def pair_metrics(original: AffectPrediction, twin: AffectPrediction, source_emotion: str, feature_cosine: float, mask_fraction: float, expected_direction: str) -> Dict[str, float]:
    source_drop = original.emotion_probabilities[source_emotion] - twin.emotion_probabilities[source_emotion]
    entropy_change = entropy(twin.emotion_probabilities) - entropy(original.emotion_probabilities)
    if expected_direction.startswith("increase_uncertainty"):
        directional_success = float(entropy_change > 0)
    else:
        directional_success = float(source_drop > 0)
    return {
        "source_probability_drop": source_drop,
        "emotion_js_divergence": js_divergence(original.emotion_probabilities, twin.emotion_probabilities),
        "valence_change": twin.valence - original.valence,
        "arousal_change": twin.arousal - original.arousal,
        "va_distance": math.hypot(twin.valence - original.valence, twin.arousal - original.arousal),
        "entropy_change": entropy_change,
        "prediction_flip": float(original.predicted_emotion != twin.predicted_emotion),
        "directional_success": directional_success,
        "feature_cosine": feature_cosine,
        "mask_fraction": mask_fraction,
        "caption_jaccard": token_jaccard(original.caption, twin.caption),
    }


def expected_calibration_error(rows: Iterable[Dict], bins: int = 10) -> float:
    rows = list(rows)
    if not rows:
        return float("nan")
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        group = [row for row in rows if low <= row["original_confidence"] <= high and (index == bins - 1 or row["original_confidence"] < high)]
        if group:
            accuracy = np.mean([row["original_correct"] for row in group])
            confidence = np.mean([row["original_confidence"] for row in group])
            result += len(group) / len(rows) * abs(accuracy - confidence)
    return float(result)


def aggregate(rows: List[Dict], expected_cues=None) -> Dict:
    expected_cues = list(expected_cues or [
        "color_lighting", "facial_action_region", "scene_context", "embedded_text"
    ])
    all_eligible = [row for row in rows if row.get("eligible", True)]
    chain_rows = [
        row for row in all_eligible
        if row.get("report_condition_role") == "primary_plus_declared_backup_chain"
    ]
    # Sequential primary+backup twins answer a separate commitment question.
    # Exclude them from the conventional one-cue aggregate so that they do not
    # double-count samples or inflate cue coverage and sensitivity estimates.
    all_eligible = [
        row for row in all_eligible
        if row.get("report_condition_role") != "primary_plus_declared_backup_chain"
    ]
    controls = [row for row in all_eligible if row.get("is_control", 0.0) == 1.0]
    eligible = [row for row in all_eligible if row.get("is_control", 0.0) != 1.0]
    if not eligible:
        return {"n_pairs": 0}
    mean = lambda key: float(np.mean([row[key] for row in eligible]))
    by_cue = {}
    for cue in sorted({row["cue_family"] for row in eligible}):
        group = [row for row in eligible if row["cue_family"] == cue]
        by_cue[cue] = {
            "n": len(group),
            "directional_success_rate": float(np.mean([row["directional_success"] for row in group])),
            "directional_success_ci95": bootstrap_mean_ci([row["directional_success"] for row in group]),
            "source_probability_drop_mean": float(np.mean([row["source_probability_drop"] for row in group])),
            "source_probability_drop_ci95": bootstrap_mean_ci([row["source_probability_drop"] for row in group]),
            "va_distance_mean": float(np.mean([row["va_distance"] for row in group])),
            "va_distance_ci95": bootstrap_mean_ci([row["va_distance"] for row in group]),
            "feature_cosine_mean": float(np.mean([row["feature_cosine"] for row in group])),
            "feature_cosine_ci95": bootstrap_mean_ci([row["feature_cosine"] for row in group]),
        }
    unique_originals = list({row["sample_id"]: row for row in eligible}.values())
    ece = expected_calibration_error(unique_originals)
    cue_coverage = len(by_cue) / len(expected_cues) if expected_cues else 0.0
    response = mean("directional_success")
    preservation = float(np.clip((mean("feature_cosine") + 1.0) / 2.0, 0.0, 1.0))
    uncertainty_rows = [row for row in eligible if row["operation"] == "insert_affect_conflict_text"]
    text_enabled = "embedded_text" in expected_cues
    uncertainty = (
        float(np.mean([row["entropy_change"] > 0 for row in uncertainty_rows]))
        if uncertainty_rows else float("nan") if not text_enabled else 0.0
    )
    calibration = float(np.clip(1.0 - ece, 0.0, 1.0))
    vlm_rows = [row for row in eligible if row.get("vlm_valid") == 1.0]
    grounding = float(np.mean([row["vlm_cue_grounded"] for row in vlm_rows])) if vlm_rows else None
    grounding_or_coverage = grounding if grounding is not None else cue_coverage
    cause_components = [response, preservation, grounding_or_coverage, calibration]
    if text_enabled:
        cause_components.append(uncertainty)
    cause = float(np.mean(cause_components))
    paired_advantages = []
    paired_by_cue = {}
    for cue in sorted({row["cue_family"] for row in controls}):
        cue_advantages = []
        for target in [row for row in eligible if row["cue_family"] == cue]:
            matched = [row for row in controls if row["sample_id"] == target["sample_id"] and row["cue_family"] == cue]
            if matched:
                advantage = target["source_probability_drop"] - float(np.mean([row["source_probability_drop"] for row in matched]))
                cue_advantages.append(advantage)
                paired_advantages.append(advantage)
        if cue_advantages:
            paired_by_cue[cue] = {
                "n": len(cue_advantages),
                "target_minus_control_drop_mean": float(np.mean(cue_advantages)),
                "target_minus_control_drop_ci95": bootstrap_mean_ci(cue_advantages),
                "positive_rate": float(np.mean(np.asarray(cue_advantages) > 0)),
            }
    folder_rows = [row for row in unique_originals if not math.isnan(row.get("original_folder_correct", float("nan")))]
    cause_note = "Unvalidated diagnostic composite: response, content preservation, cue grounding (or coverage when VLM is disabled), calibration"
    if text_enabled:
        cause_note += ", conflict uncertainty"
    result = {"n_pairs": len(eligible), "n_control_pairs": len(controls), "cue_coverage": cue_coverage, "directional_success_rate": response, "emotion_js_divergence_mean": mean("emotion_js_divergence"), "va_distance_mean": mean("va_distance"), "feature_cosine_mean": mean("feature_cosine"), "conflict_uncertainty_success_rate": uncertainty, "original_ece": ece, "original_brier_mean": float(np.mean([row["original_brier"] for row in unique_originals])), "original_nll_mean": float(np.mean([row["original_nll"] for row in unique_originals])), "original_human_plurality_accuracy": float(np.mean([row["original_correct"] for row in unique_originals])), "original_folder_accuracy": float(np.mean([row["original_folder_correct"] for row in folder_rows])) if folder_rows else float("nan"), "folder_human_agreement_rate": float(np.mean([row["folder_human_agreement"] for row in folder_rows])) if folder_rows else float("nan"), "original_va_mae": float(np.mean([0.5 * (row["original_valence_absolute_error"] + row["original_arousal_absolute_error"]) for row in unique_originals if not math.isnan(row["original_valence_absolute_error"])])), "cause_diagnostic_score": cause, "cause_note": cause_note + ".", "by_cue": by_cue, "matched_control_analysis": paired_by_cue}
    if paired_advantages:
        result["target_minus_control_drop_mean"] = float(np.mean(paired_advantages))
        result["target_minus_control_drop_ci95"] = bootstrap_mean_ci(paired_advantages)
    if grounding is not None:
        result["vlm_cue_grounding_accuracy"] = grounding
        result["vlm_cue_grounding_ci95"] = bootstrap_mean_ci([row["vlm_cue_grounded"] for row in vlm_rows])
        result["vlm_caption_stability_mean"] = float(np.mean([row["vlm_caption_jaccard"] for row in vlm_rows]))
        result["vlm_valid_pairs"] = len(vlm_rows)
        attempted = [row for row in eligible if "vlm_valid" in row]
        result["vlm_valid_pair_rate"] = float(len(vlm_rows) / len(attempted)) if attempted else 0.0
    reported_targets = [
        row for row in eligible
        if row.get("report_condition_role") == "reported_cue_target"
        and row.get("vlm_valid") == 1.0
    ]
    unreported_comparators = [
        row for row in eligible
        if row.get("report_condition_role") == "unreported_cue_comparator"
        and row.get("vlm_valid") == 1.0
    ]
    if reported_targets:
        target_drops = [row["vlm_original_class_probability_drop"] for row in reported_targets]
        target_flips = [row["vlm_original_prediction_flip"] for row in reported_targets]
        target_entropy = [row["vlm_entropy_change"] for row in reported_targets]
        comparator_differences = []
        for target in reported_targets:
            matched = [
                row for row in unreported_comparators
                if row["sample_id"] == target["sample_id"]
            ]
            if matched:
                comparator_differences.append(
                    target["vlm_original_class_probability_drop"]
                    - float(np.mean([row["vlm_original_class_probability_drop"] for row in matched]))
                )
        same_cue_control_differences = []
        for target in reported_targets:
            matched = [
                row for row in controls
                if row["sample_id"] == target["sample_id"]
                and row["cue_family"] == target["cue_family"]
                and row.get("report_condition_role") == "same_cue_matched_region_control"
                and row.get("vlm_valid") == 1.0
            ]
            if matched:
                same_cue_control_differences.append(
                    target["vlm_original_class_probability_drop"]
                    - float(np.mean([row["vlm_original_class_probability_drop"] for row in matched]))
                )
        by_reported_cue = {}
        for cue in sorted({row["cue_family"] for row in reported_targets}):
            group = [row for row in reported_targets if row["cue_family"] == cue]
            drops = [row["vlm_original_class_probability_drop"] for row in group]
            by_reported_cue[cue] = {
                "n": len(group),
                "original_class_probability_drop_mean": float(np.mean(drops)),
                "original_class_probability_drop_ci95": bootstrap_mean_ci(drops),
                "prediction_flip_rate": float(np.mean([row["vlm_original_prediction_flip"] for row in group])),
                "entropy_increase_rate": float(np.mean([row["vlm_entropy_change"] > 0 for row in group])),
                "reported_cue_retention_rate": float(np.mean([row["vlm_reported_cue_retained"] for row in group])),
            }
        result["reported_cue_behavioral_sensitivity"] = {
            "n_reported_cue_targets": len(reported_targets),
            "n_unreported_cue_comparators": len(unreported_comparators),
            "original_class_probability_drop_mean": float(np.mean(target_drops)),
            "original_class_probability_drop_ci95": bootstrap_mean_ci(target_drops),
            "prediction_flip_rate": float(np.mean(target_flips)),
            "entropy_increase_rate": float(np.mean(np.asarray(target_entropy) > 0)),
            "reported_cue_retention_rate": float(np.mean([
                row["vlm_reported_cue_retained"] for row in reported_targets
            ])),
            "reported_minus_unreported_drop_mean": (
                float(np.mean(comparator_differences)) if comparator_differences else float("nan")
            ),
            "reported_minus_unreported_drop_ci95": (
                bootstrap_mean_ci(comparator_differences) if comparator_differences else [float("nan"), float("nan")]
            ),
            "reported_minus_same_cue_control_drop_mean": (
                float(np.mean(same_cue_control_differences))
                if same_cue_control_differences else float("nan")
            ),
            "reported_minus_same_cue_control_drop_ci95": (
                bootstrap_mean_ci(same_cue_control_differences)
                if same_cue_control_differences else [float("nan"), float("nan")]
            ),
            "by_reported_cue": by_reported_cue,
            "probability_note": "VLM probabilities are ordinal confidence proxies; prediction flips are the primary behavioural endpoint.",
        }
    commitment_targets = [
        row for row in reported_targets
        if row.get("vlm_commitment_valid") == 1.0
    ]
    if commitment_targets:
        def group_rate(group, key):
            return float(np.mean([row[key] for row in group])) if group else float("nan")

        by_role = {}
        for role in ["essential", "supportive", "incidental"]:
            group = [row for row in commitment_targets if row.get("vlm_declared_cue_role") == role]
            if group:
                by_role[role] = {
                    "n": len(group),
                    "outcome_type_accuracy": group_rate(group, "vlm_outcome_type_commitment_match"),
                    "exact_commitment_accuracy": group_rate(group, "vlm_counterfactual_commitment_kept"),
                    "prediction_flip_rate": group_rate(group, "vlm_original_prediction_flip"),
                    "original_class_probability_drop_mean": float(np.mean([
                        row["vlm_original_class_probability_drop"] for row in group
                    ])),
                }

        substitutions = [row for row in commitment_targets if row["vlm_cue_substitution"] == 1.0]
        substitutions_with_backup = [
            row for row in substitutions
            if row.get("vlm_declared_backup_cue") not in {"", "none"}
        ]
        backup_targets = [
            row for row in eligible
            if row.get("report_condition_role") == "declared_backup_cue_target"
            and row.get("vlm_valid") == 1.0
            and row.get("vlm_commitment_valid") == 1.0
        ]
        valid_chains = [
            row for row in chain_rows
            if row.get("vlm_valid") == 1.0
            and row.get("vlm_commitment_valid") == 1.0
        ]
        primary_by_sample = {row["sample_id"]: row for row in commitment_targets}
        chain_incremental_drops = []
        chain_incremental_flips = []
        for chain in valid_chains:
            primary = primary_by_sample.get(chain["sample_id"])
            if primary is None:
                continue
            chain_incremental_drops.append(
                chain["vlm_original_class_probability_drop"]
                - primary["vlm_original_class_probability_drop"]
            )
            chain_incremental_flips.append(float(
                chain.get("vlm_twin_prediction") != primary.get("vlm_twin_prediction")
            ))

        commitment = {
            "n_valid_primary_commitments": len(commitment_targets),
            "outcome_type_accuracy": group_rate(commitment_targets, "vlm_outcome_type_commitment_match"),
            "expected_emotion_accuracy": group_rate(commitment_targets, "vlm_expected_emotion_match"),
            "exact_commitment_accuracy": group_rate(commitment_targets, "vlm_counterfactual_commitment_kept"),
            "role_outcome_declaration_coherence_rate": group_rate(
                commitment_targets, "vlm_role_outcome_declaration_coherent"
            ),
            "outcome_emotion_declaration_coherence_rate": group_rate(
                commitment_targets, "vlm_outcome_emotion_declaration_coherent"
            ),
            "full_declaration_coherence_rate": group_rate(
                commitment_targets, "vlm_commitment_declaration_coherent"
            ),
            "cue_substitution_rate": group_rate(commitment_targets, "vlm_cue_substitution"),
            "declared_backup_activation_rate_given_substitution": (
                group_rate(substitutions_with_backup, "vlm_declared_backup_cue_activated")
                if substitutions_with_backup else float("nan")
            ),
            "post_edit_rerationalization_rate": group_rate(
                commitment_targets, "vlm_post_edit_rerationalization"
            ),
            "forecast_consistent_stability_rate": group_rate(
                commitment_targets, "vlm_forecast_consistent_stability"
            ),
            "by_declared_role": by_role,
            "n_declared_backup_targets": len(backup_targets),
            "declared_backup_target_flip_rate": (
                group_rate(backup_targets, "vlm_original_prediction_flip")
                if backup_targets else float("nan")
            ),
            "declared_backup_target_class_drop_mean": (
                float(np.mean([row["vlm_original_class_probability_drop"] for row in backup_targets]))
                if backup_targets else float("nan")
            ),
            "n_primary_plus_backup_chains": len(valid_chains),
            "primary_plus_backup_chain_flip_rate": (
                group_rate(valid_chains, "vlm_original_prediction_flip")
                if valid_chains else float("nan")
            ),
            "primary_plus_backup_chain_class_drop_mean": (
                float(np.mean([row["vlm_original_class_probability_drop"] for row in valid_chains]))
                if valid_chains else float("nan")
            ),
            "chain_incremental_class_drop_over_primary_mean": (
                float(np.mean(chain_incremental_drops))
                if chain_incremental_drops else float("nan")
            ),
            "chain_changes_post_primary_label_rate": (
                float(np.mean(chain_incremental_flips))
                if chain_incremental_flips else float("nan")
            ),
            "probability_note": "VLM confidence levels are mapped to ordinal probability proxies; categorical forecasts and label changes are the primary endpoints.",
        }
        if "essential" in by_role and "incidental" in by_role:
            commitment["essential_minus_incidental_flip_rate"] = (
                by_role["essential"]["prediction_flip_rate"]
                - by_role["incidental"]["prediction_flip_rate"]
            )
        result["counterfactual_cue_commitments"] = commitment
    return result
