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


def aggregate(rows: List[Dict]) -> Dict:
    all_eligible = [row for row in rows if row.get("eligible", True)]
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
    cue_coverage = len(by_cue) / 4.0
    response = mean("directional_success")
    preservation = float(np.clip((mean("feature_cosine") + 1.0) / 2.0, 0.0, 1.0))
    uncertainty_rows = [row for row in eligible if row["operation"] == "insert_affect_conflict_text"]
    uncertainty = float(np.mean([row["entropy_change"] > 0 for row in uncertainty_rows])) if uncertainty_rows else 0.0
    calibration = float(np.clip(1.0 - ece, 0.0, 1.0))
    vlm_rows = [row for row in eligible if row.get("vlm_valid") == 1.0]
    grounding = float(np.mean([row["vlm_cue_grounded"] for row in vlm_rows])) if vlm_rows else None
    grounding_or_coverage = grounding if grounding is not None else cue_coverage
    cause = float(np.mean([response, preservation, grounding_or_coverage, uncertainty, calibration]))
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
    result = {"n_pairs": len(eligible), "n_control_pairs": len(controls), "cue_coverage": cue_coverage, "directional_success_rate": response, "emotion_js_divergence_mean": mean("emotion_js_divergence"), "va_distance_mean": mean("va_distance"), "feature_cosine_mean": mean("feature_cosine"), "conflict_uncertainty_success_rate": uncertainty, "original_ece": ece, "original_brier_mean": float(np.mean([row["original_brier"] for row in unique_originals])), "original_nll_mean": float(np.mean([row["original_nll"] for row in unique_originals])), "original_human_plurality_accuracy": float(np.mean([row["original_correct"] for row in unique_originals])), "original_folder_accuracy": float(np.mean([row["original_folder_correct"] for row in folder_rows])) if folder_rows else float("nan"), "folder_human_agreement_rate": float(np.mean([row["folder_human_agreement"] for row in folder_rows])) if folder_rows else float("nan"), "original_va_mae": float(np.mean([0.5 * (row["original_valence_absolute_error"] + row["original_arousal_absolute_error"]) for row in unique_originals if not math.isnan(row["original_valence_absolute_error"])])), "cause_diagnostic_score": cause, "cause_note": "Unvalidated diagnostic composite: response, content preservation, cue grounding (or coverage when VLM is disabled), conflict uncertainty, calibration.", "by_cue": by_cue, "matched_control_analysis": paired_by_cue}
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
    return result
