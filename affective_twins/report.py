"""Human-readable audit report and qualitative contact sheet."""

from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont

from .io import read_jsonl


def _font(size: int):
    for path in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Helvetica.ttf"]:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_contact_sheet(run_dir: Path, limit: int = 8) -> Path:
    samples = {row["sample_id"]: row for row in read_jsonl(run_dir / "samples.jsonl")}
    eligible = [row for row in read_jsonl(run_dir / "interventions.jsonl") if row["eligible"]]
    interventions = []
    for cue in ["color_lighting", "facial_action_region", "scene_context", "embedded_text"]:
        cue_rows = [
            row for row in eligible
            if row["cue_family"] == cue
            and row.get("metadata", {}).get("report_condition_role") == "reported_cue_target"
        ]
        if not cue_rows:
            cue_rows = [row for row in eligible if row["cue_family"] == cue]
        interventions.extend(cue_rows[: max(1, limit // 4)])
    interventions = interventions[:limit]
    panel_width, panel_height, label_height = 280, 190, 45
    canvas = Image.new("RGB", (panel_width * 2, (panel_height + label_height) * len(interventions)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, intervention in enumerate(interventions):
        paths = [samples[intervention["sample_id"]]["image_path"], intervention["image_path"]]
        for column, path in enumerate(paths):
            with Image.open(path) as source:
                image = source.convert("RGB")
            image.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
            x = column * panel_width + (panel_width - image.width) // 2
            y = row_index * (panel_height + label_height) + (panel_height - image.height) // 2
            canvas.paste(image, (x, y))
        role = intervention.get("metadata", {}).get("report_condition_role", "")
        label = "{} | {} | {}".format(intervention["cue_family"], role, intervention["operation"])
        draw.text((8, row_index * (panel_height + label_height) + panel_height + 8), label, fill="black", font=_font(14))
    path = run_dir / "contact_sheet.png"
    canvas.save(path)
    return path


def write_report(run_dir: Path, summary: Dict) -> Path:
    sheet = make_contact_sheet(run_dir)
    cue_rows = []
    for cue, metrics in summary.get("by_cue", {}).items():
        cue_rows.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} |".format(cue, metrics["n"], metrics["directional_success_rate"], metrics["source_probability_drop_mean"], metrics["feature_cosine_mean"]))
    faithfulness = summary.get("report_conditioned_faithfulness")
    if faithfulness:
        reported_cue_rows = []
        for cue, metrics in faithfulness.get("by_reported_cue", {}).items():
            reported_cue_rows.append(
                "| {} | {} | {:.3f} | {:.1%} | {:.1%} |".format(
                    cue,
                    metrics["n"],
                    metrics["original_class_probability_drop_mean"],
                    metrics["prediction_flip_rate"],
                    metrics["reported_cue_retention_rate"],
                )
            )
        faithfulness_section = """## Report-conditioned faithfulness

- Reported-cue target coverage: {coverage:.1%}
- No reported-cue target rate: {failure:.1%}
- Valid original-report rate: {valid_report_rate:.1%}
- Valid reported-cue targets: {targets}
- Unreported-cue comparators: {comparators}
- Original VLM-class probability drop: {drop:.3f}
- VLM prediction-flip rate: {flip:.1%}
- Reported-minus-unreported drop: {unreported_difference:.3f}
- Reported-minus-same-cue-control drop: {control_difference:.3f}

| Originally reported cue | n | Original-class drop | Prediction flip | Cue retained after edit |
|---|---:|---:|---:|---:|
{reported_cue_rows}

The target VLM supplies both the pre-intervention report and the post-intervention response. Probability changes use an ordinal confidence proxy; prediction flips are the primary behavioural endpoint.
""".format(
            coverage=summary.get("reported_cue_target_coverage", 0.0),
            failure=summary.get("reported_cue_audit_failure_rate", 0.0),
            valid_report_rate=summary.get("original_report_valid_rate", 0.0),
            targets=faithfulness["n_reported_cue_targets"],
            comparators=faithfulness["n_unreported_cue_comparators"],
            drop=faithfulness["original_class_probability_drop_mean"],
            flip=faithfulness["prediction_flip_rate"],
            unreported_difference=faithfulness["reported_minus_unreported_drop_mean"],
            control_difference=faithfulness["reported_minus_same_cue_control_drop_mean"],
            reported_cue_rows="\n".join(reported_cue_rows),
        )
    else:
        faithfulness_section = ""
    if summary.get("report_conditioned", False):
        protocol_intro = (
            "This run audits the exact cue named by the target VLM before intervention. "
            "The stored original report selects the target region and is reused for the "
            "same-model pre/post comparison; an independent evaluator supplies secondary diagnostics."
        )
    else:
        protocol_intro = (
            "This run evaluates matched original/counterfactual pairs with an evaluator independent "
            "from the model used to select affect-sensitive colour or facial regions."
        )
    report = """# Counterfactual Affective Twins audit

{protocol_intro} Valence and arousal use the normalized range `[-1, 1]`.

## Run summary

- Samples: {n_samples}
- Eligible pairs: {n_pairs}
- Skipped cue/sample combinations: {n_skipped}
- Cue coverage: {cue_coverage:.1%}
- Directional success: {directional_success_rate:.1%}
- Mean emotion-distribution JS divergence: {emotion_js_divergence_mean:.3f}
- Mean VA displacement: {va_distance_mean:.3f}
- Mean frozen-feature cosine preservation: {feature_cosine_mean:.3f}
- Original-prediction ECE: {original_ece:.3f}
- Original Brier score against human distribution: {original_brier_mean:.3f}
- Original accuracy against human plurality: {original_human_plurality_accuracy:.1%}
- Original accuracy against Flickr folder label: {original_folder_accuracy:.1%}
- Folder/human-plurality agreement: {folder_human_agreement_rate:.1%}
- Original VA MAE: {original_va_mae:.3f}
- Conflict uncertainty success: {conflict_uncertainty_success_rate:.1%}
- CAUSE diagnostic score: {cause_diagnostic_score:.3f}

The CAUSE value is an unvalidated diagnostic composite, not a benchmark claim.

{faithfulness_section}

## Results by cue

| Cue family | n | Directional success | Source-probability drop | Feature cosine |
|---|---:|---:|---:|---:|
{cue_rows}

## Qualitative pairs

![Original and counterfactual pairs]({sheet_name})

## Interpretation boundary

These results measure model sensitivity under controlled image edits. They do not establish human-perceptual causality. Facial operations ablate localized AU-related evidence regions; they do not synthesize anatomically exact Action Unit activations.
""".format(
        cue_rows="\n".join(cue_rows),
        sheet_name=sheet.name,
        faithfulness_section=faithfulness_section,
        protocol_intro=protocol_intro,
        **summary
    )
    path = run_dir / "report.md"
    path.write_text(report)
    return path
