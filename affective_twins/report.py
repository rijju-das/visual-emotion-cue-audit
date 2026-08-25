"""Human-readable audit report and qualitative contact sheet."""

from pathlib import Path
import textwrap
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont

from .io import read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _portable_image_path(run_dir: Path, recorded_path: str, source: bool) -> Path:
    """Resolve server-authored absolute paths after a run is downloaded locally."""
    recorded = Path(recorded_path)
    if recorded.is_file():
        return recorded
    if source:
        local_source = PROJECT_ROOT / "data" / "audit80" / "images" / recorded.name
        if local_source.is_file():
            return local_source
    else:
        matches = list((run_dir / "twins").glob("*/{}".format(recorded.name)))
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError(
        "Could not resolve downloaded image path: {}".format(recorded_path)
    )


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
    cue_order = ["color_lighting", "facial_action_region", "scene_context", "embedded_text"]
    available_cues = [cue for cue in cue_order if any(row["cue_family"] == cue for row in eligible)]
    per_cue = max(1, (limit + max(1, len(available_cues)) - 1) // max(1, len(available_cues)))
    for cue in available_cues:
        cue_rows = [
            row for row in eligible
            if row["cue_family"] == cue
            and row.get("metadata", {}).get("report_condition_role") == "reported_cue_target"
        ]
        if not cue_rows:
            cue_rows = [row for row in eligible if row["cue_family"] == cue]
        interventions.extend(cue_rows[:per_cue])
    interventions = interventions[:limit]
    panel_width, panel_height, label_height = 280, 190, 108
    canvas = Image.new("RGB", (panel_width * 2, (panel_height + label_height) * len(interventions)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, intervention in enumerate(interventions):
        paths = [
            _portable_image_path(
                run_dir, samples[intervention["sample_id"]]["image_path"], source=True
            ),
            _portable_image_path(run_dir, intervention["image_path"], source=False),
        ]
        for column, path in enumerate(paths):
            with Image.open(path) as source:
                image = source.convert("RGB")
            image.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
            x = column * panel_width + (panel_width - image.width) // 2
            y = row_index * (panel_height + label_height) + (panel_height - image.height) // 2
            canvas.paste(image, (x, y))
        metadata = intervention.get("metadata", {})
        role = metadata.get("report_condition_role", "")
        evidence = metadata.get("original_reported_evidence", "")
        selected = metadata.get("selected_candidate_label", "")
        label = "{} | {}\ncommitment: {} / {}\nreport: {} -> edit: {}\nbackup: {} ({})".format(
            intervention["cue_family"],
            role,
            metadata.get("original_declared_cue_role", "n/a"),
            metadata.get("original_expected_intervention_outcome", "n/a"),
            textwrap.shorten(evidence, width=34, placeholder="..."),
            textwrap.shorten(selected, width=34, placeholder="..."),
            metadata.get("original_declared_backup_cue", "n/a"),
            textwrap.shorten(metadata.get("original_declared_backup_evidence", ""), width=24, placeholder="..."),
        )
        draw.multiline_text(
            (8, row_index * (panel_height + label_height) + panel_height + 6),
            label,
            fill="black",
            font=_font(13),
            spacing=4,
        )
    path = run_dir / "contact_sheet.png"
    canvas.save(path)
    return path


def write_report(run_dir: Path, summary: Dict) -> Path:
    sheet = make_contact_sheet(run_dir)
    cue_rows = []
    for cue, metrics in summary.get("by_cue", {}).items():
        cue_rows.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} |".format(cue, metrics["n"], metrics["directional_success_rate"], metrics["source_probability_drop_mean"], metrics["feature_cosine_mean"]))
    sensitivity = summary.get("reported_cue_behavioral_sensitivity")
    # Backward-compatible read path for historical runs; new summaries never
    # label behavioural consistency as explanation faithfulness.
    faithfulness = sensitivity or summary.get("report_conditioned_faithfulness")
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
        faithfulness_section = """## Report-conditioned behavioural sensitivity

- Reported-cue target coverage: {coverage:.1%}
- No reported-cue target rate: {failure:.1%}
- Valid original-report rate: {valid_report_rate:.1%}
- Exact evidence-grounding rate among valid reports: {grounding_rate:.1%}
- Valid reported-cue targets: {targets}
- Unreported-cue comparators: {comparators}
- Original VLM-class probability drop: {drop:.3f}
- VLM prediction-flip rate: {flip:.1%}
- Reported-minus-unreported drop: {unreported_difference:.3f}
- Reported-minus-same-cue-control drop: {control_difference:.3f}

| Originally reported cue | n | Original-class drop | Prediction flip | Cue retained after edit |
|---|---:|---:|---:|---:|
{reported_cue_rows}

The target VLM supplies both the pre-intervention report and the post-intervention response. This section measures behavioural consistency with its reported evidence, not access to its internal causal mechanism. Probability changes use an ordinal confidence proxy; prediction flips are the primary endpoint.
""".format(
            coverage=summary.get("reported_cue_target_coverage", 0.0),
            failure=summary.get("reported_cue_audit_failure_rate", 0.0),
            valid_report_rate=summary.get("original_report_valid_rate", 0.0),
            grounding_rate=summary.get("valid_report_exact_grounding_rate", 0.0),
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
    commitments = summary.get("counterfactual_cue_commitments")
    if commitments:
        role_rows = []
        for role, metrics in commitments.get("by_declared_role", {}).items():
            role_rows.append(
                "| {} | {} | {:.1%} | {:.1%} | {:.1%} | {:.3f} |".format(
                    role,
                    metrics["n"],
                    metrics["outcome_type_accuracy"],
                    metrics["exact_commitment_accuracy"],
                    metrics["prediction_flip_rate"],
                    metrics["original_class_probability_drop_mean"],
                )
            )
        commitment_section = """## Counterfactual cue commitments

Before seeing any edited image, the VLM committed to the named cue's role, the categorical effect of attenuating it, the post-edit emotion, and a backup cue. These scores test whether those prospective claims survive the intervention.

- Valid grounded commitments evaluated: {n}
- Outcome-type forecast accuracy: {outcome:.1%}
- Expected post-edit emotion accuracy: {emotion:.1%}
- Exact commitment accuracy (both): {exact:.1%}
- Full declaration coherence: {coherence:.1%}
- Post-edit cue substitution rate: {substitution:.1%}
- Declared-backup activation given substitution: {backup:.1%}
- Post-edit rerationalization rate: {rerationalization:.1%}
- Declared-backup targets grounded: {backup_targets}
- Sequential primary+backup twins evaluated: {chains}
- Chain incremental class drop over primary: {chain_increment:.3f}

| Declared role | n | Outcome forecast | Exact commitment | Prediction flip | Class drop |
|---|---:|---:|---:|---:|---:|
{role_rows}

The sequential twin is generated only when both the predeclared primary and backup evidence can be grounded. It tests whether removing the advertised fallback adds an effect beyond removing the primary cue alone.
""".format(
            n=commitments["n_valid_primary_commitments"],
            outcome=commitments["outcome_type_accuracy"],
            emotion=commitments["expected_emotion_accuracy"],
            exact=commitments["exact_commitment_accuracy"],
            coherence=commitments["full_declaration_coherence_rate"],
            substitution=commitments["cue_substitution_rate"],
            backup=commitments["declared_backup_activation_rate_given_substitution"],
            rerationalization=commitments["post_edit_rerationalization_rate"],
            backup_targets=commitments["n_declared_backup_targets"],
            chains=commitments["n_primary_plus_backup_chains"],
            chain_increment=commitments["chain_incremental_class_drop_over_primary_mean"],
            role_rows="\n".join(role_rows),
        )
    else:
        commitment_section = ""
    if summary.get("report_conditioned", False):
        if summary.get("commitment_audit", False):
            protocol_intro = (
                "This run prospectively audits the exact cue named by the target VLM. "
                "Before intervention, the model commits to what should happen and which backup cue "
                "it would use; the immutable report then grounds the primary and backup twins."
            )
        else:
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
    report = """# Counterfactual Cue Commitment audit

{protocol_intro} Valence and arousal use the normalized range `[-1, 1]`.

## Run summary

- Samples: {n_samples}
- Target VLM: {vlm_model}
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
{faithfulness_section}

{commitment_section}

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
        commitment_section=commitment_section,
        protocol_intro=protocol_intro,
        **summary
    )
    path = run_dir / "report.md"
    path.write_text(report)
    return path
