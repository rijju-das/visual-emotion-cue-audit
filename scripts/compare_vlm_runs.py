#!/usr/bin/env python3
"""Compare counterfactual cue-commitment runs without pooling target VLMs."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def parse_run(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=RUN_DIRECTORY")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--run must be LABEL=RUN_DIRECTORY")
    return label.strip(), Path(path).resolve()


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def collect(label, run_dir):
    required = ["summary.json", "samples.jsonl", "original_reports.jsonl", "predictions.jsonl"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("{} is missing: {}".format(run_dir, ", ".join(missing)))
    summary = json.loads((run_dir / "summary.json").read_text())
    samples = {row["sample_id"]: row for row in read_jsonl(run_dir / "samples.jsonl")}
    reports = read_jsonl(run_dir / "original_reports.jsonl")
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    valid_reports = [row for row in reports if row.get("valid_report")]
    correct_reports = [
        row for row in valid_reports
        if row.get("predicted_emotion") == samples.get(row["sample_id"], {}).get("emotion")
    ]
    targets = [row for row in predictions if row.get("report_condition_role") == "reported_cue_target"]
    exact_targets = [row for row in targets if float(row.get("reported_evidence_region_match", 0.0)) == 1.0]
    valid_targets = [row for row in exact_targets if float(row.get("vlm_valid", 0.0)) == 1.0]
    drops = [
        finite(row.get("vlm_original_class_probability_drop"))
        for row in valid_targets
    ]
    drops = [value for value in drops if value is not None]
    flips = [float(row.get("vlm_original_prediction_flip", 0.0)) for row in valid_targets]
    commitments = summary.get("counterfactual_cue_commitments", {})
    if not commitments:
        raise ValueError(
            "{} is a legacy run without counterfactual cue commitments; rerun with a commitment config".format(run_dir)
        )
    cue_counts = Counter(row.get("evidence_cue", "invalid") for row in valid_reports)
    return {
        "label": label,
        "run_dir": str(run_dir),
        "vlm_model": summary.get("vlm_model") or (reports[0].get("reporting_model") if reports else ""),
        "samples": len(samples),
        "valid_report_rate": len(valid_reports) / len(reports) if reports else 0.0,
        "valid_commitment_rate": summary.get("original_commitment_valid_rate", 0.0),
        "valid_report_human_accuracy": len(correct_reports) / len(valid_reports) if valid_reports else 0.0,
        "exact_grounding_rate": summary.get(
            "valid_commitment_primary_target_rate",
            len(exact_targets) / len(valid_reports) if valid_reports else 0.0,
        ),
        "exact_grounded_targets": len(exact_targets),
        "valid_exact_target_pairs": len(valid_targets),
        "target_prediction_flip_rate": sum(flips) / len(flips) if flips else 0.0,
        "target_original_class_drop_mean": sum(drops) / len(drops) if drops else 0.0,
        "outcome_type_accuracy": commitments.get("outcome_type_accuracy"),
        "expected_emotion_accuracy": commitments.get("expected_emotion_accuracy"),
        "exact_commitment_accuracy": commitments.get("exact_commitment_accuracy"),
        "declaration_coherence_rate": commitments.get("full_declaration_coherence_rate"),
        "cue_substitution_rate": commitments.get("cue_substitution_rate"),
        "backup_activation_rate": commitments.get("declared_backup_activation_rate_given_substitution"),
        "rerationalization_rate": commitments.get("post_edit_rerationalization_rate"),
        "chain_count": commitments.get("n_primary_plus_backup_chains", 0),
        "chain_incremental_drop": commitments.get("chain_incremental_class_drop_over_primary_mean"),
        "reported_cue_counts": dict(sorted(cue_counts.items())),
    }


def percent(value):
    value = finite(value)
    return "n/a" if value is None else "{:.1%}".format(value)


def number(value):
    value = finite(value)
    return "n/a" if value is None else "{:.4f}".format(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) < 2:
        parser.error("provide at least two --run LABEL=RUN_DIRECTORY arguments")

    rows = [collect(label, run_dir) for label, run_dir in args.run]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps({"runs": rows}, indent=2, sort_keys=True))

    scalar_columns = [key for key, value in rows[0].items() if not isinstance(value, dict)]
    with (output_dir / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_columns)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in scalar_columns} for row in rows])

    table_rows = []
    for row in rows:
        table_rows.append(
            "| {label} | {valid} | {commitment} | {grounding} | {targets} | {coherence} | {outcome} | {emotion} | {exact} | {backup} | {rerationalization} | {chains} | {increment} |".format(
                label=row["label"],
                valid=percent(row["valid_report_rate"]),
                commitment=percent(row["valid_commitment_rate"]),
                grounding=percent(row["exact_grounding_rate"]),
                targets=row["exact_grounded_targets"],
                coherence=percent(row["declaration_coherence_rate"]),
                outcome=percent(row["outcome_type_accuracy"]),
                emotion=percent(row["expected_emotion_accuracy"]),
                exact=percent(row["exact_commitment_accuracy"]),
                backup=percent(row["backup_activation_rate"]),
                rerationalization=percent(row["rerationalization_rate"]),
                chains=row["chain_count"],
                increment=number(row["chain_incremental_drop"]),
            )
        )
    report = """# Counterfactual Cue Commitment VLM comparison

Each VLM is audited against its own immutable prospective commitment. Results are not pooled because models can name different primary and backup evidence and therefore induce different twins.

| VLM | Valid reports | Valid commitments | Exact grounding | Targets | Coherent contract | Outcome forecast | Emotion forecast | Exact commitment | Backup activation | Rerationalization | Chains | Chain incremental drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

Categorical outcome and post-edit emotion forecasts are the primary endpoints. Probability drops use an ordinal confidence proxy. Backup activation tests whether a model switches to the fallback it declared before intervention; rerationalization records a switch to some other cue. A low grounding rate is itself an audit result because the stated evidence could not be matched without substitution.
""".format(rows="\n".join(table_rows))
    (output_dir / "comparison.md").write_text(report)
    print(json.dumps({"output_dir": str(output_dir), "runs": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
