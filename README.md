# Counterfactual Cue Commitments for Visual Emotion Models

An executable audit of whether a visual emotion model can predict its own behaviour under a cue intervention. The working paper title is:

> **Do Visual Emotion Models Keep Their Cue Commitments? Counterfactual Self-Prediction in Affective Reasoning**

The protocol is prospective. On the untouched image, the target VLM must first report:

1. its emotion prediction;
2. one visible primary cue and the exact evidence region;
3. whether that cue is `essential`, `supportive`, or `incidental`;
4. the expected intervention outcome: `label_change`, `confidence_decrease_same_label`, or `no_material_change`;
5. the expected post-intervention emotion; and
6. one backup cue and its visible evidence, or `none`.

Only then does the framework ground and attenuate the named evidence. It re-queries the same VLM and measures whether the categorical outcome and emotion forecast were correct, whether the model substituted another cue, whether it activated the backup it declared in advance, or whether it instead produced a post-edit rationale that was not predeclared. When both primary and backup evidence can be grounded, the framework also creates a sequential primary-plus-backup twin and measures its incremental effect.

This is more specific than asking whether an edited cue changes a prediction. The central object is a falsifiable, pre-intervention behavioural commitment. It does not claim direct access to the model's internal causal mechanism.

## Intervention families

- `color_lighting`: object- or surface-level chroma/lighting attenuation over complete semantic masks such as person, wall, floor, or sky. Rectangular patch interventions are not used.
- `facial_action_region`: landmark-contoured brows, eyes, or mouth ablation using strong blur and pixelation; Aff-Wild2 examples retain frame-level AU annotations.
- `scene_context`: background chroma/detail attenuation while preserving segmented foreground subjects.
- `embedded_text`: optional OCR-grounded removal. It is disabled for the current Emotion6 + Aff-Wild2 audit because these selected images do not provide suitable text evidence.

A cue is eligible only when its stated evidence can be matched to a generated region. Ambiguous semantic matches require the same physical region to be selected under shuffled option orders. Failed grounding remains an explicit audit failure; the framework does not silently replace the named evidence with a convenient patch.

## Data and labels

The portable audit set contains 60 Emotion6 images, balanced using the ten strongest unique human-plurality annotations per modeled emotion, plus 20 Aff-Wild2 frames with annotation-backed facial AUs. Emotion6's Flickr folder label is preserved separately as `nominal_emotion` for disagreement analysis. Neutral mass is retained in the human distribution but renormalized out of the six-way evaluator target. The 60 Emotion6 audit images are held out from evaluator training.

All code, assets, configurations, outputs, masks, and reports belong inside this repository. No generated file is written into `Learning-Emotion-palettes` or another thesis project.

## Local setup and checks

```bash
cd /Users/rijju/Documents/PhD_thesis/visual-emotion-cue-audit
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[face,vlm,dev]'

export TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/huggingface"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"

.venv/bin/affective-twins doctor --config configs/emotion6_abaw80_commitment_qwen25vl3b_server.json
.venv/bin/python -m pytest -q
```

The full commitment audit is intended for the GPU server. Submit the main Qwen run with:

```bash
sbatch jobs/full_vlm_gpu.job
```

It writes a new run to `runs/emotion6_abaw80_commitment_qwen25vl3b_server/`. To run the compact SmolVLM baseline followed by Qwen and compare their prospective commitments, use:

```bash
sbatch jobs/compare_vlms_gpu.job
```

The models run sequentially to reduce GPU-memory pressure. Their comparison is written to `runs/vlm_cue_commitment_comparison/`.

## Commitment-audit outputs

- `original_reports.jsonl`: immutable pre-intervention predictions, evidence, cue roles, outcome forecasts, post-edit emotion forecasts, and backup commitments.
- `interventions.jsonl`: grounded primary targets, declared-backup targets, controls, sequential chains, masks, and explicit skip reasons.
- `predictions.jsonl` and `pair_metrics.csv`: target-VLM pre/post responses plus independent ResNet evaluator diagnostics.
- `summary.json`: commitment accuracy, role-stratified effects, cue substitution, backup activation, rerationalization, and sequential-chain effects.
- `twins/commitment_chain/`: primary-plus-declared-backup twins.
- `contact_sheet.png`, `report.md`, and `human_validation_template.csv`: qualitative review and validation material.

All pre-commitment results are isolated under `runs/baseline/`, including the two `emotion6_abaw80_exact_*` runs and their comparison. They are retained only for provenance and baseline analysis. They predate prospective cue-role, outcome, and backup commitments and cannot be used as results for the new claim. New commitment runs are written directly under `runs/` with the `emotion6_abaw80_commitment_*` prefix.

## Primary endpoints

The strongest endpoints are categorical:

- outcome-type forecast accuracy;
- expected post-edit emotion accuracy;
- exact commitment accuracy, requiring both forecasts to be correct;
- prediction-flip rate stratified by declared cue role;
- declared-backup activation after cue substitution;
- post-edit rerationalization rate; and
- incremental effect of primary-plus-backup ablation over the primary intervention alone.

VLM confidence levels are mapped to ordinal probability proxies. Probability drops are therefore secondary diagnostics, not calibrated likelihood claims. The independent ResNet evaluator provides additional sensitivity, calibration, and feature-preservation diagnostics but does not decide whether the target VLM kept its commitment.

## Interpretation boundary

These experiments test behavioural consistency under controlled image edits. They do not prove that a verbal report identifies an internal causal computation. Facial edits remove visible AU-related regions but do not synthesize anatomically exact muscle activations. Segmentation limits which objects and surfaces can be grounded. Human validation is still required before claiming that twins preserve all non-target affective content.

The original colour-only pilot and earlier submission artefacts remain in `outputs/`, `abstract/`, and `submission/` for provenance; their claims should not be copied into the commitment paper without a fresh server run.
