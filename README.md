# Counterfactual Affective Twins

An end-to-end, standalone framework for auditing whether affect models respond to inspectable evidence rather than only matching image-level labels.

The framework creates matched twins that alter one cue family at a time:

- `color_lighting`: independently selected, luminance-preserving local chroma removal;
- `facial_action_region`: brow, eye, or mouth evidence ablation in a verified human face region;
- `scene_context`: subject-preserving background chroma/detail attenuation;
- `embedded_text`: OCR-localized text removal when text exists, plus contradictory affect-word insertion.

It evaluates categorical emotion, continuous valence–arousal, response direction, probability/distribution change, uncertainty under conflict, content preservation, calibration, caption stability, and cue-evidence grounding. Every twin has a mask and a JSONL manifest entry. Ineligible cues remain in the manifest with a skip reason.

## Project boundary

All code, environments, model assets, generated twins, masks, checkpoints, manifests, and reports are inside this folder. Existing Emotion6 annotations and the MediaPipe task file are read-only inputs referenced by the default configuration; no files are written to other thesis projects.

## Reproduce

```bash
cd /Users/rijju/Documents/PhD_thesis/visual-emotion-cue-audit
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[face,vlm,dev]'

export TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/huggingface"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"

.venv/bin/affective-twins doctor
.venv/bin/affective-twins train
.venv/bin/affective-twins generate
.venv/bin/affective-twins evaluate
.venv/bin/affective-twins report
```

Use `.venv/bin/affective-twins all` after a checkpoint exists to run generation, evaluation, and reporting together. Configuration is in `configs/emotion6_full.json`.

## Outputs

The verified run is in `runs/emotion6_full/`:

- `samples.jsonl` and `interventions.jsonl`: reproducible source/twin manifests;
- `twins/` and `masks/`: cue-separated image pairs and exact edit masks;
- `predictions.jsonl` and `pair_metrics.csv`: pair-level outputs;
- `summary.json`: aggregate calibration, uncertainty, VA, and cue metrics;
- `contact_sheet.png` and `report.md`: visually checked qualitative/quantitative report;
- `human_validation_template.csv`: blinded human-rating protocol.

After annotators complete the template, summarize it with:

```bash
.venv/bin/affective-twins human-summary \
  --annotations runs/emotion6_full/human_validation_template.csv
```

## Verified classifier/VA run

The deterministic seed-42 run uses 60 balanced held-out Emotion6 images. The independent evaluator has 51.7% test accuracy and 0.326 VA MAE on the normalized `[-1, 1]` scale. It evaluates 181 eligible twins:

| Cue | Eligible pairs | Directional success | Mean source-probability drop | Feature cosine |
|---|---:|---:|---:|---:|
| Colour/lighting | 60 | 95.0% | 0.070 | 0.975 |
| Scene context | 60 | 60.0% | 0.015 | 0.941 |
| Embedded-text conflict | 60 | 45.0% uncertainty increase | 0.029 | 0.974 |
| Facial action region | 1 | 100.0% | 0.070 | 0.982 |

The face result has `n=1` and is not a general performance claim. Fifty-nine images lacked a verified human face. This conservative eligibility decision was introduced after visual QA found animal-face false positives in weaker detectors.

## VLM status

`SmolVLMAdapter` implements the complete VLM contract: discrete emotion probabilities, VA, confidence, cue identification, visible evidence, and a literal caption. The 500M default checkpoint was smoke-tested, but it returned free text rather than the requested JSON. The adapter records this explicitly as `fallback_free_text` and assigns conservative confidence. Therefore `enable_vlm` remains `false` for the reported benchmark; enable it only after the selected VLM passes `scripts/smoke_vlm.py` with `parse_status: valid_json`.

## Interpretation boundary

This is a complete executable research framework, not proof of perceptual causality. Current face edits ablate AU-related anatomical regions; they do not synthesize exact muscle activations. Context masks use a deterministic saliency prior rather than semantic segmentation. Human validation remains required before reporting the twins as perception-preserving causal interventions. The CAUSE composite in `summary.json` is explicitly unvalidated and should be treated as a run diagnostic, not a leaderboard score.

The original colour-only pilot and anonymous one-page PDF remain in `outputs/`, `abstract/`, and `submission/` for provenance.
