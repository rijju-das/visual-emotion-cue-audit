# Counterfactual Affective Twins

An end-to-end, standalone framework for auditing whether affect models respond to inspectable evidence rather than only matching image-level labels.

The framework creates matched twins that alter one cue family at a time:

- `color_lighting`: two explicit Mask R-CNN twins—luminance-preserving chroma removal over the complete subject (all people merged) and background-only exposure reduction;
- `facial_action_region`: landmark-contoured eyebrow, eye, or mouth evidence ablation, linked only to annotation-supported AUs;
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

Use `.venv/bin/affective-twins all --config configs/emotion6_abaw80_vlm_server.json` after a checkpoint exists to run generation, evaluation, and reporting together. The GPU workflow is also available in `jobs/full_vlm_gpu.job`.

The strengthened subject/background and landmark run uses `configs/emotion6_abaw80_vlm_subject_background_server.json`; `jobs/full_vlm_gpu.job` is already configured for it. It writes to `runs/emotion6_abaw80_vlm_subject_background_server/`, leaving both earlier runs untouched. Each eligible colour image retains both the complete-subject and background-light twins rather than selecting only one. Mask R-CNN weights are downloaded into `.cache/torch` on the first server run. If no qualifying instance is detected, a dependency-free SLIC-style superpixel region is used and marked in metadata; rectangular grid patches are never used.

## Outputs

The verified GPU run is in `runs/emotion6_abaw80_vlm_server/`:

- `samples.jsonl` and `interventions.jsonl`: reproducible source/twin manifests;
- `twins/` and `masks/`: cue-separated image pairs and exact edit masks;
- `predictions.jsonl` and `pair_metrics.csv`: pair-level outputs;
- `summary.json`: aggregate calibration, uncertainty, VA, and cue metrics;
- `contact_sheet.png` and `report.md`: visually checked qualitative/quantitative report;
- `human_validation_template.csv`: blinded human-rating protocol.

After annotators complete the template, summarize it with:

```bash
.venv/bin/affective-twins human-summary \
  --config configs/emotion6_abaw80_vlm_server.json \
  --annotations runs/emotion6_abaw80_vlm_server/human_validation_template.csv
```

## Verified earlier GPU audit

The deterministic seed-42 run uses 80 images: 60 balanced Emotion6 examples and 20 Aff-Wild2 frames with frame-level Action Unit labels. It evaluates 201 target twins and 76 matched controls. Confidence intervals are 95% bootstrap intervals.

| Cue | Target pairs | Directional success | Mean source-probability drop | Feature cosine |
|---|---:|---:|---:|---:|
| Colour/lighting | 60 | 95.0% | 0.070 | 0.975 |
| Scene context | 60 | 60.0% | 0.015 | 0.941 |
| Embedded-text conflict | 60 | 53.3% uncertainty increase | 0.001 | 0.993 |
| Facial action region | 21 | 61.9% | -0.005 | 0.951 |

The matched-control analysis supports selective colour sensitivity: the mean target-minus-control probability drop is 0.075 (95% CI [0.050, 0.104]) and is positive for 58 of 60 images. The AU-region result is mixed. Its target-minus-control difference is -0.047 (95% CI [-0.127, 0.013]), so the present facial ablations do not establish selective AU reliance. Fifty-nine non-face Emotion6 images remain explicitly ineligible rather than being forced through a face intervention.

## VLM status

`SmolVLMAdapter` implements constrained reporting of discrete emotion, valence--arousal, confidence, cue identification, visible evidence, and a literal caption. The full run evaluates all 201 target pairs. Of these, 188 pairs have valid constrained outputs (93.5%); invalid outputs are retained and excluded from VLM-specific aggregates rather than silently repaired. Cue-grounding accuracy is 35.1% (95% CI [28.7%, 42.0%]), and mean original--twin caption Jaccard similarity is 0.416. These results measure the behaviour of the 500M checkpoint and should not be generalized to VLMs as a class.

## Interpretation boundary

This is a complete executable research framework, not proof of perceptual causality. Current face edits precisely localize AU-related facial components, but still ablate visual evidence rather than synthesizing exact muscle activations. Object masks are limited to the instance categories recognized by the pretrained detector, and samples with no qualifying object are explicitly skipped. Context masks use a deterministic saliency prior rather than semantic segmentation. Human validation remains required before reporting the twins as perception-preserving causal interventions. The CAUSE composite in `summary.json` is explicitly unvalidated and should be treated as a run diagnostic, not a leaderboard score.

The original colour-only pilot and anonymous one-page PDF remain in `outputs/`, `abstract/`, and `submission/` for provenance.
