# Counterfactual Affective Twins

An end-to-end, standalone framework for auditing whether affect models respond to the same inspectable evidence they report using.

The framework can create matched twins that alter one cue family at a time:

- `color_lighting`: luminance-preserving chroma removal over complete Mask2Former/SegFormer semantic entities (person, object, wall, floor, or another surface), with Mask R-CNN subject/background fallback and no rectangular patch intervention;
- `facial_action_region`: expanded landmark-contoured eyebrow, eye, or mouth masks with strong blur/pixelation; Aff-Wild2 examples additionally retain frame-level AU support and identify inactive-region controls;
- `scene_context`: panoptic-foreground-preserving background chroma/detail attenuation;
- `embedded_text`: optional OCR-localized text removal for datasets containing genuine text; disabled in the present Emotion6 + Aff-Wild2 audit because the selected images contain no text evidence.

The primary protocol is report-conditioned. For this dataset, the target VLM must select among three evidence-bearing cue families: colour/lighting, facial action region, and scene context. It names cue-specific visible evidence on the untouched image: one object or surface for colour, one of brows/eyes/mouth for facial evidence, or a background setting for context. That exact answer must match an eligible segmented entity, landmark component, or context mask. Ambiguous object matches are accepted only when the VLM selects the same physical region under multiple shuffled option orders. No unrelated region is substituted when grounding fails. Only afterward is the reported evidence manipulated. Unreported cue interventions and nearest-area regions within the reported cue family are retained as explicit comparators. The same target VLM is queried again to measure original-class probability change, prediction flips, entropy, cue retention, and reported-target minus control effects. Every twin has a mask and a JSONL manifest entry; invalid or ungroundable reports remain explicit.

## Project boundary

All code, environments, model assets, generated twins, masks, checkpoints, manifests, and reports are inside this folder. Existing Emotion6 annotations and the MediaPipe task file are read-only inputs referenced by the default configuration; no files are written to other thesis projects.

Emotion6 targets are derived from the seven-bin human annotation distribution, not from the Flickr retrieval folder. The complete `folder/filename` key is used to join annotations, preventing collisions between repeated basenames such as `anger/55.jpg` and `sadness/55.jpg`. The six-way model excludes neutral-plurality images, trains against the human distribution renormalized over the six modeled emotions, retains the original neutral-inclusive distribution, and stores the folder label separately as `nominal_emotion`. The portable audit set contains the 10 highest unique human-plurality probabilities per modeled emotion; its minimum class-specific probabilities are recorded in `data/audit80/summary.json`. These 60 images are held out from evaluator training.

## Reproduce

```bash
cd /Users/rijju/Documents/PhD_thesis/visual-emotion-cue-audit
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[face,vlm,dev]'

export TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/huggingface"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"

.venv/bin/python scripts/build_audit80_manifest.py
.venv/bin/affective-twins doctor
.venv/bin/affective-twins train
.venv/bin/affective-twins generate
.venv/bin/affective-twins evaluate
.venv/bin/affective-twins report
```

Use `.venv/bin/affective-twins all --config configs/emotion6_abaw80_exact_qwen25vl3b_server.json` after a checkpoint exists to run the stronger, lower-memory Qwen2.5-VL-3B report-conditioned audit. The GPU workflow is available in `jobs/full_vlm_gpu.job`.

The exact-grounding configurations are `configs/emotion6_abaw80_exact_qwen25vl3b_server.json` and `configs/emotion6_abaw80_exact_smolvlm500m_server.json`. Their outputs are isolated in `runs/emotion6_abaw80_exact_qwen25vl3b_server/` and `runs/emotion6_abaw80_exact_smolvlm500m_server/`, leaving every earlier run untouched. Mask2Former and SegFormer propose complete objects and background surfaces. In the server configurations, these segmenters run on CPU while the VLM uses automatic GPU placement, avoiding simultaneous GPU residency. A colour report must name one of those entities; a face report must say brows, eyes, or mouth; and a context report targets the subject-preserving full background. Embedded text is neither offered to the VLM nor generated as an intervention in these configurations. A failed or order-sensitive grounding is recorded as ineligible rather than replaced by a locator-selected target.

To compare the stronger VLM with the compact baseline on the server, submit:

```bash
sbatch jobs/compare_vlms_gpu.job
```

This runs both audits sequentially and writes `comparison.md`, `comparison.csv`, and `comparison.json` to `runs/vlm_exact_grounding_comparison/`. Each model is evaluated against its own report and its own induced twins; the script does not pool unlike interventions.

## Report-conditioned outputs

Each exact-grounding run contains:

- `original_reports.jsonl`: the immutable pre-intervention emotion, cue, evidence, caption, and report validity used to condition twin generation;
- `interventions.jsonl`: the selected reported-cue target, its grounding decision, unreported-cue comparators, matched-region controls, masks, and explicit skips;
- `predictions.jsonl` and `pair_metrics.csv`: same-model pre/post responses and independent-evaluator diagnostics;
- `summary.json`: report-conditioned faithfulness, target coverage, prediction flips, and target-minus-control effects;
- `twins/`, `masks/`, `contact_sheet.png`, `report.md`, and `human_validation_template.csv`: image artefacts and validation materials.

The exact stored report is reused during evaluation; it is not regenerated after the intervention target has been observed.

## Earlier verified outputs

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
  --config configs/emotion6_abaw80_vlm_subject_background_server.json \
  --annotations runs/emotion6_abaw80_vlm_report_conditioned_server/human_validation_template.csv
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

The VLM adapter implements constrained reporting of discrete emotion, valence--arousal, confidence, cue identification, cue-specific visible evidence, and a literal caption. The primary model is `Qwen/Qwen2.5-VL-3B-Instruct`; `HuggingFaceTB/SmolVLM-500M-Instruct` is retained as a compact baseline. Invalid outputs and evidence that cannot be grounded exactly are retained as audit failures rather than silently repaired. The earlier 500M result below predates exact evidence grounding and should not be used as the final claim: it evaluated 201 target pairs, obtained 188 valid constrained outputs (93.5%), cue-family grounding accuracy of 35.1% (95% CI [28.7%, 42.0%]), and caption Jaccard similarity of 0.416.

## Interpretation boundary

This is a complete executable research framework, not proof of perceptual causality. Current face edits localize and strongly blur/pixelate AU-related brows, eyes, or mouth evidence, but do not synthesize anatomically exact muscle activations. Object masks are limited to the categories recognized by the pretrained panoptic and semantic segmenters, and reports naming an unavailable entity are explicitly skipped. Context intervention attenuates the segmented scene background while preserving detected foreground subjects. Human validation remains required before reporting the twins as perception-preserving causal interventions. The CAUSE composite in `summary.json` is explicitly unvalidated and should be treated as a run diagnostic, not a leaderboard score.

The original colour-only pilot and anonymous one-page PDF remain in `outputs/`, `abstract/`, and `submission/` for provenance.
