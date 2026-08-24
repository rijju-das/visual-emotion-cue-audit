#!/bin/bash

set -euo pipefail

PROJECT_DIR=/home/rdas/visual-emotion-cue-audit
mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"
sbatch jobs/full_vlm_gpu.job
