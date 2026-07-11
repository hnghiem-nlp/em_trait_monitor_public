#!/usr/bin/env bash
# Generic single-cell driver for the two-phase pipeline.
#
# Usage:
#   bash scripts/run_pipeline.sh <model_key> <perturbation> <seed> [<lr>]
#
# Example:
#   bash scripts/run_pipeline.sh llama3-8b bad_medical 42
#
# This runs:
#   Phase 1 (once per model): trait-direction extraction
#       python -m experiments.extract_directions --model <model_key>
#   Phase 2 (per (model, pert, seed) cell): LoRA SFT + per-checkpoint trajectory
#       python -m experiments.train_and_measure \
#           --model <model_key> --data-source <pert> --seed <seed> --lr <lr>
#
# Outputs:
#   Phase 1: results/directions/<model>/trait_directions.pt + probe_results.json
#   Phase 2: results/trajectories/<model>/<pert>/seed_<seed>/
#                trajectory.json + activations.pt + checkpoints/

set -euo pipefail

MODEL=${1:?usage: bash scripts/run_pipeline.sh <model_key> <pert> <seed> [<lr>]}
PERT=${2:?usage: bash scripts/run_pipeline.sh <model_key> <pert> <seed> [<lr>]}
SEED=${3:?usage: bash scripts/run_pipeline.sh <model_key> <pert> <seed> [<lr>]}
LR=${4:-4e-5}

cd "$(dirname "$0")/.."

# Phase 1: trait extraction (skip if already done)
DIRECTIONS_OUT="results/directions/${MODEL}/trait_directions.pt"
if [[ -f "$DIRECTIONS_OUT" ]]; then
    echo "[phase 1] trait directions already exist for ${MODEL}, skipping."
else
    echo "[phase 1] extracting trait directions for ${MODEL}..."
    python -m experiments.extract_directions --model "$MODEL"
fi

# Phase 2: training + checkpoint-level trajectory measurement
echo "[phase 2] training + measuring ${MODEL} / ${PERT} / seed=${SEED} / lr=${LR}..."
python -m experiments.train_and_measure \
    --model "$MODEL" \
    --data-source "$PERT" \
    --seed "$SEED" \
    --lr "$LR"

echo "Done. Trajectory at: results/trajectories/${MODEL}/${PERT}/seed_${SEED}/trajectory.json"
