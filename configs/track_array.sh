#!/usr/bin/env bash
#SBATCH -p low
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 10:00:00
#SBATCH -J bats-track-array
#SBATCH -o logs/track.%A_%a.out
#SBATCH -e logs/track.%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRJ="$(cd "$SCRIPT_DIR/.." && pwd)"   # repo root
CFG_DIR="$PRJ/configs/generated"      # ← EDIT THIS if needed

mkdir -p "$PRJ/logs"

TOTAL=$(ls "$CFG_DIR"/*.yaml 2>/dev/null | wc -l)
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL ]]; then
    exit 0
fi

CFG=$(ls "$CFG_DIR"/*.yaml | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")

cd "$PRJ"
pixi run python -m src.tracking --config "$CFG"
