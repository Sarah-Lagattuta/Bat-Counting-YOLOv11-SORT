#!/usr/bin/env bash
#SBATCH -p low
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 10:00:00
#SBATCH -J bats-track-array
#SBATCH -o /quobyte/ckreudergrp/slaga/bats_thermal/results/v7_logs/track.%A_%a.out
#SBATCH -e /quobyte/ckreudergrp/slaga/bats_thermal/results/v7_logs/track.%A_%a.err
#SBATCH --array=0-1079

set -euo pipefail

module load conda
export PYTHONPATH=""
source /home/gjospin/.bashrc
conda activate /quobyte/ckreudergrp/gjospin/envs/bats-farm-gpu-py39

# go to project root with pixi.toml
cd /quobyte/ckreudergrp/gjospin/2025_lagattuta_bats-farm-gpu

CFG_DIR=/quobyte/ckreudergrp/slaga/bats_thermal/configs/grey_generated

TOTAL=$(ls "$CFG_DIR"/*.yaml 2>/dev/null | wc -l)
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL ]]; then
    exit 0
fi

CFG=$(ls "$CFG_DIR"/*.yaml | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")

/home/slaga/.pixi/bin/pixi run python -m src.tracking --config "$CFG"
