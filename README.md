# Lagattuta_YOLOv11_SORT
A bat detection and tracking algorithm using YOLOv11 and DeepSORT to count bats in greyscale thermal video

This text document describes the workflow to run one of the YOLOv11n models trained with data from the NSF Center for Pandemic Insights. For the purposes of this workflow, the model will be referred to as ModelX, as there are several models with differing training data subsets that this workflow can be used for.

In the box folder there are the following directories and files:

Lagattuta_YOLOv11_SORT/
→ src/
	→ tracking.py [tracks bats with SORT]
	→ detection.py [detects bats with YOLOv11]
	→ bg_subtract_new.py [subtracts background elements]
→ configs/
	→ make_configs.sh [generates yaml configs for model runs]
	→ track_array.sh [runs a specified model on a set of generated configs]
	→ videos.list [defines videos to run and associated regions of interest]
	→ generated/
		→ [generated yaml files will be placed here]
→ models/
    → ALL_noaug [model trained on ALL training data without image augmentation]
        → weights/
			→ best.pt [best weights for associated model]
	→ [additional models can be placed here]
→ pixi.toml 
→ pixi.lock
→ videos/
	→ [place videos to run here]
→ results/
	→ counts/
		→ [count csv files will populate here]
	→ annotations/
		→ [annotated videos will populate here]
	→ README.txt

1) Environment setup:

This project uses pixi.

To install dependencies:
pixi install

2) video upload and setup:

Upload all videos to be run into the videos directory.

Edit /configs/videos.list to have the following structure:
Site abbreviation|video name|ROI coordinates (x_min, y_min, x_max, y_max)

For example:
CPO|C2.1.1[Val2]_grey.mov|0.0 0.42 1.0 0.52

Add a new row for every video to be run

3) edit make_configs.sh:

#!/usr/bin/env bash
set -euo pipefail
PRJ= #project directory /Lagattuta_YOLOv11_SORT
MODELS=(ALL_noaug) #add models here if applicable
VARIANTS=("BGon_ROIon") #add variants here if applicable
COUNTS_DIR="$PRJ/results/counts"
ANN_DIR="$PRJ/results/annotations"
CFG_DIR="$PRJ/configs/generated"
mkdir -p "$COUNTS_DIR" "$ANN_DIR" "$CFG_DIR"
rm -f "$CFG_DIR"/*.yaml
while IFS='|' read -r site file coords; do
  file=$(echo "$file" | tr -d ' ')
  coords=( $coords )
  for m in "${MODELS[@]}"; do
    for v in "${VARIANTS[@]}"; do
      safe=$(echo "${m}_${site}_${file}_${v}" | tr '[]() ' '____')
      cfg="$CFG_DIR/${safe}.yaml"
      model_file="$PRJ/models/$m/weights/best.pt"
      csv="$COUNTS_DIR/${safe}.csv"
      ann="$ANN_DIR/${safe}"
      bg_flag=false
      roi_flag=false
      [[ "$v" == BGon_* ]] && bg_flag=true
      [[ "$v" == *_ROIon ]] && roi_flag=true
      {
        echo "tracking:"
        echo "  detection_confidence: 0.15"
        echo "  model_file: \"$model_file\""
        echo "  results_path: \"$csv\""
        echo "  imgsz: 1280"
        echo "  sort:"
        echo "    iou_threshold: 0.20"
        echo "    max_age: 30"
        echo "    min_hits: 4"
        echo "  video_files:"
        echo "  - amplification: 1.0"
        echo "    path: \"$PRJ/videos/$file\""
        if $roi_flag; then
          echo "    tracking_region:"
          printf "    - %s\n" "${coords[@]}"
        else
          echo "    tracking_region:"
          echo "    - 0.0"
          echo "    - 0.0"
          echo "    - 1.0"
          echo "    - 1.0"
        fi
        echo "  write_annotated_frames: \"$ann\""
        echo "background_subtraction:"
        if $bg_flag; then
          echo "  enabled: true"
          echo "  window_size: 30"
	else
	  echo "  enabled: false"
        fi
      } > "$cfg"
    done
  done
done < "$PRJ/configs/videos.list"
echo "Generated $(ls -1 $CFG_DIR/*.yaml | wc -l) configs."

4) bash make_configs.sh
5) edit track_array.sh

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
#SBATCH --array=0-1079 #change for the number of configs you want to run within generated
set -euo pipefail
module load conda
export PYTHONPATH=""
source /home/gjospin/.bashrc
conda activate /quobyte/ckreudergrp/gjospin/envs/bats-farm-gpu-py39
# go to project root with pixi.toml
cd /quobyte/ckreudergrp/gjospin/2025_lagattuta_bats-farm-gpu
CFG_DIR=  #config directory /Lagattuta_YOLOv11_SORT/configs/generated
TOTAL=$(ls "$CFG_DIR"/*.yaml 2>/dev/null | wc -l)
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL ]]; then
    exit 0
fi
CFG=$(ls "$CFG_DIR"/*.yaml | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
/home/slaga/.pixi/bin/pixi run python -m src.tracking --config "$CFG"
