#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRJ="$(cd "$SCRIPT_DIR/.." && pwd)"

MODELS=(PB_noaug)
VARIANTS=("BGon_ROIon")

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
        echo "  detection_confidence: 0.10"
        echo "  model_file: \"$model_file\""
        echo "  results_path: \"$csv\""
        echo "  imgsz: 1280"
        echo "  sort:"
        echo "    iou_threshold: 0.10"
        echo "    max_age: 30"
        echo "    min_hits: 5"
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
