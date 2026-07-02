<img width="288" height="205.8" alt="image" src="https://github.com/user-attachments/assets/fb2843a3-4b14-4789-a6f4-ca8ac8446569" /> <img width="308.7" height="205.8" alt="image" src="https://github.com/user-attachments/assets/ef0f0ea3-fb66-43b3-9b2a-d650d7e0e9fb" />
<img width="197.5" height="205.8" alt="Screenshot 2026-07-01 at 4 38 13 PM" src="https://github.com/user-attachments/assets/ac18267c-f295-4cb1-862d-f1ea89ad29a5" />


# Bat Counting with YOLOv11-SORT

This project uses a specialized machine learning model to track and count flying bats in thermal video. The model pipeline uses YOLO11 (You Only Look Once), an image detection deep learning model, and SORT (Simple Online and Realtime Tracking), a multi-object tracking algorithm, to recognize and track bat flight trajectories. Each unique trajectory recognized by the model in a given video is counted as one bat.

Bat population counts are used widely for conservation management and wildlife disease research. Bringing bat counting models into edge computing will allow for near-real-time surveillance of population dynamics for a wide range of bat species in diverse environments.

# Edge Computing Challenge

The goal of this challenge is to adapt an existing thermal-video bat detection and tracking pipeline for edge computing. You will work with our pretrained YOLO weights, sample thermal videos from real deployments, and our existing tracking/counting pipeline, then optimize the workflow for faster, lighter-weight deployment on constrained hardware.

# Sample Videos and Counts

To see how this model works, please view the example videos with annotations from running the bat counting model in (`examples/sample_annotations/`). To guide your work in adapting these models to edge computing, we've provided sample, untracked videos in (`videos/`) as well as their associated counts from the PB_noaug model in (`examples/sample_counts.csv`).

# Introduction to the Existing Pipeline

This document describes the workflow to run one of the YOLO11 models (trained with data from the NSF Center for Pandemic Insights) and track and count bats with SORT. This workflow is presently developed for being run with an HPC (High Performance Cluster), but can also be locally run using Pixi. 

The eventual goal for this project would be to have an edge computing device run the model automatically with the user only needed to define a region of interest bounding box (as seen in configs/videos.list) for a specific filming location. The device would collect nightly population data from the thermal camera and transmit bat count data rather than researchers having to collect video data, download, and process it for each deployment.

---

## Edits Before Running

Before running, update the flagged files: search for `EDIT THIS` in this README and repo code files.

---

## Project Structure

```
Bat-Counting-YOLOv11-SORT/
├── src/
│   ├── tracking.py              # tracks bats with SORT
│   ├── detection.py             # detects bats with YOLOv11
│   ├── bg_subtract_new.py       # subtracts background elements
│   └── utils/
│       └── get_args.py          # loads and parses YAML config files
│
├── sort/
│   └── sort.py                  # SORT tracking algorithm implementation
│
├── configs/
│   ├── make_configs.sh          # generates yaml configs for model runs
│   ├── track_array.sh           # runs model on generated yaml configs
│   ├── videos.list              # defines videos + ROIs
│   └── generated/               # generated yaml configs
│
├── models/
│   └── PB_noaug/
│       └── weights/
│           └── best.pt          # model weights
│
├── videos/                      # place input videos here
│
├── results/
│   ├── counts/                  # output CSV counts
│   └── annotations/             # annotated videos
│
├── pixi.toml                    # environment + dependencies
├── pixi.lock                    # locked dependency versions
└── README.md
```

---

## 1) Environment Setup

This project uses **pixi**.

Install dependencies:

```bash
pixi install
```

---

## 2) Video Upload and Setup

1. Upload videos into:

```
videos/
```

2. Edit:

```
configs/videos.list
```

Format:

```
Site|video_name|x_min y_min x_max y_max
```

Example:

```
CPO|C2.1.1[Val2]_grey.mov|0.0 0.42 1.0 0.52
```

Add one row per video.

---

## 3) Edit `make_configs.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# automatically resolve repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRJ="$(cd "$SCRIPT_DIR/.." && pwd)"

MODELS=(ALL_noaug)                   # ← EDIT THIS [OPTIONAL]: change if using other models
VARIANTS=("BGon_ROIon")              # ← EDIT THIS [OPTIONAL]: change if testing variants

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

echo "Generated $(ls -1 "$CFG_DIR"/*.yaml 2>/dev/null | wc -l) configs."
```

---

## 4) Generate Configs

```bash
bash make_configs.sh
```

---

## 5) Edit `track_array.sh`

```bash id="u3qv8l"
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

#SBATCH --array=0-99   # ← EDIT THIS: match number of generated configs

set -euo pipefail

# automatically resolve repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRJ="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG_DIR="$PRJ/configs/generated"      # ← EDIT THIS [OPTIONAL]: change if using different config directory

mkdir -p "$PRJ/logs"

TOTAL=$(ls "$CFG_DIR"/*.yaml 2>/dev/null | wc -l)

if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL ]]; then
    exit 0
fi

CFG=$(ls "$CFG_DIR"/*.yaml | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")

cd "$PRJ"
pixi run python -m src.tracking --config "$CFG"
```

---

## Summary Workflow

1. Install dependencies (`pixi install`)
2. Add videos
3. Edit `videos.list`
4. Update required fields (`EDIT THIS`)
5. Run config generation
6. Submit SLURM array job

---
