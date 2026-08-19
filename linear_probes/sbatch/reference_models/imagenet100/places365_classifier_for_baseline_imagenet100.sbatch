#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=mind-0-[18,20,22,24],mind-1-[7,9,11],mind-1-19
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=120:00:00
#SBATCH --job-name=places365_cls_r18
#SBATCH -o places365_cls_%j.out
#SBATCH -e places365_cls_%j.err

set -euo pipefail

module load anaconda3-2023.03
source /opt/anaconda3-2023.03/etc/profile.d/conda.sh
conda activate new_env_gpu

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=1337
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- Script + data ---
SCRIPT="/user_data/dylandia/eb_notebooks/places365_scene_classifier_imagenet_v2.py"
DATA_ROOT="/lab_data/tarrlab/common/datasets/Places365/train"
CKPT="/lab_data/hendersonlab/ckpts/checkpoint_120-resnet18-simclr-imagenet100.ckpt"

# --- Comparable config to VGGFace2 ---
ARCH="resnet18"
TRAIN_FRAC="0.8"
SEED="1337"
IMG_SIZE="224"

PROBE_EPOCHS="20"
FINETUNE_EPOCHS="100"
PROBE_LR="1e-3"
FINETUNE_LR="3e-4"

WEIGHT_DECAY="0.01"
LABEL_SMOOTH="0.1"

BATCH_SIZE="60"
WORKERS="4"

SELECT_METRIC="top1"
EARLY_STOP_METRIC="loss"
EARLY_STOP_PATIENCE="10"

# --- Output ---
OUT_BASE="/user_data/dylandia/eb_processedData/places365_classifier_runs/base_imagenet100"
RUN_NAME="net100_${ARCH}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
RUN_DIR="${OUT_BASE}/${RUN_NAME}"
mkdir -p "$RUN_DIR"

SPLIT_CACHE="/user_data/dylandia/eb_processedData/places365_classifier_runs/base/resnet18_20260203_043434_306387/split_indices.npz"

python -u "$SCRIPT" \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CKPT" \
  --out-dir "$RUN_DIR" \
  --arch "$ARCH" \
  --train-frac "$TRAIN_FRAC" \
  --seed "$SEED" \
  --img-size "$IMG_SIZE" \
  --stage both \
  --probe-epochs "$PROBE_EPOCHS" \
  --finetune-epochs "$FINETUNE_EPOCHS" \
  --unfreeze layer4 \
  --probe-lr "$PROBE_LR" \
  --finetune-lr "$FINETUNE_LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --label-smoothing "$LABEL_SMOOTH" \
  -b "$BATCH_SIZE" \
  -j "$WORKERS" \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 4 \
  --dataloader-timeout 0 \
  --mp-context fork \
  --select-metric "$SELECT_METRIC" \
  --early-stop-metric "$EARLY_STOP_METRIC" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --max-confusions 2000 \
  --split-cache "$SPLIT_CACHE"
