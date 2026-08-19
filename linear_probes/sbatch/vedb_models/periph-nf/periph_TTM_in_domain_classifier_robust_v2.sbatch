#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=periphTTM_indomain_cls
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=120:00:00
#SBATCH -o periphTTM_indomain_%j.out
#SBATCH -e periphTTM_indomain_%j.err
#SBATCH --exclude=mind-0-[18,20,22,24],mind-1-[7,9,11],mind-1-19

set -euo pipefail

module load anaconda3-2023.03
source /opt/anaconda3-2023.03/etc/profile.d/conda.sh
conda activate new_env_gpu

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

DATA_ROOT="/user_data/dylandia/eb_processedData/forProcessing/ttm_frames/2026_periph_ttm_frames"
MANIFEST="/user_data/dylandia/eb_processedData/in_domain_core17_manifest.csv"
CKPT="/user_data/dylandia/eb_notebooks/SimCLR-master/runs/PeriphTTM_SimCLR_Feb042026/checkpoint_0120.pth.tar"
LABEL_COL="task"

# Upload the updated script to this path on the cluster:
SCRIPT="/user_data/dylandia/eb_notebooks/SimCLR-master/in_domain_classifier_simclr_staged_robust_v3.py"

RUN_NAME="periph_TTM_indomain_valselect_testonce_${SLURM_JOB_ID}"
OUT_DIR="/user_data/dylandia/eb_processedData/in_domain_classifier_runs/New_Periph_TTM_Resnet18"

python -u "$SCRIPT" \
  --data-root "$DATA_ROOT" \
  --manifest-csv "$MANIFEST" \
  --checkpoint "$CKPT" \
  --label-col "$LABEL_COL" \
  --finetune-splits train \
  --eval-splits val \
  --final-eval-splits test \
  --require-ok \
  --arch resnet18 \
  --stage both \
  --probe-epochs 20 \
  --probe-lr 1e-3 \
  --bn-eval-during-probe \
  --unfreeze layer4 \
  --finetune-epochs 100 \
  --finetune-lr 6e-4 \
  --unseen-eval-policy drop \
  --select-metric macro_f1 \
  -b 256 \
  -j 8 \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 4 \
  --dataloader-timeout 0 \
  --fp16 \
  --seed 1337 \
  --out-dir "$OUT_DIR" \
  --run-name "$RUN_NAME"
