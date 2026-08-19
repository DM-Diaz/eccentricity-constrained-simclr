#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=mind-1-15
#SBATCH --mem=32G
#SBATCH --time=120:00:00
#SBATCH --job-name=perttm_vggface2_cls
#SBATCH -o perttm_vggface2_cls_%j.out
#SBATCH -e perttm_vggface2_cls_%j.err
#SBATCH --chdir=/user_data/dylandia/eb_notebooks/SimCLR-master
#SBATCH --exclude=mind-0-[18,20,22,24],mind-1-[7,9,11,19]

set -euo pipefail

module load anaconda3-2023.03
source /opt/anaconda3-2023.03/etc/profile.d/conda.sh
conda activate new_env_gpu

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Prevent BLAS thread blowups (DataLoader already parallelizes)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -----------------------------
# Paths
# -----------------------------
DATA_ROOT="/lab_data/hendersonlab/datasets/VGG-Face2/data/train"

# Put the fixed script on the cluster at this path:
SCRIPT="/user_data/dylandia/eb_notebooks/vggface2_identity_classifier_v2.py"

# Select backbone arch to match your SimCLR checkpoint:
ARCH="resnet18"   # resnet18 | resnet50

# Stream-specific SimCLR checkpoint (baseline/fovea/periph):
CKPT="/user_data/dylandia/eb_notebooks/SimCLR-master/runs/PeriphTTM_SimCLR_Feb042026/checkpoint_0120.pth.tar"

# Output
RUN_ROOT="/user_data/dylandia/eb_processedData/vggface2_classifier_runs/Periph_TTM"
RUN_DIR="${RUN_ROOT}/${ARCH}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

# -----------------------------
# Method-matched hyperparams
# effective batch = 60 * 2 = 120
# -----------------------------
MICRO_BS=60
GRAD_ACCUM=2

python -u "$SCRIPT" \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CKPT" \
  --out-dir "$RUN_DIR" \
  --arch "$ARCH" \
  --stage both \
  --probe-epochs 20 \
  --finetune-epochs 100 \
  --unfreeze layer4 \
  --select-metric top1 \
  --label-smoothing 0.1 \
  --probe-lr 1e-3 \
  --finetune-lr 3e-4 \
  --weight-decay 0.01 \
  -b "$MICRO_BS" \
  -j 4 \
  --prefetch-factor 4 \
  --dataloader-timeout 0 \
  --mp-context fork \
  --allow-truncated-images \
  --split-cache "$RUN_DIR/split_indices.npz"

