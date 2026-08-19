#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=simclr_manifest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=120:00:00
#SBATCH -o simclr_fov_%j.out
#SBATCH -e simclr_fov_%j.err
#SBATCH --exclude=mind-0-[18,20,22,24],mind-1-[7,9,11],mind-1-19

module load anaconda3-2023.03
source /opt/anaconda3-2023.03/etc/profile.d/conda.sh
conda activate new_env_gpu

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# prevent CPU oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ulimit -n 1048576 || true

cd /user_data/dylandia/eb_notebooks/SimCLR-master || exit 1

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Python: $(which python)"
nvidia-smi || true
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

python -u run_updated_fixed_v5.py \
  -data /user_data/dylandia/eb_processedData/fovea/2026_gaze_crops_224 \
  --manifest-csv /user_data/dylandia/eb_processedData/all_manifests_combined_synmapped_with_split_fovea.csv \
  --include-splits train \
  --path-template '{data_root}/{session}/{filename}' \
  --session-col session \
  --filename-col filename \
  --split-col split \
  --require-ok \
  -j 8 \
  --mp-context spawn \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 4 \
  --dataloader-timeout 0 \
  --gpu-aug \
  --fp16-precision \
  -b 512 \
  --epochs 120 \
  --lr 0.0006
