#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=map_gaze_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=144:00:00
#SBATCH -o map_gaze_gpu_%j.out
#SBATCH -e map_gaze_gpu_%j.err
#SBATCH --exclude=mind-1-15

module load anaconda3-2023.03
source /opt/anaconda3-2023.03/etc/profile.d/conda.sh
conda activate new_env_gpu

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Prevent CPU oversubscription (helps stability on shared nodes)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ulimit -n 1048576 || true

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Python: $(which python)"
nvidia-smi || true

# --- Working dir / script paths ---
WORKDIR="/user_data/dylandia/eb_notebooks/NeuroFovea_PyTorch"
cd "$WORKDIR" || { echo "ERROR: cannot cd to $WORKDIR"; exit 1; }

PYTHON_BIN="$(which python)"
METAMER_PY="$WORKDIR/Metamer_Transform_fixed_v2_fastbatch.py"
[[ -f "$METAMER_PY" ]] || { echo "ERROR: missing $METAMER_PY"; exit 1; }

# --- Runtime config ---
TIMEOUT_SEC=1800

IMG_SIZE=224
CENTER=$((IMG_SIZE/2))

# Fixation coordinate grid used in the CSV.
# If your CSV is truly in 512px coords, set FIX_COORD_SIZE=224.
# If values look like fx~0..223 and r~56, keep this at 224.
FIX_COORD_SIZE=224

BASE_DIR="/user_data/dylandia"
SESSION_ROOT="/lab_data/hendersonlab/datasets/vedb_video/sessions"
FRAME_ROOT="$BASE_DIR/eb_processedData/periph/2026_periph_gaze"
FIX_ROOT="$BASE_DIR/eb_processedData/fovea/2026_gaze_crops_224"
OUT_BASE="$BASE_DIR/eb_processedData/forProcessing/ttm_frames/2026_periph_ttm_frames/"

SCALE="0.4"

# Warm RF cache once (fast; avoids first-frame slowdown)
"$PYTHON_BIN" "$METAMER_PY" --scale "$SCALE" --precache


# Build session list from FRAME_ROOT (faster than scanning SESSION_ROOT)
mapfile -t SESSIONS < <(find "$FRAME_ROOT" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)

if [[ ${#SESSIONS[@]} -eq 0 ]]; then
  echo "ERROR: no session directories found under $FRAME_ROOT"
  exit 1
fi

# If running as a job array, process exactly one session; otherwise loop all.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  session="${SESSIONS[$SLURM_ARRAY_TASK_ID]}"
  SESSIONS=( "$session" )
  echo "Array task ${SLURM_ARRAY_TASK_ID}: session=$session"
fi

for session in "${SESSIONS[@]}"; do
  echo "==> SESSION $session"

  crop_dir="$FRAME_ROOT/$session"
  out_dir="$OUT_BASE/$session"
  mkdir -p "$out_dir"

  # Fixation CSV (historical filename; may still be 224-space coords)
  csv_fix="$FIX_ROOT/$session/${session}_fixations_512.csv"

  echo "    frames: $crop_dir"
  echo "    fixcsv: $csv_fix"
  echo "    out:    $out_dir"

  # Run once per session (models + RF cache loaded once). This is the major speed-up.
  # Use --skip_existing to resume without recomputing.
  set +e
  timeout "$TIMEOUT_SEC" "$PYTHON_BIN" "$METAMER_PY" \
    --image_dir "$crop_dir" \
    --pattern "frame_*.png" \
    --output "$out_dir" \
    --scale "$SCALE" \
    --image_size "$IMG_SIZE" \
    --fixations_csv "$csv_fix" \
    --fix_coord_size "$FIX_COORD_SIZE" \
    --fallback_center \
    --skip_existing \
    --plain_names \
    >"$out_dir/session_run.log" 2>&1
  exit_code=$?
  set -e

  if [[ $exit_code -ne 0 ]]; then
    echo "    ✖ session failed: $session (exit=$exit_code)"
    tail -n 80 "$out_dir/session_run.log" || true
  else
    echo "    ✓ session done: $session"
  fi
done
