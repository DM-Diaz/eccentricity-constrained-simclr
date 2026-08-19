#!/usr/bin/env python3
"""
Peripheral *occlusion* images (NO BLUR):
Overlay a feathered grey circular occluder at the foveal crop location
(gaze-contingent), leaving the rest of the frame unchanged.

Input:
  /user_data/dylandia/eb_processedData/forProcessing/2026_frames_224/<session>/frame_*.png

Fixations (from our foveal cropping):
  /user_data/dylandia/eb_processedData/fovea/2026_gaze_crops_224/all_fixations_224.json

Output:
  /user_data/dylandia/eb_processedData/periph/2026_periph_gaze/<session>/frame_*.png

Last checked: 8/10/26
"""

from __future__ import annotations
from pathlib import Path
import os, json, cv2, logging, argparse
import concurrent.futures as fut
from tqdm.auto import tqdm
import numpy as np

# ---------------------------- PATHS ------------------------------------
BASE_DIR  = Path(os.environ.get("EB_BASE", "/user_data/dylandia"))
FRAMES224 = BASE_DIR / "eb_processedData/forProcessing/2026_frames_224"
OUT_ROOT  = BASE_DIR / "eb_processedData/periph/2026_periph_gaze"
FIX_JSON  = BASE_DIR / "eb_processedData/fovea/2026_gaze_crops_224/all_fixations_224.json"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------- PARAMS -----------------------------------
# VERIFY VALUES BEFORE CHANGING! DEFAULTS SET ALREADY!
H = W = 224
GREY_VAL   = 128
FEATHER_K  = 15          # odd → soft edge
DEFAULT_RAD = 56         # 112 crop -> rad 56

MAX_PROCS  = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
IO_THREADS = int(os.environ.get("IO_THREADS", "8"))
VERBOSITY  = 1          # 0,1,2

logging.basicConfig(
    level=logging.WARNING - 10 * min(VERBOSITY, 2),
    format="%(levelname)s: %(message)s"
)

GREY = np.full((H, W, 3), GREY_VAL, np.uint8)

# ---------------------------- CLAMPING ----------------------------------
def clamp_disc(cx: int, cy: int, rad: int):
    """
    Clamp (cx,cy,rad) so the *entire disc* stays inside the image.
    This prevents edge-hugging/partial occluders when fixations are OOB.
    """
    rad = int(rad)
    if rad <= 0:
        rad = DEFAULT_RAD

    # keep rad feasible for current canvas
    max_rad = min((W - 1) // 2, (H - 1) // 2)
    if rad > max_rad:
        rad = max_rad

    cx = int(cx); cy = int(cy)

    # clamp center so [cx-rad, cx+rad] and [cy-rad, cy+rad] are in-bounds
    lo_x, hi_x = rad, (W - 1 - rad)
    lo_y, hi_y = rad, (H - 1 - rad)

    # if rad is absurd and collapses the range (shouldn't after max_rad), fallback
    if hi_x < lo_x or hi_y < lo_y:
        cx = W // 2
        cy = H // 2
        rad = min(DEFAULT_RAD, max_rad)
        return cx, cy, rad

    cx = int(np.clip(cx, lo_x, hi_x))
    cy = int(np.clip(cy, lo_y, hi_y))
    return cx, cy, rad

# ---------------------------- MASK CACHE --------------------------------
def _base_alpha_for_radius(rad: int) -> np.ndarray:
    """Return feathered alpha mask (H,W) centered at (112,112) for given rad."""
    if rad not in _base_alpha_for_radius._cache:
        hard = np.zeros((H, W), np.uint8)
        cv2.circle(hard, (W // 2, H // 2), int(rad), 255, -1)
        alpha = cv2.GaussianBlur(hard, (FEATHER_K, FEATHER_K), 0).astype(np.float32) / 255.0
        _base_alpha_for_radius._cache[rad] = alpha
    return _base_alpha_for_radius._cache[rad]
_base_alpha_for_radius._cache = {}

def alpha_at(cx: int, cy: int, rad: int) -> np.ndarray:
    """Return alpha mask (H,W,1) translated to center (cx,cy)."""
    base = _base_alpha_for_radius(rad)
    dx = cx - (W // 2)
    dy = cy - (H // 2)
    if dx == 0 and dy == 0:
        return base[..., None]
    M = np.array([[1, 0, dx],
                  [0, 1, dy]], dtype=np.float32)
    shifted = cv2.warpAffine(
        base, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0
    )
    return shifted[..., None]

# ---------------------------- I/O WORKER --------------------------------
def write_occluded(task):
    src, dst, cx, cy, rad = task
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        logging.warning("Corrupt PNG: %s", src)
        return 0

    # final safety clamp (cheap, prevents silent nonsense)
    cx, cy, rad = clamp_disc(cx, cy, rad)

    a3 = alpha_at(cx, cy, rad)  # (H,W,1) float32 in [0,1]
    out = (a3 * GREY + (1.0 - a3) * img).astype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), out)
    return 1

# ---------------------------- SESSION WORKER ----------------------------
def process_session(args):
    sess_dir, fix_map, force = args
    sess = sess_dir.name
    frame_paths = sorted(sess_dir.glob("frame_*.png"))
    if not frame_paths:
        return 0, 0, 0  # written, missing_fix, total

    out_dir = OUT_ROOT / sess
    out_dir.mkdir(parents=True, exist_ok=True)

    # idempotent guard (skip only if it really looks complete)
    if not force:
        first_out = out_dir / frame_paths[0].name
        last_out  = out_dir / frame_paths[-1].name
        if first_out.exists() and last_out.exists():
            return len(frame_paths), 0, len(frame_paths)

    sess_fix = fix_map.get(sess, {})
    tasks = []
    missing = 0

    for fp in frame_paths:
        fid = fp.stem
        coords = sess_fix.get(fid)

        if coords is None:
            cx = cy = 112
            rad = DEFAULT_RAD
            missing += 1
        else:
            cx, cy, rad, *_ = coords
            cx, cy, rad = clamp_disc(cx, cy, rad)

        dst = out_dir / fp.name
        tasks.append((fp, dst, cx, cy, rad))

    written = 0
    with fut.ThreadPoolExecutor(max_workers=IO_THREADS) as tp:
        for n in tp.map(write_occluded, tasks, chunksize=64):
            written += n

    if missing:
        logging.warning("%s: %d/%d frames missing fixations → center fallback",
                        sess, missing, len(frame_paths))

    return written, missing, len(frame_paths)

# ---------------------------- MAIN --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Overwrite outputs for sessions even if they already exist.")
    args = ap.parse_args()

    if not FIX_JSON.is_file():
        raise FileNotFoundError(f"Fixation JSON not found: {FIX_JSON}")

    fix_map = json.loads(FIX_JSON.read_text())

    sessions = [p for p in FRAMES224.iterdir() if p.is_dir()]
    if not sessions:
        raise RuntimeError(f"No session directories found in: {FRAMES224}")

    total_written = 0
    total_missing = 0
    total_frames  = 0

    with fut.ProcessPoolExecutor(max_workers=MAX_PROCS) as ex:
        args_iter = ((s, fix_map, args.force) for s in sessions)
        for written, missing, nframes in tqdm(
            ex.map(process_session, args_iter),
            total=len(sessions),
            desc="sessions"
        ):
            total_written += written
            total_missing += missing
            total_frames  += nframes

    print(f"✓ Peripheral occlusion complete: {total_written:,} PNGs → {OUT_ROOT}")
    if total_frames:
        print(f"  Missing fixations: {total_missing:,}/{total_frames:,} ({total_missing/total_frames:.1%})")
    print(f"  Grey={GREY_VAL}, feather kernel={FEATHER_K}, default rad={DEFAULT_RAD}")
    print("  (No blur applied.)")

if __name__ == "__main__":
    main()
