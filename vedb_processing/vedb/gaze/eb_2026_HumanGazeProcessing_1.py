#!/usr/bin/env python3
# Last checked: 8/11/26
"""
map_gaze_frames_cluster.py
==========================

Maps Pupil-Labs gaze timestamps to video-frame indices and stores the result
in Apache Parquet files—one Parquet per session. Designed for **batch** use on
CMU's MIND cluster under SLURM.

V5 features implemented (final final)
--------------
• Robust input validation (video FPS/frames > 0, >= 4 sync points, R² check, NaN/shape checks).
• Multiprocessing using the safest context for SLURM (`spawn`). Worker count is
  capped at `min(SLURM_CPUS_ON_NODE, os.cpu_count())`.
• Atomic Parquet writes to avoid race conditions (as suggested by Jaime M., thx Jaime).
• Detailed logging for every skipped / failed session.
• New CLI option `--session-list` (defaults to
  `/user_data/dylandia/eb_processedData/big_dirs.txt`) so the script processes
  **only** sessions that appear in that text file.

Usage examples
--------------
```bash
# raw per-sample rows, only the sessions in big_dirs.txt
sbatch --cpus-per-task=16 --mem=48G --time=72:00:00 ./map_gaze_frames_cluster.py

# one-row-per-frame mean, custom list of sessions
python map_gaze_frames_cluster.py --summary --session-list my_subset.txt
"""
from __future__ import annotations
from pathlib import Path
import os, yaml, cv2, argparse, logging, numpy as np, pandas as pd
import concurrent.futures as fut
from multiprocessing import get_context
from tqdm.auto import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────
BASE_DIR   = Path("/lab_data/hendersonlab/datasets")
GAZE_DIR   = BASE_DIR / "vedb_gaze"
VIDEO_DIR  = BASE_DIR / "vedb_video/sessions"
DEFAULT_LIST = Path("/user_data/dylandia/eb_processedData/session_frame_counts.txt")  # CHANGE THIS
OUT_DIR    = Path("/user_data/dylandia/eb_processedData/forProcessing/2026_frame_gaze_maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VERBOSITY  = int(os.getenv("VERBOSITY", 1))
logging.basicConfig(level=logging.WARNING - 10*min(VERBOSITY, 2),
                    format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── HELPER FUNCTIONS ──────────────────────────────────────────────────
def read_yaml(fp: Path):
    with open(fp, "r") as f:
        return yaml.safe_load(f)

def video_meta(fp: Path):
    cap = cv2.VideoCapture(str(fp))
    if not cap.isOpened():
        raise OSError(f"cannot open video {fp}")
    fps  = cap.get(cv2.CAP_PROP_FPS)
    nfrm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or nfrm <= 0:
        raise ValueError(f"invalid video meta fps={fps}, frames={nfrm} for {fp}")
    return fps, nfrm

def fit_eye2vid(yml: dict[str, list]):
    keys = [("calibration_times", "calibration_orig_times"),
            ("validation_times", "validation_orig_times")]
    eye, vid = [], []
    for e, v in keys:
        if e in yml and v in yml:
            eye.extend(np.ravel(yml[e]))
            vid.extend(np.ravel(yml[v]))
    eye = np.asarray(eye, float)
    vid = np.asarray(vid, float)
    if eye.size < 4:
        raise ValueError("<4 sync points; cannot fit")
    a, b = np.polyfit(eye, vid, 1)
    # Compute R²
    ss_res = ((vid - (a*eye + b))**2).sum()
    ss_tot = ((vid - vid.mean())**2).sum()
    r2 = 1 - ss_res/ss_tot if ss_tot else 0.0
    if r2 < 0.9:
        raise ValueError(f"poor sync fit r2={r2:.3f}")
    return a, b

def load_gaze(fp: Path):
    gz = np.load(fp, allow_pickle=True, mmap_mode="r")
    out = {}
    for eye in ("left", "right"):
        rec = gz[eye].item()
        ts  = rec["timestamp"].astype(float)
        nx  = rec["norm_pos"][:, 0].astype(np.float32)
        ny  = rec["norm_pos"][:, 1].astype(np.float32)
        cf  = rec["confidence"].astype(np.float32)
        mask = np.isfinite(ts) & np.isfinite(nx) & np.isfinite(ny) & np.isfinite(cf)
        out[eye] = (ts[mask], nx[mask], ny[mask], cf[mask])
    return out

# ─── WORKER ────────────────────────────────────────────────────────────

def process_session(sess: str, summary: bool) -> int:
    try:
        vid_fp = VIDEO_DIR / sess / "worldPrivate.mp4"
        yml_fp = GAZE_DIR / sess / "marker_times.yaml"
        npz_fp = GAZE_DIR / sess / "processedGaze" / "gaze.npz"
        if not (vid_fp.is_file() and yml_fp.is_file() and npz_fp.is_file()):
            logger.warning("%s: missing inputs", sess)
            return 0

        out_name = f"{sess}_{'gaze_mean' if summary else 'gaze_raw'}.parquet"
        out_fp_tmp = OUT_DIR / ("." + out_name)  # temp file for atomic write
        out_fp     = OUT_DIR / out_name
        if out_fp.exists():
            return 0  # idempotent skip

        fps, nfrm = video_meta(vid_fp)
        a, b      = fit_eye2vid(read_yaml(yml_fp))
        gaz       = load_gaze(npz_fp)

        # concatenate eyes
        eye_lbl = np.concatenate([
            np.full(gaz["left"][0].shape, "L"),
            np.full(gaz["right"][0].shape, "R")
        ])
        ts   = np.concatenate([gaz["left"][0],  gaz["right"][0]])
        nx   = np.concatenate([gaz["left"][1],  gaz["right"][1]])
        ny   = np.concatenate([gaz["left"][2],  gaz["right"][2]])
        conf = np.concatenate([gaz["left"][3],  gaz["right"][3]])

        vid_sec = a * ts + b
        frm_idx = np.round(vid_sec * fps).astype(np.int64)
        valid   = (frm_idx >= 0) & (frm_idx < nfrm)
        if not valid.any():
            logger.warning("%s: no valid frame indices after sync", sess)
            return 0

        if summary:
            df = (pd.DataFrame(dict(frame_idx=frm_idx[valid],
                                    norm_x=nx[valid],
                                    norm_y=ny[valid],
                                    conf=conf[valid]))
                    .groupby("frame_idx", sort=False)
                    .mean()
                    .reset_index())
        else:
            df = pd.DataFrame(dict(frame_idx=frm_idx[valid],
                                   eye=eye_lbl[valid],
                                   norm_x=nx[valid],
                                   norm_y=ny[valid],
                                   conf=conf[valid]))

        # atomic write
        df.to_parquet(out_fp_tmp, index=False)
        os.replace(out_fp_tmp, out_fp)
        return len(df)

    except Exception as e:
        logger.error("%s failed: %s", sess, e)
        return 0

# ─── MAIN ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Map gaze to frames (cluster‑ready)")
    parser.add_argument("--summary", action="store_true", help="mean‑per‑frame output instead of raw")
    parser.add_argument("--session-list", type=Path, default=DEFAULT_LIST,
                        help="path to text file containing session IDs to process")
    args = parser.parse_args()

    # determine worker count for SLURM
    slurm_cpus = int(os.getenv("SLURM_CPUS_ON_NODE", "0")) or None
    max_workers = min(slurm_cpus or 9999, os.cpu_count() or 4)

    # read session list
    if not args.session_list.is_file():
        raise SystemExit(f"Session list {args.session_list} not found")
    with open(args.session_list) as f:
        wanted = {ln.strip() for ln in f if ln.strip()}
    if not wanted:
        raise SystemExit("Session list is empty")

    # filter to sessions that physically exist in VIDEO_DIR
    sessions = [s for s in wanted if (VIDEO_DIR / s).is_dir()]
    missing  = wanted - set(sessions)
    for m in sorted(missing):
        logger.warning("%s listed but not found in VIDEO_DIR", m)
    if not sessions:
        raise SystemExit("No valid sessions to process")
    sessions.sort()

    total = 0
    with fut.ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn")) as ex:
        for n in tqdm(ex.map(process_session, sessions, [args.summary]*len(sessions)),
                      total=len(sessions), desc="Sessions"):
            total += n
    logger.info("✓ wrote %s rows → %s", format(total, ","), OUT_DIR)

if __name__ == "__main__":
    main()
