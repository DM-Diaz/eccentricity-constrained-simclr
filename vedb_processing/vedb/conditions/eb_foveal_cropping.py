#!/usr/bin/env python3
"""
Gaze-driven foveal crops with grey annulus
(224×224 source → 112×112 crop → upsample to 224×224 + grey rim).

If a frame lacks high-confidence gaze (conf < 0.30) a centre crop is
used and flagged. Outputs PNG crops, fixation CSVs, and master JSONs.
Last checked: 8/10/26
"""

# ───────────────────────── CONFIG ───────────────────────────────────────
# imports
from __future__ import annotations
from pathlib import Path
import os, json, csv, cv2, numpy as np, pandas as pd, logging
import concurrent.futures as fut
from tqdm.auto import tqdm

# Base can be overridden on cluster: export EB_BASE=/user_data/dylandia
BASE        = Path(os.environ.get("EB_BASE", "/user_data/dylandia"))

FRAME_ROOT  = BASE / "eb_processedData/forProcessing/2026_frames_224"
PARQ_ROOT   = BASE / "eb_processedData/forProcessing/2026_frame_gaze_maps"
OUT_ROOT    = BASE / "eb_processedData/fovea/2026_gaze_crops_224"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# DEFAULTS SET ALREADY!
CROP        = 112                       # crop side
RAD         = CROP // 2                 # 56 px (used in metadata)
MASK_RADIUS = CROP                      # 112 px visible disc
SCALE       = 512 / 224
CONF_THR    = 0.30

# Respect SLURM allocation when available
MAX_PROCS   = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))

# Avoid filesystem thread-storm inside each process
IO_THREADS  = int(os.environ.get("IO_THREADS", "2"))

VERBOSITY   = 1
# ────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING - 10 * min(VERBOSITY, 2),
    format="%(levelname)s: %(message)s"
)

# ───────── build grey annulus mask once per process ─────────────────────
H = W = 224
GREY_VAL   = 128
FEATHER_K  = 15                          # odd → soft edge

_hard = np.zeros((H, W), np.uint8)
cv2.circle(_hard, (H//2, W//2), MASK_RADIUS, 255, -1)   # 112-px radius
_alpha = cv2.GaussianBlur(_hard, (FEATHER_K, FEATHER_K), 0).astype(np.float32) / 255
ALPHA3 = _alpha[..., None]                              # broadcast
GREY   = np.full((H, W, 3), GREY_VAL, np.uint8)

# ───────── helper -------------------------------------------------------
def clamp(cx: int, cy: int, crop=CROP, canvas=224):
    r = crop // 2
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(canvas, x1 + crop), min(canvas, y1 + crop)
    return x2 - crop, y2 - crop, x2, y2

# ───────── per-session worker ------------------------------------------
def process_session(sess: str):
    try:
        frm_dir = FRAME_ROOT / sess
        parq_fp = PARQ_ROOT / f"{sess}_gaze_raw.parquet"
        if not (frm_dir.is_dir() and parq_fp.is_file()):
            logging.warning("%s: missing inputs", sess)
            return sess, {}, {}

        dest = OUT_ROOT / sess
        if dest.exists() and any(dest.glob("*_fovea.png")):   # idempotent
            return sess, {}, {}
        dest.mkdir(parents=True, exist_ok=True)

        gdf = pd.read_parquet(parq_fp, columns=["frame_idx", "norm_x", "norm_y", "conf"])
        gdf = gdf[gdf.conf >= CONF_THR]
        buckets = (
            gdf.groupby("frame_idx")
               .apply(lambda g: (g.norm_x.values, g.norm_y.values, g.conf.values))
               .to_dict()
        )

        rows224, rows512, tasks = [], [], []
        for png in frm_dir.glob("frame_*.png"):
            fid  = png.stem
            idx  = int(fid.split("_")[1])
            pack = buckets.get(idx)

            if pack is None:          # centre fallback
                cx = cy = 112
                flag = "centre"
            else:                     # gaze centroid
                xs, ys, ws = pack
                s = ws.sum()
                if s <= 0:
                    cx = cy = 112
                    flag = "centre"
                else:
                    w_norm = ws / s
                    cx = int(round((xs * w_norm).sum() * 224))
                    cy = int(round((ys * w_norm).sum() * 224))
                    flag = "gaze"

            x1, y1, x2, y2 = clamp(cx, cy)
            tasks.append((png, dest / f"{fid}_fovea.png", x1, y1, x2, y2))

            rows224.append([fid, cx, cy, RAD, flag])
            rows512.append([
                fid,
                int(round(cx * SCALE)),
                int(round(cy * SCALE)),
                int(round(RAD * SCALE)),
                flag
            ])

        if not tasks:
            return sess, {}, {}

        # ---- thread-pool I/O ------------------------------------------
        def _save(t):
            src, dst, x1, y1, x2, y2 = t
            img = cv2.imread(str(src))
            if img is None:
                logging.warning("%s: failed to read %s", sess, src.name)
                return
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                logging.warning("%s: empty crop for %s", sess, src.name)
                return
            up = cv2.resize(crop, (W, H), cv2.INTER_LINEAR)
            out = (ALPHA3 * up + (1 - ALPHA3) * GREY).astype(np.uint8)
            cv2.imwrite(str(dst), out)

        with fut.ThreadPoolExecutor(IO_THREADS) as tp:
            list(tp.map(_save, tasks, chunksize=64))

        # ---- CSVs -----------------------------------------------------
        hdr = ["frame_id", "fx", "fy", "crop_radius", "source"]
        with (dest / f"{sess}_fixations_224.csv").open("w", newline="") as f:
            csv.writer(f).writerows([hdr, *rows224])
        with (dest / f"{sess}_fixations_512.csv").open("w", newline="") as f:
            csv.writer(f).writerows([hdr, *rows512])

        # ---- JSON dicts ----------------------------------------------
        fix224 = {r[0]: (r[1], r[2], r[3], r[4]) for r in rows224}
        fix512 = {r[0]: (r[1], r[2], r[3], r[4]) for r in rows512}
        return sess, fix224, fix512

    except Exception as e:
        logging.error("%s failed: %s", sess, e)
        return sess, {}, {}

# ───────── MAIN ---------------------------------------------------------
def main():
    sessions = sorted(
        p.stem.replace("_gaze_raw", "")
        for p in PARQ_ROOT.glob("*_gaze_raw.parquet")
    )
    all224, all512 = {}, {}

    with fut.ProcessPoolExecutor(MAX_PROCS) as ex:
        for sess, f224, f512 in tqdm(
            ex.map(process_session, sessions),
            total=len(sessions),
            desc="sessions"
        ):
            if f224:
                all224[sess] = f224
                all512[sess] = f512

    (OUT_ROOT / "all_fixations_224.json").write_text(json.dumps(all224))
    (OUT_ROOT / "all_fixations_512.json").write_text(json.dumps(all512))

    tot = sum(len(v) for v in all224.values())
    centre = sum(v[-1] == "centre" for sess in all224.values() for v in sess.values())
    if tot:
        print(f"✓ {tot:,} crops  |  centre fallbacks {centre:,}  ({centre/tot:.1%})")
    else:
        print("⚠️  0 crops written — check inputs.")

if __name__ == "__main__":
    main()
