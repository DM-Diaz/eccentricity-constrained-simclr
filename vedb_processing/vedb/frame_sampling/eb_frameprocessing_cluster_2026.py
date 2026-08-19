# Last checked: 8/10/26
# Imports
from pathlib import Path
import os, re, logging
import multiprocessing as mp
import cv2, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sessions_file", required=True,
                    help="Path to batch .csv or .txt of valid session IDs")
args = parser.parse_args()

# ─────────── tweak here ──────────────────────────────────────────────
BASE_DIR      = Path("/user_data/dylandia")
VEDB_DIR      = Path("/lab_data/hendersonlab/datasets/vedb_video/sessions")

SLURM_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
CPU_COUNT  = os.cpu_count() or 2
MAX_WORKERS = min(32, SLURM_CPUS if SLURM_CPUS > 0 else CPU_COUNT)

# PARAMS
DRY_RUN       = False # Set True if testing
STRIDE_SEC    = 1          # Lower sec if want more data (but results in higher overhead/storage cost and likely higher temporal autocorrelation between frames)
MAX_FRAMES    = 1000       # Upper bound
VERBOSITY     = 1
# ─────────────────────────────────────────────────────────────────────

VIDEO_ROOT    = VEDB_DIR
OUT224_ROOT   = BASE_DIR / "eb_processedData" / "forProcessing" / "2026_frames_224"
OUT512_ROOT   = BASE_DIR / "eb_processedData" / "forProcessing" / "2026_frames_512"
MANIFEST_ROOT = BASE_DIR / "eb_processedData" / "forProcessing" / "frame_manifests"

MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.WARNING - 10*min(VERBOSITY, 2),
    format="%(levelname)s: %(message)s"
)

assert VIDEO_ROOT.is_dir(), "✖ VIDEO_ROOT missing – check VEDB_DIR"

import torchvision
from torchvision.transforms import InterpolationMode as IM

_VER = tuple(map(int, torchvision.__version__.split(".")[:2]))
if _VER >= (0, 14):
    PREPROCESS = T.Compose([
        T.Resize(256, interpolation=IM.BICUBIC, antialias=True),
        T.CenterCrop(224)
    ])
else:
    PREPROCESS = T.Compose([
        T.Resize(256, interpolation=IM.BICUBIC),
        T.CenterCrop(224)
    ])

# original exclusions + expanded "junk segment" exclusions
EXCLUDE_RE = re.compile(r"\b(setup|calib|calibration|validation|start|end)\b", re.I)

# This is the pattern used post-hoc, embedded here.
# It will catch:
#  - stop/end/ending/finish record(ing)
#  - set up / set-up / setup / setting up
#  - calib / calibration
#  - valid / validation
# THESE ARE PARTS OF THE RECORDING WE DO NOT WANT!!
JUNK_TASK_RE = re.compile(
    r"(?:\b(?:stop|end|ending|finish)\s*rec(?:ord(?:ing)?)?\b"
    r"|\bset[\s_-]*up\b|\bsetting\s*up\b"
    r"|\bcalib(?:ration)?\b|\bvalid(?:ation)?\b"
    r"|\bcheck(?:ing)?\s*rec(?:ord(?:ing)?)?\b)",   # ADDED
    re.I
)


def normalize_columns(df: pd.DataFrame, session: str) -> pd.DataFrame:
    norm = (
        pd.Index(df.columns.astype(str))
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )
    if norm.duplicated().any():
        dupes = norm[norm.duplicated()].unique().tolist()
        raise ValueError(f"{session}: duplicate columns after normalization: {dupes}")
    df.columns = norm

    # canonical headers
    if "task" not in df.columns:
        for alt in ["tasks", "activity", "activities", "label", "labels", "category", "class", "task_name", "tasklabel"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "task"})
                break

    if "lastframe" not in df.columns:
        for c in df.columns:
            if ("last" in c) and ("frame" in c):
                df = df.rename(columns={c: "lastframe"})
                break

    return df

def clean_text(x) -> str:
    """Light cleaning; no synonym mapping."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().lower()
    s = s.replace("’", "'").replace("`", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_frame_id(s: str) -> int:
    if pd.isna(s):
        raise ValueError("missing lastframe")
    m = re.search(r"(\d+)", str(s))
    if not m:
        raise ValueError(f"no frame index in {s!r}")
    return int(m.group(1))

def load_session_ids(path: Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sessions file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        df = normalize_columns(df, session="sessions_file")
        if "session" in df.columns:
            sids = df["session"].astype(str).tolist()
        else:
            sids = df.iloc[:, 0].astype(str).tolist()
    else:
        sids = [ln.strip() for ln in path.read_text().splitlines()]

    sids = [s.rstrip("/").strip() for s in sids if s and s.strip()]

    seen = set()
    out = []
    for s in sids:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def gather_sessions(root: Path, sessions_file: Path):
    valid_sids = load_session_ids(sessions_file)
    if not valid_sids:
        logging.warning(f"{sessions_file} is empty – no sessions to process.")
        return []
    return [root / sid for sid in valid_sids if (root / sid).is_dir()]

def harvest_session(session_dir: Path) -> tuple[str, int]:
    session = session_dir.name
    csv_path   = session_dir / f"{session}.csv"
    video_path = session_dir / "worldPrivate.mp4"

    if not (csv_path.exists() and video_path.exists()):
        logging.warning("%s: missing CSV or video – skip.", session)
        return session, 0

    # ---- parse CSV into segments ----
    try:
        df = pd.read_csv(csv_path)
        df = normalize_columns(df, session=session)

        if "lastframe" not in df.columns:
            logging.warning("%s: No lastframe-like column found – skip.", session)
            return session, 0

        if "task" not in df.columns:
            logging.warning("%s: No task/tasks-like column found; using empty labels.", session)
            df["task"] = ""

        if "location" not in df.columns:
            df["location"] = ""

        # Clean task/location
        df["task"] = df["task"].apply(clean_text)
        df["location"] = df["location"].apply(clean_text)

        df["lastframe"] = df["lastframe"].fillna("frame000000.jpg")
        df["lastframenum_raw"] = df["lastframe"].apply(parse_frame_id)

        df = df.sort_values(by="lastframenum_raw")

        # Exclude:
        #   - empty tasks
        #   - classic setup/calib/validation/start/end
        #   - expanded recording/setup/calib/validation patterns
        task_norm = df["task"]  # already lowercase
        exclude_mask = (task_norm == "") | task_norm.str.contains(EXCLUDE_RE, na=False) | task_norm.str.contains(JUNK_TASK_RE, na=False)

        segments = []
        prev_end = 0
        for idx, row in df.iterrows():
            seg_start = prev_end + 1
            seg_end   = int(row["lastframenum_raw"])
            task      = row["task"]
            loc       = row.get("location", "")
            if (not exclude_mask.loc[idx]) and (seg_start <= seg_end):
                segments.append((seg_start, seg_end, task, loc))
            prev_end = seg_end

        if not segments:
            logging.warning("%s: No usable segments found – skip.", session)
            return session, 0

        overall_end_raw = segments[-1][1]

    except Exception as e:
        logging.error("%s: CSV processing failed – %s. Skip.", session, e)
        return session, 0

    # ---- decode video and sample frames ----
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.warning("%s: Video failed to open – skip.", session)
            return session, 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps < 1:
            fps = 30.0

        step = max(1, int(round(STRIDE_SEC * fps)))

        # Heuristic: if annotation last frame >= total_frames, treat annotation as 1-based and shift to 0-based
        offset = -1 if (total_frames > 0 and overall_end_raw >= total_frames) else 0

        dest224 = OUT224_ROOT / session
        dest512 = OUT512_ROOT / session
        dest224.mkdir(parents=True, exist_ok=True)
        dest512.mkdir(parents=True, exist_ok=True)

        def frame_gen(segments, step, max_frames):
            count = 0
            for seg_id, (seg_start, seg_end, task, loc) in enumerate(segments):
                current = seg_start
                while current <= seg_end and count < max_frames:
                    yield {
                        "session": session,
                        "sample_i": count,
                        "segment_id": seg_id,
                        "segment_start": seg_start,
                        "segment_end": seg_end,
                        "task": task,
                        "location": loc,
                        "frame_idx": current,
                        "video_frame_idx": current + offset,
                    }
                    current += step
                    count += 1

        samples = list(frame_gen(segments, step, MAX_FRAMES))

        if total_frames > 0:
            samples = [s for s in samples if 0 <= s["video_frame_idx"] < total_frames]

        if not samples:
            cap.release()
            return session, 0

        samples.sort(key=lambda d: d["video_frame_idx"])

        grabbed = 0
        manifest_rows = []

        first_v = samples[0]["video_frame_idx"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_v)
        pos = first_v

        for s in samples:
            target_v = s["video_frame_idx"]

            while pos < target_v:
                ok = cap.grab()
                if not ok:
                    s["ok"] = False
                    s["error"] = f"grab failed while seeking to {target_v}"
                    manifest_rows.append(s)
                    break
                pos += 1
            else:
                ret, frame_bgr = cap.read()
                if not ret:
                    s["ok"] = False
                    s["error"] = f"read failed at video_frame_idx={target_v}"
                    manifest_rows.append(s)
                    continue
                pos += 1

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img_pil   = Image.fromarray(frame_rgb)
                img224    = PREPROCESS(img_pil)
                img512    = img224.resize((512, 512), Image.BICUBIC)

                fname = f"frame_{int(s['frame_idx']):06d}.png"
                s["filename"] = fname

                if not DRY_RUN:
                    try:
                        img224.save(dest224 / fname)
                        img512.save(dest512 / fname)
                        s["ok"] = True
                        s["error"] = ""
                        grabbed += 1
                    except OSError as e:
                        s["ok"] = False
                        s["error"] = f"save failed for {fname}: {e}"
                else:
                    s["ok"] = True
                    s["error"] = ""

                manifest_rows.append(s)

        cap.release()

        out_manifest = MANIFEST_ROOT / f"{session}_frames.csv"
        pd.DataFrame(manifest_rows).to_csv(out_manifest, index=False)

        return session, grabbed

    except Exception as e:
        logging.error("%s: Video processing failed – %s. Skip.", session, e)
        return session, 0

if __name__ == "__main__":
    sessions = gather_sessions(VIDEO_ROOT, Path(args.sessions_file))
    if not sessions:
        print("No sessions found. Check sessions_file and VIDEO_ROOT.")
        raise SystemExit(1)

    pool_workers = min(MAX_WORKERS, len(sessions))

    with mp.Pool(processes=pool_workers) as pool:
        results = []
        for sess, cnt in tqdm(pool.imap_unordered(harvest_session, sessions),
                              total=len(sessions),
                              desc="Sessions",
                              ascii=True,
                              ncols=80):
            results.append((sess, cnt))

    harvested = sum(c for _, c in results)
    print(f"✔ Done. {harvested} frame(s) across {len(sessions)} session(s).")
    if DRY_RUN:
        print("   (dry-run: nothing written)")
