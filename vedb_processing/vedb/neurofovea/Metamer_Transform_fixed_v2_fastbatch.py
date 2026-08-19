#!/usr/bin/env python3
# Last checked: 8/11/26
# NOTE: This was adapted from NeuroFovea's original Metamer_Transform.py script!
"""
Metamer_Transform.py (fixed ver!!)

Key fixes vs the uploaded version:
- Adds receptive-field caching to avoid re-reading hundreds of PNG masks every run.
- Removes the expensive 512-channel tiling of the 2D mask; uses a 2D boolean mask directly.
- Makes noise match the input content size (no hard-coded 512×512).
- Optionally prebuild RF cache via --precache (can be called once before batch runs).
- Sanitizes scale tag in output filenames (0.4 -> 0p4) to avoid double-dot stems.
- Makes throttling optional (off by default).

This should be drop-in compatible with the existing CLI flags.
"""

import argparse
import csv
import re
from pathlib import Path
import math
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization, coral


# ────────────────────────────── CLI ──────────────────────────────
parser = argparse.ArgumentParser(
    description="Foveated metamer renderer (CUDA-safe, fixation-aware)."
)
# Basic options
parser.add_argument('--image', type=str, help='File path to the content image')
parser.add_argument('--image_dir', type=str, help='Directory path to a batch of content images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder-content-similar.pth')  # NOTE: different from classic decoder.pth

# Even more options
parser.add_argument('--image_size', type=int, default=512,
                    help='Resize shorter edge; 0 keeps original size')
parser.add_argument('--crop', action='store_true', help='Center crop to square of size image_size (if > 0)')
parser.add_argument('--save_ext', default='.png', help='Output image extension')
parser.add_argument('--output', type=str, default='output', help='Directory to save output image(s)')
parser.add_argument('--scale', type=str, default='0.4',
                    help='Rate of growth of the Log-Polar Receptive Fields (one of 0.25,0.3,0.4,0.5,0.6,0.7)')
parser.add_argument('--verbose', type=int, default=0,
                    help='Verbose prints (0/1). Use 1 only for debugging.')
parser.add_argument('--reference', type=int, default=0,
                    help='If 1, renders the reference image (no crowding).')

# Fixation options (in coordinates of the resized image; default center)
parser.add_argument('--fix_x', type=int, default=None, help='Fixation x (pixels in resized image)')
parser.add_argument('--fix_y', type=int, default=None, help='Fixation y (pixels in resized image)')
parser.add_argument('--fovea_radius', type=int, default=None, help='Foveal radius in pixels (currently unused)')

# Per-image fixation table (optional; enables per-image mask shift while keeping a single Python process)
parser.add_argument('--fixations_csv', type=str, default=None,
                    help='CSV with rows: frame_id,fix_x,fix_y,fovea_radius (header ok). frame_id should match stems like frame_000123.')
parser.add_argument('--fix_coord_size', type=int, default=None,
                    help='Coordinate grid of fixation CSV (e.g., 224 or 512). If omitted, inferred from data when possible.')
parser.add_argument('--fallback_center', action='store_true',
                    help='If a frame_id is missing in CSV, use center fixation instead of skipping.')
parser.add_argument('--pattern', type=str, default="*.png",
                    help='When using --image_dir, glob pattern for images (default: *.png).')
parser.add_argument('--skip_existing', action='store_true',
                    help='Skip outputs that already exist in --output directory.')
parser.add_argument('--plain_names', action='store_true',
                    help='If set, output names are <stem>.png (no scale tag).')

# RF cache options
parser.add_argument('--rf_root', type=str, default='./Receptive_Fields',
                    help='Root folder containing MetaWindows_clean_s{scale}/')
parser.add_argument('--cache_dir', type=str, default=None,
                    help='Where to store RF cache (.pt). Default: <rf_root>/_cache')
parser.add_argument('--precache', action='store_true',
                    help='Build RF cache for the selected --scale and exit (no image required).')

# Optional throttling (these r off by default)
parser.add_argument('--throttle_every', type=int, default=0,
                    help='If >0, sleep after every N images (useful to be kind to shared filesystems).')
parser.add_argument('--throttle_sec', type=float, default=0.0,
                    help='Seconds to sleep when throttling.')

args = parser.parse_args()

# ───────────────────────── constants & transforms ─────────────────
scale_in  = ['0.25', '0.3', '0.4', '0.5', '0.6', '0.7']
scale_out = [   377,    301,    187,    126,    103,     91]
Pooling_Region_Map = dict(zip(scale_in, scale_out))

if args.scale not in Pooling_Region_Map:
    raise ValueError(f"--scale must be one of {scale_in}, got: {args.scale}")

verb = int(args.verbose)

# Output post-resize (kept at 224×224 to match your original)
resize_output = transforms.Resize((224, 224))
to_pil_image = transforms.ToPILImage()
to_tensor    = transforms.ToTensor()

def test_transform(size, crop):
    tl = []
    if size and size > 0:
        tl.append(transforms.Resize(size))
    if crop and size and size > 0:
        tl.append(transforms.CenterCrop(size))
    tl.append(transforms.ToTensor())
    return transforms.Compose(tl)

# ───────────────────────── receptive fields (cached) ──────────────
def _rf_cache_path(scale: str, cache_dir: Path) -> Path:
    # Replace '.' to keep cache filenames simple (optional)
    tag = scale.replace('.', 'p')
    return cache_dir / f"rf_cache_s{tag}.pt"

def _build_rf_cache(scale: str, rf_root: Path, cache_dir: Path) -> Path:
    """
    Build and store:
      - mask_total_base: FloatTensor (N,64,64) in [0,1]
      - alpha_list: list[float] length N
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _rf_cache_path(scale, cache_dir)
    if cache_path.exists():
        return cache_path

    d = 1.281  # psychophysics constant (26 deg ~ 512 px)
    N = Pooling_Region_Map[scale]
    masks = torch.zeros(N, 64, 64, dtype=torch.float32)
    alpha_list = [0.0] * N

    rf_dir = rf_root / f"MetaWindows_clean_s{scale}"
    if not rf_dir.exists():
        raise FileNotFoundError(f"RF directory not found: {rf_dir}")

    for i in range(N):
        png = rf_dir / f"{i}.png"
        # One open; do deterministic NEAREST resizes for binary-ish masks
        im = Image.open(png).convert("L")
        im64  = im.resize((64, 64),   resample=Image.NEAREST)
        im512 = im.resize((512, 512), resample=Image.NEAREST)

        m64  = to_tensor(im64)[0]     # (64,64)
        m512 = to_tensor(im512)[0]    # (512,512)

        masks[i] = m64

        mask_size  = (m512 > 0.5).sum().item()
        recep_size = math.sqrt(mask_size / math.pi) * 26.0 / 512.0
        alpha_list[i] = 0.0 if i == 0 else -1.0 + 2.0/(1.0 + math.exp(-recep_size * d))

    payload = {
        "scale": scale,
        "N": N,
        "mask_total_base": masks,   # (N,64,64)
        "alpha_list": alpha_list,
    }
    torch.save(payload, cache_path)
    return cache_path

def load_receptive_fields_cached():
    """
    Returns shifted masks for the current fixation:
      mask_total_cpu: FloatTensor (N,64,64)
      alpha_list    : list[float]
    """
    rf_root = Path(args.rf_root)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (rf_root / "_cache")
    cache_path = _build_rf_cache(args.scale, rf_root, cache_dir)

    d = torch.load(cache_path, map_location="cpu")
    mask_total = d["mask_total_base"].clone()
    alpha_list = d["alpha_list"]

    # If a per-image fixation CSV is provided, keep the RF masks centered here and shift per-image later.
    if args.fixations_csv:
        return mask_total, alpha_list, cache_path

    # Fixation shift: map resized-image coords → 64×64 mask grid
    img_side = float(args.image_size if (args.image_size and args.image_size > 0) else 512)
    cx = img_side/2.0 if args.fix_x is None else float(args.fix_x)
    cy = img_side/2.0 if args.fix_y is None else float(args.fix_y)
    s = 64.0 / img_side  # e.g., 64/512 = 1/8
    dx = int(round((cx - img_side/2.0) * s))
    dy = int(round((cy - img_side/2.0) * s))
    if dx != 0 or dy != 0:
        mask_total = torch.roll(mask_total, shifts=(dy, dx), dims=(1, 2))

    return mask_total, alpha_list, cache_path


# ───────────────────────── precache-only mode ─────────────────────
if args.precache:
    rf_root = Path(args.rf_root)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (rf_root / "_cache")
    cache_path = _build_rf_cache(args.scale, rf_root, cache_dir)
    print(f"RF cache ready: {cache_path}", flush=True)
    raise SystemExit(0)


# ─────────────────────────── device / logging ─────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("━━━━━━━━ CUDA CHECK ━━━━━━━━━")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
else:
    print("⚠ Running on CPU")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", flush=True)

# ─────────────────────────── I/O setup ────────────────────────────
output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

if not (args.image or args.image_dir):
    raise ValueError("Provide either --image or --image_dir (or use --precache).")

if args.image:
    image_paths = [Path(args.image)]
else:
    image_dir = Path(args.image_dir)
    image_paths = sorted([f for f in image_dir.glob(args.pattern) if f.is_file()])


# ─────────────────────────── models / weights ─────────────────────
decoder = net.decoder
vgg     = net.vgg

def _safe_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')

warnings.filterwarnings("default")
decoder.load_state_dict(_safe_load(args.decoder))
vgg.load_state_dict(_safe_load(args.vgg))

vgg = nn.Sequential(*list(vgg.children())[:31])
decoder.eval(); vgg.eval()

decoder.to(device)
vgg.to(device)


# ───────────────────────── core rendering ─────────────────────────
# Cache for resized boolean masks keyed by (Hf, Wf)
_mask_resize_cache = {}

def foveated_style_transfer(vgg, decoder, content_3chw, mask_bool_dev, alpha_list, reference: int, fix_x=None, fix_y=None, img_side=None):
    """
    content_3chw: (3,H,W) on device
    mask_bool_dev: (N,64,64) bool on device
    """
    # Noise matches input content size (no hard-coded 512×512)
    noise = torch.randn_like(content_3chw)
    noise = coral(noise, content_3chw)

    # Batchify
    style   = content_3chw.unsqueeze(0)   # auto-style: style=content
    content = content_3chw.unsqueeze(0)
    noise   = noise.unsqueeze(0)

    # Feature maps
    content_f = vgg(content)

    # Ensure mask matches feature-map resolution (e.g., 224px input -> 28x28 features; 512px -> 64x64)
    Hf, Wf = int(content_f.shape[-2]), int(content_f.shape[-1])
    mask_use = mask_bool_dev
    if tuple(mask_use.shape[-2:]) != (Hf, Wf):
        key = (Hf, Wf)
        cached = _mask_resize_cache.get(key)
        if cached is None or cached.device != mask_use.device:
            cached = F.interpolate(mask_use.float().unsqueeze(1), size=(Hf, Wf), mode="nearest").squeeze(1).bool()
            _mask_resize_cache[key] = cached
        mask_use = cached

    # Per-image fixation shift (safe): roll mask in FEATURE-MAP coordinates.
    # We do this after resizing masks to (Hf,Wf), and we avoid mutating cached masks by using torch.roll (returns new tensor).
    if fix_x is not None and fix_y is not None and img_side is not None:
        try:
            img_side_f = float(img_side)
            # map pixel shift -> feature shift
            sx = float(Wf) / img_side_f
            sy = float(Hf) / img_side_f
            dx = int(round((float(fix_x) - img_side_f/2.0) * sx))
            dy = int(round((float(fix_y) - img_side_f/2.0) * sy))
            if dx != 0 or dy != 0:
                mask_use = torch.roll(mask_use, shifts=(dy, dx), dims=(1, 2))
        except Exception:
            # If anything goes wrong, just fall back to centered mask for that image.
            pass

    if reference == 1:
        return decoder(content_f)

    style_f   = vgg(style)
    noise_f   = vgg(noise)

    foveated_f = torch.zeros_like(content_f)

    N = mask_use.size(0)
    for i in range(N):
        mask2d = mask_use[i]  # (Hf,Wf)
        if not mask2d.any():
            continue

        alpha_i = float(alpha_list[i])

        # Flatten spatial dims via boolean mask (same behavior as original)
        c_mask = content_f[:, :, mask2d].unsqueeze(3)
        s_mask = style_f[:,   :, mask2d].unsqueeze(3)
        n_mask = noise_f[:,   :, mask2d].unsqueeze(3)

        tex_mask = adaptive_instance_normalization(n_mask, s_mask)
        mixed    = (1.0 - alpha_i) * c_mask + alpha_i * tex_mask
        foveated_f[:, :, mask2d] = mixed.squeeze(3)

    return decoder(foveated_f)


# ───────────────────────── main loop ──────────────────────────────
image_tf = test_transform(args.image_size, args.crop)

if verb == 1:
    print(f"{len(image_paths)} images queued", flush=True)

with torch.no_grad():
    mask_total_cpu, alpha_list, cache_path = load_receptive_fields_cached()
    if verb == 1:
        print(f"Using RF cache: {cache_path}", flush=True)

    mask_bool_dev = (mask_total_cpu.to(device, non_blocking=True) > 0.001)

    # Optional: load per-image fixations (frame_id -> (fx, fy, rad))
    fix_map = None
    inferred_fix_coord_size = None
    if args.fixations_csv:
        fix_map = {}
        csv_path = Path(args.fixations_csv)
        if csv_path.is_file():
            with csv_path.open("r", newline="") as f:
                r = csv.reader(f)
                rows = list(r)
            start = 0
            # tolerate header if first cell isn't a frame id
            if rows and rows[0] and (not str(rows[0][0]).startswith("frame_")):
                start = 1
            max_xy = 0.0
            for row in rows[start:]:
                if len(row) < 3:
                    continue
                fid = str(row[0]).strip().rstrip()
                if not fid:
                    continue
                try:
                    fx = float(row[1]); fy = float(row[2])
                    rad = float(row[3]) if (len(row) > 3 and str(row[3]).strip() != "") else None
                except Exception:
                    continue
                fix_map[fid] = (fx, fy, rad)
                max_xy = max(max_xy, fx, fy)
            # Infer coord grid if not provided
            if args.fix_coord_size is not None:
                inferred_fix_coord_size = int(args.fix_coord_size)
            else:
                # Heuristic: if max fixation coordinate exceeds image_size, assume 512 grid; else assume image_size grid.
                if max_xy > (float(args.image_size) + 1.0) and max_xy <= 1024.0:
                    inferred_fix_coord_size = 512
                else:
                    inferred_fix_coord_size = int(args.image_size) if (args.image_size and args.image_size > 0) else 512
            if verb == 1:
                print(f"[fix] Loaded {len(fix_map)} fixations from {csv_path} (coord_size={inferred_fix_coord_size})", flush=True)
        else:
            if verb == 1:
                print(f"[fix] No fixation CSV found at {csv_path}; will {'fallback to center' if args.fallback_center else 'skip missing'}", flush=True)


    for z, image_path in enumerate(image_paths):
        # Determine output name early (enables --skip_existing)
        scale_tag = args.scale.replace('.', 'p')
        if args.reference == 0:
            if args.plain_names:
                output_name = output_dir / f"{image_path.stem}{args.save_ext}"
            else:
                output_name = output_dir / f"{image_path.stem}_s{scale_tag}{args.save_ext}"
        else:
            output_name = output_dir / f"{image_path.stem}_Reference{args.save_ext}"

        if args.skip_existing and output_name.exists():
            if verb == 1:
                print(f"[{z+1}/{len(image_paths)}] SKIP existing {output_name.name}", flush=True)
            continue

        try:
            with Image.open(str(image_path)) as im:
                im = im.convert('RGB')
                image = image_tf(im)  # (3,H,W) CPU
        except Exception as e:
            print(f"[{z+1}/{len(image_paths)}] FAILED to read {image_path}: {e}", flush=True)
            continue

        image = image.to(device, non_blocking=True)

        if torch.cuda.is_available() and verb == 1:
            torch.cuda.synchronize()
        t0 = time.time()

        # Per-image fixation if provided
        fx = None; fy = None
        if fix_map is not None:
            stem = image_path.stem
            fid = stem
            # normalize common suffixes
            for suf in ('_fovea', '_periph'):
                if fid.endswith(suf):
                    fid = fid[:-len(suf)]
            # strip scale tag if present (e.g., _s0p4)
            fid = re.sub(r'_s\d+p\d+$', '', fid)
            got = fix_map.get(fid)
            if got is None:
                if not args.fallback_center:
                    if verb == 1:
                        print(f"[{z+1}/{len(image_paths)}] SKIP no fixation for {fid}", flush=True)
                    continue
                else:
                    got = None
            if got is not None:
                fx, fy, _rad = got
            # Determine coordinate scaling
            img_side = float(image.shape[-1])
            coord_side = float(inferred_fix_coord_size) if inferred_fix_coord_size is not None else img_side
            if fx is None or fy is None:
                fx = img_side/2.0
                fy = img_side/2.0
            else:
                if abs(coord_side - img_side) > 1e-6:
                    fx = fx * (img_side / coord_side)
                    fy = fy * (img_side / coord_side)
                # clamp
                fx = max(0.0, min(img_side - 1.0, float(fx)))
                fy = max(0.0, min(img_side - 1.0, float(fy)))

        output = foveated_style_transfer(vgg, decoder, image, mask_bool_dev, alpha_list, args.reference,
                                        fix_x=fx, fix_y=fy, img_side=(float(image.shape[-1]) if image is not None else None))

        if torch.cuda.is_available() and verb == 1:
            torch.cuda.synchronize()
        t1 = time.time()

        # Bring back to CPU and finalize
        output = output.detach().cpu()
        output2 = to_pil_image(torch.clamp(output.squeeze(0), 0, 1))
        output  = torch.clamp(to_tensor(resize_output(output2)), 0, 1)

        save_image(output, str(output_name))
        print(f"[{z+1}/{len(image_paths)}] {output_name.name}  |  {t1 - t0:.2f}s", flush=True)

        if args.throttle_every and args.throttle_sec and (z + 1) % int(args.throttle_every) == 0:
            time.sleep(float(args.throttle_sec))
