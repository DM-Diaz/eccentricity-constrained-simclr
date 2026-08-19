#!/usr/bin/env python3
# Last checked: 8/11/26
"""
in-domain classifier using SimCLR pretrained weights.

What this script does
- Reads a manifest CSV with at least: session, filename, split, and a label column (e.g., task).
- Builds image paths using a template: {data_root}/{session}/{filename}.
- Loads a SimCLR checkpoint into a torchvision ResNet backbone (projection head ignored).
- Runs staged supervised training:
    1) Linear probe (frozen backbone; train fc only)
    2) Fine-tune (unfreeze layer4+fc or all)
- Evaluates on split(s) we specify (default: test), writes:
    - best_checkpoint.pt (best by eval accuracy across both stages)
    - final_checkpoint.pt
    - history.json (epoch-by-epoch metrics, includes stage)
    - metrics.json (summary + per-class metrics)
    - confusion_matrix.csv (best)
    - label2id.json
    - run_config.json (args + environment fingerprint)

Reproducibility & robustness features
- Seeding for Python/NumPy/PyTorch (+ per-worker DataLoader seeding)
- Optional deterministic mode (--deterministic) for reproducibility
- Dataset integrity checks (required columns, missing files, empty splits)
- Split/label distribution reporting
- Safe checkpoint key matching with detailed load report
- Optional DataLoader timeout to avoid cluster hangs

NOTE:
- !!! Sbatch params take precedence so make sure to verify against default values!!!
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import socket
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms


# -------------------------
# Reproducibility helpers
# -------------------------
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_determinism() -> None:
    """
    Best-effort deterministic behavior.
    Note: Some ops may not have deterministic implementations on all GPUs.
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Older PyTorch or limited support; keep best-effort flags above.
        pass


def make_worker_init_fn(base_seed: int):
    def _init_fn(worker_id: int):
        # Derive a unique seed per worker
        s = (base_seed + worker_id) % (2**32 - 1)
        seed_all(s)
    return _init_fn


def sha256_file(path: Path, max_mb: int = 64) -> str:
    """
    Hash a file (up to max_mb MB) to fingerprint inputs/checkpoints.
    For huge checkpoints, hashing the first N MB is usually enough for a log.
    """
    h = hashlib.sha256()
    max_bytes = max_mb * 1024 * 1024
    read = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
            if read >= max_bytes:
                break
    return h.hexdigest()


# -------------------------
# Dataset
# -------------------------
class ManifestFrames(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], transform=None) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        return im, y, str(path)


# -------------------------
# Checkpoint loading (SimCLR -> torchvision ResNet)
# -------------------------
def _strip_prefixes(k: str) -> str:
    prefixes = [
        "module.",
        "model.",
        "net.",
        "encoder.",
        "backbone.",
        "resnet.",
        "online_network.",
        "target_network.",
        "encoder_q.",
        "student.",
        "teacher.",
    ]
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                changed = True
    return k


def _extract_state_dict(ckpt: dict) -> dict:
    # Common containers
    for key in ["state_dict", "model_state", "model_state_dict", "net", "model"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    # Sometimes ckpt itself is the state dict
    return ckpt


@dataclass
class CkptLoadReport:
    matched: int
    skipped: int
    missing: int
    unexpected: int


def load_simclr_into_resnet_backbone(resnet: nn.Module, ckpt_path: Path, verbose: bool = True) -> CkptLoadReport:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = _extract_state_dict(ckpt)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint at {ckpt_path} does not look like a state dict. Type={type(state)}")

    model_sd = resnet.state_dict()
    load_sd: Dict[str, torch.Tensor] = {}

    matched = 0
    skipped = 0

    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue

        k2 = _strip_prefixes(k)

        # Skip projection/prediction heads and classifier heads
        low = k2.lower()
        if any(s in low for s in ["projector", "projection", "predictor", "prototypes", "head"]):
            skipped += 1
            continue
        if k2.startswith("fc.") or k2.startswith("classifier."):
            skipped += 1
            continue

        if k2 in model_sd and model_sd[k2].shape == v.shape:
            load_sd[k2] = v
            matched += 1
        else:
            skipped += 1

    missing_keys, unexpected_keys = resnet.load_state_dict(load_sd, strict=False)

    report = CkptLoadReport(
        matched=matched,
        skipped=skipped,
        missing=len(missing_keys),
        unexpected=len(unexpected_keys),
    )

    if verbose:
        print(f"[ckpt] Loaded into backbone: matched={report.matched} skipped={report.skipped}")
        print(f"[ckpt] load_state_dict(strict=False): missing={report.missing} unexpected={report.unexpected}")
        # Don't spam long key lists; show a few
        if len(missing_keys) > 0:
            show = missing_keys[:20]
            print(f"[ckpt] missing keys (examples): {show}{' ...' if len(missing_keys)>20 else ''}")
        if len(unexpected_keys) > 0:
            show = unexpected_keys[:20]
            print(f"[ckpt] unexpected keys (examples): {show}{' ...' if len(unexpected_keys)>20 else ''}")

    return report


# -------------------------
# Metrics
# -------------------------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def per_class_metrics(cm: np.ndarray) -> Dict[str, List[float]]:
    # precision, recall, f1 per class
    tp = np.diag(cm).astype(np.float64)
    fp = np.sum(cm, axis=0).astype(np.float64) - tp
    fn = np.sum(cm, axis=1).astype(np.float64) - tp

    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec  = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1   = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)

    return {
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "f1": f1.tolist(),
        "support": np.sum(cm, axis=1).astype(int).tolist(),
    }


def macro_f1(f1: List[float], support: List[int]) -> float:
    # simple macro average over classes that have at least 1 sample
    f1_arr = np.asarray(f1, dtype=np.float64)
    sup = np.asarray(support, dtype=np.int64)
    mask = sup > 0
    if mask.sum() == 0:
        return 0.0
    return float(f1_arr[mask].mean())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> Dict:
    model.eval()
    ce = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    loss_sum = 0.0

    y_true_list = []
    y_pred_list = []

    for x, y, _paths in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = ce(logits, y)

        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        loss_sum += loss.item() * y.size(0)

        y_true_list.append(y.detach().cpu().numpy())
        y_pred_list.append(pred.detach().cpu().numpy())

    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)

    y_true = np.concatenate(y_true_list) if y_true_list else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_list) if y_pred_list else np.array([], dtype=np.int64)

    cm = confusion_matrix(y_true, y_pred, num_classes)
    pcm = per_class_metrics(cm)

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "total": int(total),
        "confusion_matrix": cm,
        "per_class": pcm,
        "macro_f1": macro_f1(pcm["f1"], pcm["support"]),
    }


def save_confusion_matrix_csv(cm: np.ndarray, out_path: Path, id2label: Dict[int, str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [id2label[i] for i in range(len(id2label))]
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + labels)
        for i, row in enumerate(cm):
            w.writerow([labels[i]] + row.tolist())


# -------------------------
# Model stage control
# -------------------------
def set_backbone_requires_grad(model: nn.Module, train_backbone: bool, unfreeze: str = "all") -> None:
    """
    - train_backbone=False: train only fc
    - train_backbone=True:
        - unfreeze='all': train everything
        - unfreeze='layer4': train layer4 + fc only
    """
    for _, p in model.named_parameters():
        p.requires_grad = True

    if not train_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
        return

    if unfreeze == "layer4":
        for name, p in model.named_parameters():
            if not (name.startswith("layer4.") or name.startswith("fc.")):
                p.requires_grad = False


def set_batchnorm_eval(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def lr_schedule(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    t = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    return 0.5 * base_lr * (1.0 + np.cos(np.pi * t))


def train_stage(
    stage_name: str,
    model: nn.Module,
    dl_train: DataLoader,
    dl_eval: DataLoader,
    device: torch.device,
    num_classes: int,
    epochs: int,
    base_lr: float,
    weight_decay: float,
    warmup_epochs: int,
    select_metric: str,
    label_smoothing: float,
    early_stop_patience: int,
    fp16: bool,
    bn_eval: bool,
    out_dir: Path,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    best_tracker: dict,
    history: list,
) -> None:
    optimizer = make_optimizer(model, base_lr, weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and device.type == "cuda"))

    # Training loss (optionally label-smoothed). Keeps eval loss standard CE.
    try:
        ce = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    except TypeError:
        if float(label_smoothing) > 0:
            print("[warn] label_smoothing not supported by this torch version; proceeding with standard CrossEntropyLoss().")
        ce = nn.CrossEntropyLoss()

    if len(dl_train) == 0:
        raise RuntimeError("Train DataLoader is empty. Check split filters and file paths.")

    total_steps = epochs * len(dl_train)
    warmup_steps = int(warmup_epochs * len(dl_train))
    global_step = 0

    metric_key = "acc" if str(select_metric).lower() == "acc" else "macro_f1"
    best_stage_metric = -1.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        if bn_eval:
            set_batchnorm_eval(model)

        loss_sum = 0.0
        correct = 0
        total = 0

        for x, y, _paths in dl_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            lr = lr_schedule(global_step, total_steps, base_lr, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(fp16 and device.type == "cuda")):
                logits = model(x)
                loss = ce(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred = torch.argmax(logits.detach(), dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
            loss_sum += loss.item() * y.size(0)
            global_step += 1

        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)

        ev = evaluate(model, dl_eval, device, num_classes)
        row = {
            "stage": stage_name,
            "stage_epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "eval_loss": float(ev["loss"]),
            "eval_acc": float(ev["acc"]),
            "eval_macro_f1": float(ev["macro_f1"]),
        }
        history.append(row)

        print(f"[{stage_name} {epoch:03d}/{epochs}] train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"eval loss={ev['loss']:.4f} acc={ev['acc']:.4f} macroF1={ev['macro_f1']:.4f}")

        ev_metric = float(ev[metric_key])

        # Stage-local early stopping (optional)
        if ev_metric > best_stage_metric:
            best_stage_metric = ev_metric
            no_improve = 0
        else:
            no_improve += 1

        # Global best checkpoint selection (across stages) by selected metric
        if ev_metric > best_tracker["best_metric"]:
            best_tracker["best_metric"] = ev_metric
            best_tracker["best_metric_name"] = metric_key
            best_tracker["best_acc"] = float(ev["acc"])
            best_tracker["best_stage"] = stage_name
            best_tracker["best_stage_epoch"] = int(epoch)
            best_tracker["best_macro_f1"] = float(ev["macro_f1"])

            torch.save({
                "stage": stage_name,
                "stage_epoch": epoch,
                "model_state": model.state_dict(),
                "best_metric_name": best_tracker["best_metric_name"],
                "best_metric": best_tracker["best_metric"],
                "best_acc": best_tracker["best_acc"],
                "best_macro_f1": best_tracker["best_macro_f1"],
                "label2id": label2id,
                "id2label": id2label,
            }, best_tracker["best_path"])

            save_confusion_matrix_csv(ev["confusion_matrix"], best_tracker["best_cm_path"], id2label)
            best_tracker["best_per_class"] = ev["per_class"]

            print(f"[save] new best by {metric_key}={best_tracker['best_metric']:.4f} "
                  f"(acc={best_tracker['best_acc']:.4f}, macroF1={best_tracker['best_macro_f1']:.4f}) "
                  f"-> {best_tracker['best_path'].name}")

        if early_stop_patience and no_improve >= int(early_stop_patience):
            print(f"[early-stop] stage={stage_name} no improvement in {metric_key} for {no_improve} epoch(s); stopping stage.")
            break

        (out_dir / "history.json").write_text(json.dumps(history, indent=2))


# -------------------------
# Manifest parsing
# -------------------------
def normalize_ok_series(s: pd.Series) -> pd.Series:
    # Accept True/False, 1/0, "true"/"false", "1"/"0"
    if s.dtype == bool:
        return s
    # Handle numeric
    if np.issubdtype(s.dtype, np.number):
        return s.astype(int) != 0
    # Handle strings
    s2 = s.astype(str).str.strip().str.lower()
    return s2.isin(["true", "1", "t", "yes", "y"])


def build_items_from_manifest(
    df: pd.DataFrame,
    data_root: Path,
    path_template: str,
    session_col: str,
    filename_col: str,
    label_col: str,
    require_ok: bool,
    ok_col: str,
    split_col: str,
    include_splits: List[str],
    strict_paths: bool,
    max_items: int,
) -> Tuple[List[Tuple[Path, str]], int]:
    # Filter ok if requested
    if require_ok:
        if ok_col not in df.columns:
            raise ValueError(f"--require-ok set but ok_col={ok_col!r} not in manifest columns.")
        ok_mask = normalize_ok_series(df[ok_col])
        df = df[ok_mask].copy()

    if split_col not in df.columns:
        raise ValueError(f"split_col={split_col!r} not found. Available: {list(df.columns)}")

    df = df[df[split_col].isin(include_splits)].copy()
    if df.empty:
        raise RuntimeError(f"No rows found for splits={include_splits} after filtering.")

    for c in (session_col, filename_col, label_col):
        if c not in df.columns:
            raise ValueError(f"Column {c!r} missing. Available: {list(df.columns)}")

    # Optional cap for quick smoke tests
    if max_items > 0:
        df = df.sample(n=min(max_items, len(df)), random_state=0).copy()

    items: List[Tuple[Path, str]] = []
    missing = 0

    for row in df.itertuples(index=False):
        session = getattr(row, session_col)
        filename = getattr(row, filename_col)
        label = getattr(row, label_col)

        p = Path(path_template.format(
            data_root=str(data_root),
            session=str(session),
            filename=str(filename),
        ))

        if not p.exists():
            missing += 1
            if strict_paths:
                raise FileNotFoundError(f"Missing file: {p}")
            continue

        items.append((p, str(label)))

    return items, missing


def make_label_maps(train_labels: List[str], eval_labels: List[str], unseen_policy: str) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    uniq_train = sorted(set(train_labels))
    label2id = {lab: i for i, lab in enumerate(uniq_train)}

    unseen = sorted(set(eval_labels) - set(train_labels))
    if unseen and unseen_policy == "error":
        msg = (
            "Found labels in eval split that were not in finetune split: "
            + str(unseen[:20]) + (" ..." if len(unseen) > 20 else "")
            + "\n\n"
            + "This is expected if the splits were created by *session* without label-stratification. "
            + "For a supervised multiclass classifier, those labels are effectively out-of-domain "
            + "because the classifier sees zero training examples for them.\n"
            + "Fix options:\n"
            + "  (1) Train on splits that include those labels (e.g., --finetune-splits train val)\n"
            + "  (2) Run with --unseen-eval-policy drop to evaluate only labels seen in finetune."
        )
        raise RuntimeError(msg)

    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label, unseen


def print_split_stats(name: str, items: List[Tuple[Path, str]]) -> None:
    labels = [lab for _p, lab in items]
    vc = pd.Series(labels).value_counts()
    print(f"[data] {name}: n={len(items)} classes={len(vc)}")
    print(vc.head(15).to_string())
    if len(vc) > 15:
        print("...")


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--manifest-csv", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)

    ap.add_argument("--path-template", type=str, default="{data_root}/{session}/{filename}")
    ap.add_argument("--session-col", type=str, default="session")
    ap.add_argument("--filename-col", type=str, default="filename")
    ap.add_argument("--split-col", type=str, default="split")
    ap.add_argument("--ok-col", type=str, default="ok")
    ap.add_argument("--label-col", type=str, default="task")

    ap.add_argument("--finetune-splits", nargs="+", default=["val"])
    ap.add_argument("--eval-splits", nargs="+", default=["test"])

    ap.add_argument("--final-eval-splits", nargs="+", default=[],
                    help=("Optional: evaluate ONCE at the end on these split(s) using best_checkpoint.pt. "
                          "This does not affect checkpoint selection. Example: --eval-splits val --final-eval-splits test"))

    ap.add_argument("--require-ok", action="store_true")
    ap.add_argument("--strict-paths", action="store_true")
    ap.add_argument("--unseen-eval-policy", type=str, default="error",
                    choices=["error", "drop"],
                    help=(
                        "What to do if eval split contains labels not present in finetune split. "
                        "'error' (default) stops with a clear message; "
                        "'drop' filters eval rows to only labels seen in finetune split and writes a report."
                    ))

    ap.add_argument("--max-finetune-items", type=int, default=0,
                    help="Cap number of finetune items for quick debugging (0 = no cap).")
    ap.add_argument("--max-eval-items", type=int, default=0,
                    help="Cap number of eval items for quick debugging (0 = no cap).")

    ap.add_argument("--arch", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    ap.add_argument("--image-size", type=int, default=224)

    ap.add_argument("-b", "--batch-size", type=int, default=256)
    ap.add_argument("--stage", type=str, default="both", choices=["probe", "finetune", "both"])

    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--finetune-epochs", type=int, default=100)

    ap.add_argument("--probe-lr", type=float, default=1e-3)
    ap.add_argument("--finetune-lr", type=float, default=6e-4)

    ap.add_argument("--unfreeze", type=str, default="layer4", choices=["all", "layer4"])
    ap.add_argument("--bn-eval-during-probe", action="store_true")

    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5)

    ap.add_argument("--select-metric", type=str, default="acc",
                    choices=["acc", "macro_f1"],
                    help="Metric used to select best_checkpoint.pt during training. Use macro_f1 for imbalanced labels.")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="Label smoothing for training cross-entropy (0.0 disables).")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="Stop a training stage early if selected metric doesn't improve for N eval epochs (0 disables).")

    ap.add_argument("--fp16", action="store_true", help="Enable AMP mixed precision.")

    ap.add_argument("-j", "--workers", type=int, default=8)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--persistent-workers", action="store_true")
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--dataloader-timeout", type=int, default=0,
                    help="Seconds to wait for a batch (0 = disabled). Helps avoid hangs on clusters.")
    ap.add_argument("--drop-last", action="store_true", help="Drop last incomplete train batch (default: False).")

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--deterministic", action="store_true",
                    help="Enable deterministic mode (slower). Recommended for final paper runs.")
    ap.add_argument("--out-dir", type=Path, default=Path("./in_domain_classifier_runs"))
    ap.add_argument("--run-name", type=str, default="")

    args = ap.parse_args()

    if any(str(s).lower() == "test" for s in args.eval_splits):
        print("[warn] eval_splits includes 'test'. If we pick best checkpoints using eval performance, this can leak test information. "
              "For paper runs, prefer using a validation split for model selection and reserve test for final, single-shot evaluation.", flush=True)

    seed_all(args.seed)
    if args.deterministic:
        enable_determinism()
    else:
        # Speed default for fixed-size image training
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} torch={torch.__version__} torchvision={torchvision.__version__}")
    print(f"[env] host={socket.gethostname()} platform={platform.platform()}")
    if torch.cuda.is_available():
        print(f"[env] gpu={torch.cuda.get_device_name(0)}")

    # Basic file checks
    if not args.manifest_csv.exists():
        raise FileNotFoundError(args.manifest_csv)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    df = pd.read_csv(args.manifest_csv)

    # Build items
    finetune_raw, miss_ft = build_items_from_manifest(
        df, args.data_root, args.path_template,
        args.session_col, args.filename_col, args.label_col,
        args.require_ok, args.ok_col,
        args.split_col, args.finetune_splits,
        args.strict_paths, args.max_finetune_items
    )
    eval_raw, miss_ev = build_items_from_manifest(
        df, args.data_root, args.path_template,
        args.session_col, args.filename_col, args.label_col,
        args.require_ok, args.ok_col,
        args.split_col, args.eval_splits,
        args.strict_paths, args.max_eval_items
    )

    print(f"[data] finetune splits={args.finetune_splits} n={len(finetune_raw)} missing_paths={miss_ft}")
    print(f"[data] eval     splits={args.eval_splits} n={len(eval_raw)} missing_paths={miss_ev}")
    if len(finetune_raw) == 0 or len(eval_raw) == 0:
        raise RuntimeError("One of the splits is empty after filtering. Check split names and file paths.")

    print_split_stats("finetune", finetune_raw)
    print_split_stats("eval", eval_raw)

    # Label maps from finetune; ensure eval doesn't contain unseen labels unless allowed
    label2id, id2label, unseen_eval_labels = make_label_maps(
        train_labels=[lab for _p, lab in finetune_raw],
        eval_labels=[lab for _p, lab in eval_raw],
        unseen_policy=args.unseen_eval_policy,
    )
    num_classes = len(label2id)
    print(f"[labels] num_classes={num_classes} label_col={args.label_col}")

    # If eval has labels unseen in finetune, either error (handled in make_label_maps) or drop them.
    dropped_eval = []
    if unseen_eval_labels and args.unseen_eval_policy == "drop":
        seen = set(label2id.keys())
        kept = []
        for p, lab in eval_raw:
            if lab in seen:
                kept.append((p, lab))
            else:
                dropped_eval.append((p, lab))
        eval_raw = kept
        print(f"[labels] unseen labels in eval (dropped): {len(unseen_eval_labels)} labels, {len(dropped_eval)} samples")


    finetune_items = [(p, label2id[lab]) for p, lab in finetune_raw if lab in label2id]
    eval_items = [(p, label2id.get(lab, -1)) for p, lab in eval_raw]
    eval_items = [(p, y) for p, y in eval_items if y >= 0]

    # If we dropped eval samples because their labels were unseen, stash a report to write later.
    unseen_drop_report = None
    if unseen_eval_labels and args.unseen_eval_policy == "drop":
        unseen_drop_report = {
            "unseen_labels": unseen_eval_labels,
            "dropped_samples": len(dropped_eval),
            "kept_samples": len(eval_items),
        }


    # Transforms
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(args.image_size * 1.14)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    ds_train = ManifestFrames(finetune_items, transform=train_tf)
    ds_eval = ManifestFrames(eval_items, transform=eval_tf)

    # DataLoader generator for reproducible shuffles
    g = torch.Generator()
    g.manual_seed(args.seed)

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        timeout=args.dataloader_timeout if args.workers > 0 else 0,
        drop_last=True if args.drop_last else False,
        worker_init_fn=make_worker_init_fn(args.seed),
        generator=g,
    )
    eval_workers = max(0, args.workers // 2)
    dl_eval = DataLoader(
        ds_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=eval_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and eval_workers > 0,
        prefetch_factor=args.prefetch_factor if eval_workers > 0 else None,
        timeout=args.dataloader_timeout if eval_workers > 0 else 0,
        drop_last=False,
        worker_init_fn=make_worker_init_fn(args.seed + 999),
    )

    # Model
    if args.arch == "resnet18":
        backbone = torchvision.models.resnet18(weights=None)
    else:
        backbone = torchvision.models.resnet50(weights=None)

    ckpt_report = load_simclr_into_resnet_backbone(backbone, args.checkpoint, verbose=True)

    in_feats = backbone.fc.in_features
    backbone.fc = nn.Linear(in_feats, num_classes)

    model = backbone.to(device)

    # Output dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or f"{args.arch}_{args.label_col}_{args.stage}_{stamp}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fingerprint inputs for reproducibility logs
    # If we dropped unseen eval labels, write report artifacts now.
    if unseen_drop_report is not None:
        (out_dir / "unseen_eval_report.json").write_text(json.dumps(unseen_drop_report, indent=2))
        if dropped_eval:
            import csv as _csv
            with (out_dir / "unseen_eval_dropped.csv").open("w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["path", "label"])
                for p, lab in dropped_eval[:200000]:
                    w.writerow([str(p), lab])


    ckpt_hash = sha256_file(args.checkpoint)
    manifest_hash = sha256_file(args.manifest_csv)

    run_fingerprint = {
        "command": " ".join(sys.argv),
        "timestamp": stamp,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "manifest_csv": str(args.manifest_csv),
        "manifest_sha256": manifest_hash,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": ckpt_hash,
        "ckpt_load_report": ckpt_report.__dict__,
        "finetune_splits": args.finetune_splits,
        "eval_splits": args.eval_splits,
        "finetune_n": len(finetune_items),
        "eval_n": len(eval_items),
    }

    (out_dir / "label2id.json").write_text(json.dumps(label2id, indent=2))
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    (out_dir / "run_fingerprint.json").write_text(json.dumps(run_fingerprint, indent=2, default=str))

    best_tracker = {
        "best_metric": -1.0,
        "best_metric_name": ("acc" if str(args.select_metric).lower()=="acc" else "macro_f1"),
        "best_acc": -1.0,
        "best_macro_f1": -1.0,
        "best_stage": "",
        "best_stage_epoch": 0,
        "best_path": out_dir / "best_checkpoint.pt",
        "best_cm_path": out_dir / "best_confusion_matrix.csv",
        "best_per_class": None,
    }
    history: list = []

    print(f"[run] out_dir={out_dir}")

    # Stage: probe
    if args.stage in ("probe", "both"):
        set_backbone_requires_grad(model, train_backbone=False)
        print("[stage] PROBE: frozen backbone, train fc only.")
        train_stage(
            stage_name="probe",
            model=model,
            dl_train=dl_train,
            dl_eval=dl_eval,
            device=device,
            num_classes=num_classes,
            epochs=int(args.probe_epochs),
            base_lr=float(args.probe_lr),
            weight_decay=float(args.weight_decay),
            warmup_epochs=int(min(args.warmup_epochs, args.probe_epochs)),
            select_metric=str(args.select_metric),
            label_smoothing=float(args.label_smoothing),
            early_stop_patience=int(args.early_stop_patience),
            fp16=bool(args.fp16),
            bn_eval=bool(args.bn_eval_during_probe),
            out_dir=out_dir,
            label2id=label2id,
            id2label=id2label,
            best_tracker=best_tracker,
            history=history,
        )

    # Stage: finetune
    if args.stage in ("finetune", "both"):
        set_backbone_requires_grad(model, train_backbone=True, unfreeze=args.unfreeze)
        print(f"[stage] FINETUNE: unfreeze={args.unfreeze} (plus fc).")
        train_stage(
            stage_name="finetune",
            model=model,
            dl_train=dl_train,
            dl_eval=dl_eval,
            device=device,
            num_classes=num_classes,
            epochs=int(args.finetune_epochs),
            base_lr=float(args.finetune_lr),
            weight_decay=float(args.weight_decay),
            warmup_epochs=int(min(args.warmup_epochs, args.finetune_epochs)),
            select_metric=str(args.select_metric),
            label_smoothing=float(args.label_smoothing),
            early_stop_patience=int(args.early_stop_patience),
            fp16=bool(args.fp16),
            bn_eval=False,
            out_dir=out_dir,
            label2id=label2id,
            id2label=id2label,
            best_tracker=best_tracker,
            history=history,
        )

    # Final eval summary (on current final model)
    final_eval = evaluate(model, dl_eval, device, num_classes)

    # Save final checkpoint
    torch.save({
        "stage": args.stage,
        "model_state": model.state_dict(),
        "final_eval": {
            "acc": final_eval["acc"],
            "macro_f1": final_eval["macro_f1"],
            "loss": final_eval["loss"],
        },
        "best": {
            "metric_name": best_tracker.get("best_metric_name", None),
            "metric": best_tracker.get("best_metric", None),
            "acc": best_tracker["best_acc"],
            "macro_f1": best_tracker.get("best_macro_f1", None),
            "stage": best_tracker["best_stage"],
            "stage_epoch": best_tracker["best_stage_epoch"],
        },
        "label2id": label2id,
        "id2label": id2label,
        "args": vars(args),
        "fingerprint": run_fingerprint,
    }, out_dir / "final_checkpoint.pt")

    # Optional final, single-shot evaluation on held-out split(s) using best checkpoint.
    # This is the paper-clean way to report test performance without peeking during training.
    final_once = None
    if args.final_eval_splits:
        final_splits = [str(s).strip() for s in args.final_eval_splits if str(s).strip()]
        if final_splits:
            best_path = best_tracker.get("best_path", None)
            ckpt_name = "final_model_in_memory"
            if best_path is not None and Path(best_path).exists():
                best_ckpt = torch.load(best_path, map_location="cpu")
                model.load_state_dict(best_ckpt["model_state"], strict=True)
                ckpt_name = Path(best_path).name
            else:
                print("[warn] --final-eval-splits requested but best_checkpoint.pt not found; using final in-memory model.", flush=True)

            final_once = {"checkpoint": ckpt_name, "splits": {}}

            eval_workers = max(0, int(args.workers) // 2)
            for split in final_splits:
                # Build raw items from manifest for this split
                raw_items, missing_paths = build_items_from_manifest(
                    df=df,
                    data_root=args.data_root,
                    path_template=args.path_template,
                    session_col=args.session_col,
                    filename_col=args.filename_col,
                    label_col=args.label_col,
                    require_ok=bool(args.require_ok),
                    ok_col=args.ok_col,
                    split_col=args.split_col,
                    include_splits=[split],
                    strict_paths=bool(args.strict_paths),
                    max_items=int(args.max_eval_items),
                )

                # Handle unseen labels relative to label2id
                unseen_labels = sorted(set([lab for _p, lab in raw_items]) - set(label2id.keys()))
                dropped = 0
                if unseen_labels:
                    if args.unseen_eval_policy == "error":
                        raise RuntimeError(
                            f"Final eval split {split!r} contains labels unseen in finetune splits: "
                            f"{unseen_labels[:20]}{' ...' if len(unseen_labels) > 20 else ''}. "
                            f"Use --unseen-eval-policy drop to filter them, or include them in training."
                        )
                    else:
                        seen = set(label2id.keys())
                        kept = []
                        for pth, lab in raw_items:
                            if lab in seen:
                                kept.append((pth, lab))
                            else:
                                dropped += 1
                        raw_items = kept

                final_items = [(p, label2id[lab]) for p, lab in raw_items if lab in label2id]
                ds_final = ManifestFrames(final_items, transform=eval_tf)
                dl_final = DataLoader(
                    ds_final,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=eval_workers,
                    pin_memory=args.pin_memory,
                    persistent_workers=args.persistent_workers and eval_workers > 0,
                    prefetch_factor=args.prefetch_factor if eval_workers > 0 else None,
                    timeout=args.dataloader_timeout if eval_workers > 0 else 0,
                    drop_last=False,
                    worker_init_fn=make_worker_init_fn(args.seed + 999),
                )

                res = evaluate(model, dl_final, device, num_classes)

                cm_path = out_dir / f"final_{split}_confusion_matrix.csv"
                save_confusion_matrix_csv(res["confusion_matrix"], cm_path, id2label)

                # Write per-split metrics artifact
                split_out = {
                    "split": split,
                    "checkpoint": ckpt_name,
                    "acc": float(res["acc"]),
                    "macro_f1": float(res["macro_f1"]),
                    "loss": float(res["loss"]),
                    "per_class": res["per_class"],
                    "n_items": int(len(ds_final)),
                    "missing_paths": int(missing_paths),
                    "dropped_unseen_labels": int(dropped),
                    "confusion_matrix_csv": cm_path.name,
                }
                (out_dir / f"final_{split}_metrics.json").write_text(json.dumps(split_out, indent=2))
                final_once["splits"][split] = split_out

    # Write metrics.json (always)
    metrics = {
        "best": {
            "metric_name": best_tracker.get("best_metric_name", None),
            "metric": best_tracker.get("best_metric", None),
            "acc": best_tracker["best_acc"],
            "macro_f1": best_tracker.get("best_macro_f1", None),
            "stage": best_tracker["best_stage"],
            "stage_epoch": best_tracker["best_stage_epoch"],
            "per_class": best_tracker.get("best_per_class", None),
            "confusion_matrix_csv": str(best_tracker["best_cm_path"].name),
        },
        "final_once": final_once,
        "final": {
            "acc": final_eval["acc"],
            "macro_f1": final_eval["macro_f1"],
            "loss": final_eval["loss"],
            "per_class": final_eval["per_class"],
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"[done] best_acc={best_tracker['best_acc']:.4f} best_stage={best_tracker['best_stage']} "
          f"final_acc={final_eval['acc']:.4f} final_macroF1={final_eval['macro_f1']:.4f}")
    print(f"[done] outputs in: {out_dir}")



if __name__ == "__main__":
    main()
