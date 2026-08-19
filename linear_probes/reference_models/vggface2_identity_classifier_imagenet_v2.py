#!/usr/bin/env python3
# Last checked: 8/11/26
"""
VGGFace2 identity classification transfer from SimCLR-pretrained encoders.

Key features (mirrors in-domain staged classifier):
- ResNet18/ResNet50 backbone (--arch).
- Load SimCLR checkpoint robustly (handles common key prefixes, DDP, etc.).
- Staged training (--stage): linear probe (frozen backbone) then fine-tune (unfreeze layer4 or all).
- Stratified 80/20 split within each identity, cached to disk.
- Logs per-epoch: train loss/acc (+ optional top5), eval loss/top1/top5/macroF1.
- Saves artifacts:
    - config.json, run_fingerprint.json, label2id.json
    - history.json (append per epoch)
    - best_checkpoint.pt (selected by --select-metric)
    - best_top_confusions.csv (compressed confusion summary)
    - final_checkpoint.pt
    - metrics.json

Notes:
- Full confusion-matrix CSV is infeasible for VGGFace2 (8631 classes); we instead save top confusions
  as (true_label, pred_label, count).
- "Accuracy" here is Top-1 accuracy; Top-5 is reported separately.
- !!!Sbatch params take precedence, make sure to validate against default vals!!!
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
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, resnet50
from tqdm import tqdm


# -------------------------
# Repro
# -------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_worker_init_fn(seed: int):
    def _fn(worker_id: int):
        s = seed + worker_id
        random.seed(s)
        np.random.seed(s % (2**32 - 1))
        torch.manual_seed(s)
    return _fn


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------
# Robust dataset wrapper (optional)
# -------------------------
class SafeImageFolder(Dataset):
    """
    Wrap ImageFolder to survive occasional corrupt images.
    Returns (image, target, path). If a sample fails to load, it retries a few random indices.
    """
    def __init__(self, base: ImageFolder, transform, max_retries: int = 10, log_bad: bool = False):
        self.base = base
        self.transform = transform
        self.max_retries = int(max_retries)
        self.log_bad = bool(log_bad)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        for attempt in range(self.max_retries):
            try:
                path, target = self.base.samples[idx]
                img = self.base.loader(path)
                if self.transform is not None:
                    img = self.transform(img)
                return img, target, path
            except Exception as e:
                if self.log_bad:
                    sys.stderr.write(f"[bad] idx={idx} attempt={attempt+1}/{self.max_retries} err={repr(e)}\n")
                idx = random.randrange(0, len(self.base))
        raise RuntimeError(f"Too many failures loading images (last idx={idx}).")


# -------------------------
# Transforms
# -------------------------
def build_transforms(mode: str, img_size: int):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
    if mode == "train":
        # Modest augmentation; keeps faces mostly intact
        return transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
    elif mode == "eval":
        return transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise ValueError(mode)


# -------------------------
# Split: stratified 80/20 within class
# -------------------------
def stratified_split_indices_by_class(
    targets: List[int],
    train_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each class, split indices into train/val with given fraction.
    Ensures >=1 example per split for classes with >=2 samples.
    """
    targets = np.asarray(targets, dtype=np.int64)
    rng = np.random.default_rng(seed)

    train_idx = []
    val_idx = []

    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        if n < 2:
            train_idx.append(idx)
            continue
        n_train = int(round(train_frac * n))
        n_train = max(1, min(n_train, n - 1))
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:])

    train_idx = np.concatenate(train_idx) if len(train_idx) else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_idx) if len(val_idx) else np.array([], dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# -------------------------
# Checkpoint loading (SimCLR -> torchvision ResNet)
# -------------------------
def _extract_state_dict(ckpt: dict) -> dict:
    for key in ["state_dict", "model_state", "model_state_dict", "net", "model"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


def _iter_strip_known_prefixes(k: str) -> str:
    prefixes = [
        "module.",
        "model.",
        "net.",
        "encoder.",
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


def _remap_sequential_resnet_keys(k: str) -> str | None:
    """
    Map Sequential-wrapped ResNet keys to canonical torchvision names.

    Expected SimCLR-style examples:
      backbone.0.weight              -> conv1.weight
      backbone.1.weight              -> bn1.weight
      backbone.4.0.conv1.weight      -> layer1.0.conv1.weight
      backbone.5.0.conv1.weight      -> layer2.0.conv1.weight
      backbone.6.0.conv1.weight      -> layer3.0.conv1.weight
      backbone.7.0.conv1.weight      -> layer4.0.conv1.weight
    """
    if not k.startswith("backbone."):
        return None

    nk = k[len("backbone."):]

    if nk.startswith("0."):
        return "conv1." + nk[2:]
    if nk.startswith("1."):
        return "bn1." + nk[2:]
    if nk.startswith("4."):
        return "layer1." + nk[2:]
    if nk.startswith("5."):
        return "layer2." + nk[2:]
    if nk.startswith("6."):
        return "layer3." + nk[2:]
    if nk.startswith("7."):
        return "layer4." + nk[2:]

    # backbone.2 and backbone.3 are relu/maxpool and have no parameters
    return None


@dataclass
class CkptLoadReport:
    matched: int
    skipped: int
    missing: int
    unexpected: int


def load_resnet_backbone_from_simclr_checkpoint(
    resnet: nn.Module,
    ckpt_path: Path,
    verbose: bool = True,
) -> CkptLoadReport:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = _extract_state_dict(ckpt)
    if not isinstance(state, dict):
        raise ValueError(
            f"Checkpoint at {ckpt_path} does not look like a state dict. Type={type(state)}"
        )

    model_sd = resnet.state_dict()
    load_sd: Dict[str, torch.Tensor] = {}

    matched = 0
    skipped = 0

    for raw_k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue

        k = _iter_strip_known_prefixes(raw_k)

        low = k.lower()
        if any(s in low for s in ["projector", "projection", "predictor", "prototypes", "head"]):
            skipped += 1
            continue
        if k.startswith("fc.") or k.startswith("classifier."):
            skipped += 1
            continue

        candidate_keys = []

        # Case 1: already canonical torchvision key
        candidate_keys.append(k)

        # Case 2: sequentialized SimCLR backbone key
        remapped = _remap_sequential_resnet_keys(k)
        if remapped is not None:
            candidate_keys.append(remapped)

        loaded_this_key = False
        for k2 in candidate_keys:
            if k2 in model_sd and model_sd[k2].shape == v.shape:
                load_sd[k2] = v
                matched += 1
                loaded_this_key = True
                break

        if not loaded_this_key:
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
        if len(missing_keys) > 0:
            show = missing_keys[:20]
            print(f"[ckpt] missing keys (examples): {show}{' ...' if len(missing_keys) > 20 else ''}")
        if len(unexpected_keys) > 0:
            show = unexpected_keys[:20]
            print(f"[ckpt] unexpected keys (examples): {show}{' ...' if len(unexpected_keys) > 20 else ''}")

    return report


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def topk_acc(logits: torch.Tensor, y: torch.Tensor, ks=(1, 5)) -> Dict[int, float]:
    maxk = int(max(ks))
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # (B,maxk)
    correct = pred.eq(y.view(-1, 1))  # (B,maxk)
    out = {}
    for k in ks:
        out[int(k)] = float(correct[:, :k].any(dim=1).float().mean().item())
    return out


@torch.no_grad()
def per_class_prf_from_preds(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, List[float]]:
    """
    Computes per-class precision/recall/f1/support using bincount (no confusion matrix).
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = np.bincount(y_true[y_true == y_pred], minlength=num_classes).astype(np.int64)
    pred_cnt = np.bincount(y_pred, minlength=num_classes).astype(np.int64)
    true_cnt = np.bincount(y_true, minlength=num_classes).astype(np.int64)

    fp = pred_cnt - tp
    fn = true_cnt - tp

    eps = 1e-12
    precision = tp / np.maximum(tp + fp, 1)  # safe int denom
    recall = tp / np.maximum(tp + fn, 1)
    f1 = (2.0 * precision * recall) / np.maximum(precision + recall, eps)

    # For classes with zero support, define metrics as 0 (consistent and JSON-friendly)
    precision = np.where(true_cnt > 0, precision, 0.0)
    recall = np.where(true_cnt > 0, recall, 0.0)
    f1 = np.where(true_cnt > 0, f1, 0.0)

    return {
        "precision": precision.astype(np.float32).tolist(),
        "recall": recall.astype(np.float32).tolist(),
        "f1": f1.astype(np.float32).tolist(),
        "support": true_cnt.astype(np.int64).tolist(),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    num_classes: int,
    amp: bool,
    max_confusions: int = 2000,
) -> Dict[str, object]:
    model.eval()
    losses = []
    n = 0

    top1s = []
    top5s = []

    y_true_all = []
    y_pred_all = []

    # track most common confusions without materializing full matrix
    from collections import Counter
    confusion_counter = Counter()

    for x, y, _paths in tqdm(dl, desc="eval", dynamic_ncols=True, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        losses.append(float(loss.item()))
        accs = topk_acc(logits, y, ks=(1, 5))
        top1s.append(accs[1])
        top5s.append(accs[5])

        pred = torch.argmax(logits, dim=1)

        # gather for macroF1/per-class
        y_true = y.detach().cpu().numpy()
        y_pred = pred.detach().cpu().numpy()
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

        # update top confusions
        if max_confusions and max_confusions > 0:
            wrong = (y_pred != y_true)
            if wrong.any():
                pairs = list(zip(y_true[wrong].tolist(), y_pred[wrong].tolist()))
                confusion_counter.update(pairs)

        n += y.numel()

    if len(y_true_all):
        y_true_all = np.concatenate(y_true_all, axis=0)
        y_pred_all = np.concatenate(y_pred_all, axis=0)
    else:
        y_true_all = np.array([], dtype=np.int64)
        y_pred_all = np.array([], dtype=np.int64)

    per_class = per_class_prf_from_preds(y_true_all, y_pred_all, num_classes=num_classes)
    support = np.asarray(per_class["support"], dtype=np.int64)
    f1 = np.asarray(per_class["f1"], dtype=np.float32)
    macro_f1 = float(f1[support > 0].mean()) if (support > 0).any() else float("nan")

    # compressed confusions
    top_confusions = []
    if max_confusions and max_confusions > 0:
        for (t, p), c in confusion_counter.most_common(int(max_confusions)):
            top_confusions.append({"true": int(t), "pred": int(p), "count": int(c)})

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "top1": float(np.mean(top1s)) if top1s else float("nan"),
        "top5": float(np.mean(top5s)) if top5s else float("nan"),
        "acc": float(np.mean(top1s)) if top1s else float("nan"),  # alias for in-domain naming
        "macro_f1": macro_f1,
        "per_class": per_class,
        "n_items": int(n),
        "top_confusions": top_confusions,
    }


def save_top_confusions_csv(top_confusions: List[Dict[str, int]], path: Path, id2label: List[str]):
    # Columns: true_id,true_label,pred_id,pred_label,count
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_id", "true_label", "pred_id", "pred_label", "count"])
        for row in top_confusions:
            t = int(row["true"]); p = int(row["pred"]); c = int(row["count"])
            w.writerow([t, id2label[t] if t < len(id2label) else str(t),
                        p, id2label[p] if p < len(id2label) else str(p),
                        c])


# -------------------------
# Model
# -------------------------
class ResNetClassifier(nn.Module):
    def __init__(self, arch: str, num_classes: int):
        super().__init__()
        arch = arch.lower()
        if arch == "resnet18":
            self.backbone = resnet18(weights=None)
            feat_dim = 512
        elif arch == "resnet50":
            self.backbone = resnet50(weights=None)
            feat_dim = 2048
        else:
            raise ValueError(f"Unknown arch: {arch}")

        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)


def set_backbone_requires_grad(model: ResNetClassifier, train_backbone: bool, unfreeze: str = "layer4"):
    # Always train classifier
    for p in model.classifier.parameters():
        p.requires_grad = True

    for p in model.backbone.parameters():
        p.requires_grad = False

    if not train_backbone:
        return

    unfreeze = (unfreeze or "layer4").lower()
    if unfreeze == "all":
        for p in model.backbone.parameters():
            p.requires_grad = True
        return

    if unfreeze == "layer4":
        for p in model.backbone.layer4.parameters():
            p.requires_grad = True
        return

    raise ValueError(f"Unknown unfreeze option: {unfreeze} (use layer4|all)")


# -------------------------
# Train stages
# -------------------------
def train_stage(
    stage_name: str,
    model: ResNetClassifier,
    dl_train: DataLoader,
    dl_eval: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    fp16: bool,
    bn_eval: bool,
    out_dir: Path,
    label2id: Dict[str, int],
    id2label: List[str],
    best_tracker: Dict[str, object],
    history: List[Dict[str, object]],
    select_metric: str,
    early_stop_patience: int,
    early_stop_metric: str,
    max_confusions: int,
):
    # BN behavior
    if bn_eval:
        model.backbone.eval()

    # Optimizer over trainable params only
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and device.type == "cuda"))

    # LR scheduler (matches described method)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.1, patience=5, verbose=True
    )

    # early-stop bookkeeping (stage-local)
    if early_stop_metric == "loss":
        best_stage = float("inf")
        cmp = lambda cur, best: cur < best - 1e-12
    else:
        best_stage = -float("inf")
        cmp = lambda cur, best: cur > best + 1e-12
    no_improve = 0

    select_metric = select_metric.lower()
    early_stop_metric = early_stop_metric.lower()

    for epoch in range(1, epochs + 1):
        model.train()
        if bn_eval:
            # keep BN frozen even while classifier is in train()
            model.backbone.eval()

        loss_sum = 0.0
        correct1 = 0
        correct5 = 0
        total = 0

        pbar = tqdm(dl_train, desc=f"{stage_name} {epoch:03d}/{epochs}", dynamic_ncols=True, leave=False)
        for x, y, _paths in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(fp16 and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = y.size(0)
            loss_sum += float(loss.item()) * bs
            total += bs

            # training acc
            pred1 = torch.argmax(logits.detach(), dim=1)
            correct1 += int((pred1 == y).sum().item())
            # top-5 train acc (exact count)
            with torch.no_grad():
                _, pred5 = logits.detach().topk(5, dim=1, largest=True, sorted=True)
                correct5 += int(pred5.eq(y.view(-1, 1)).any(dim=1).sum().item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = loss_sum / max(total, 1)
        train_acc = correct1 / max(total, 1)
        train_top5 = correct5 / max(total, 1)

        ev = evaluate(
            model=model,
            dl=dl_eval,
            device=device,
            criterion=criterion,
            num_classes=num_classes,
            amp=fp16,
            max_confusions=max_confusions,
        )

        # LR scheduler uses val loss
        sched.step(ev["loss"])

        row = {
            "stage": stage_name,
            "stage_epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "train_top5": float(train_top5),
            "eval_loss": float(ev["loss"]),
            "eval_top1": float(ev["top1"]),
            "eval_top5": float(ev["top5"]),
            "eval_acc": float(ev["acc"]),
            "eval_macro_f1": float(ev["macro_f1"]),
        }
        history.append(row)

        print(
            f"[{stage_name} {epoch:03d}/{epochs}] "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} top5={train_top5:.4f} | "
            f"eval loss={ev['loss']:.4f} top1={ev['top1']:.4f} top5={ev['top5']:.4f} macroF1={ev['macro_f1']:.4f}",
            flush=True,
        )

        # global best selection across stages
        metric_map = {
            "top1": float(ev["top1"]),
            "acc": float(ev["acc"]),
            "top5": float(ev["top5"]),
            "macro_f1": float(ev["macro_f1"]),
            "loss": -float(ev["loss"]),  # maximize (neg loss) to fit selector convention
        }
        if select_metric not in metric_map:
            raise ValueError(f"Unknown --select-metric {select_metric}. Use one of: {sorted(metric_map.keys())}")
        ev_metric = metric_map[select_metric]

        if ev_metric > float(best_tracker["best_metric"]):
            best_tracker["best_metric"] = float(ev_metric)
            best_tracker["best_metric_name"] = select_metric
            best_tracker["best_top1"] = float(ev["top1"])
            best_tracker["best_top5"] = float(ev["top5"])
            best_tracker["best_acc"] = float(ev["acc"])
            best_tracker["best_macro_f1"] = float(ev["macro_f1"])
            best_tracker["best_loss"] = float(ev["loss"])
            best_tracker["best_stage"] = stage_name
            best_tracker["best_stage_epoch"] = int(epoch)
            best_tracker["best_per_class"] = ev["per_class"]
            best_tracker["best_top_confusions"] = ev["top_confusions"]

            torch.save({
                "stage": stage_name,
                "stage_epoch": int(epoch),
                "model_state": model.state_dict(),
                "best_metric_name": best_tracker["best_metric_name"],
                "best_metric": best_tracker["best_metric"],
                "best_top1": best_tracker["best_top1"],
                "best_top5": best_tracker["best_top5"],
                "best_macro_f1": best_tracker["best_macro_f1"],
                "best_loss": best_tracker["best_loss"],
                "label2id": label2id,
                "id2label": id2label,
            }, best_tracker["best_path"])

            # write compressed confusion summary
            if best_tracker["best_top_confusions"] is not None:
                save_top_confusions_csv(best_tracker["best_top_confusions"], best_tracker["best_confusions_path"], id2label)

            print(
                f"[save] new best by {select_metric} "
                f"(top1={best_tracker['best_top1']:.4f}, macroF1={best_tracker['best_macro_f1']:.4f}) -> {best_tracker['best_path'].name}",
                flush=True,
            )

        # stage-local early stopping
        if early_stop_patience and early_stop_patience > 0:
            if early_stop_metric == "loss":
                cur = float(ev["loss"])
            elif early_stop_metric == "top1":
                cur = float(ev["top1"])
            elif early_stop_metric == "macro_f1":
                cur = float(ev["macro_f1"])
            else:
                raise ValueError("early_stop_metric must be loss|top1|macro_f1")

            if cmp(cur, best_stage):
                best_stage = cur
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= int(early_stop_patience):
                print(f"[early-stop] stage={stage_name} no improvement in {early_stop_metric} for {no_improve} epoch(s); stopping stage.", flush=True)
                break

        # persist history each epoch (like in-domain)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))


# -------------------------
# Main
# -------------------------
@dataclass
class Config:
    data_root: str
    out_dir: str
    checkpoint: str
    arch: str = "resnet50"

    train_frac: float = 0.80
    seed: int = 1337
    img_size: int = 224

    stage: str = "both"  # probe|finetune|both
    unfreeze: str = "layer4"  # layer4|all
    bn_eval_during_probe: bool = True

    probe_epochs: int = 20
    probe_lr: float = 1e-3
    finetune_epochs: int = 100
    finetune_lr: float = 3e-4

    weight_decay: float = 0.01
    label_smoothing: float = 0.1

    batch_size: int = 120
    workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    dataloader_timeout: int = 0
    mp_context: str = "spawn"

    fp16: bool = True

    # selection / stopping
    select_metric: str = "top1"      # top1|top5|macro_f1|acc|loss
    early_stop_patience: int = 10    # stage-local
    early_stop_metric: str = "loss"  # loss|top1|macro_f1

    # outputs
    split_cache: str = ""            # optional npz path
    max_confusions: int = 2000       # top confusions to write (0 disables)
    max_train_items: int = 0         # optional cap for smoke tests
    max_eval_items: int = 0          # optional cap for smoke tests
    log_bad: bool = False
    allow_truncated_images: bool = False  # tolerate truncated/corrupt jpegs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/lab_data/hendersonlab/datasets/VGG-Face2/data/train",
                    help="Folder in ImageFolder format: data_root/<identity>/*.jpg")
    ap.add_argument("--checkpoint", required=True, help="SimCLR checkpoint to initialize encoder")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet50"])

    ap.add_argument("--train-frac", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--img-size", type=int, default=224)

    ap.add_argument("--stage", default="both", choices=["probe", "finetune", "both"])
    ap.add_argument("--unfreeze", default="layer4", choices=["layer4", "all"])
    ap.add_argument("--bn-eval-during-probe", action="store_true", help="Keep BN in eval mode during probe stage")

    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    ap.add_argument("--finetune-epochs", type=int, default=100)
    ap.add_argument("--finetune-lr", type=float, default=3e-4)

    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.1)

    ap.add_argument("-b", "--batch-size", type=int, default=120)
    ap.add_argument("-j", "--workers", type=int, default=8)
    ap.add_argument("--no-pin-memory", action="store_true")
    ap.add_argument("--no-persistent-workers", action="store_true")
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--dataloader-timeout", type=int, default=0)
    ap.add_argument("--mp-context", type=str,
                    default=("fork" if sys.platform.startswith("linux") else "spawn"),
                    choices=["spawn", "forkserver", "fork"])

    ap.add_argument("--no-fp16", action="store_true")

    ap.add_argument("--select-metric", default="top1", choices=["top1", "top5", "macro_f1", "acc", "loss"])
    ap.add_argument("--early-stop-patience", type=int, default=10)
    ap.add_argument("--early-stop-metric", default="loss", choices=["loss", "top1", "macro_f1"])

    ap.add_argument("--split-cache", type=str, default="")
    ap.add_argument("--max-confusions", type=int, default=2000)
    ap.add_argument("--max-train-items", type=int, default=0)
    ap.add_argument("--max-eval-items", type=int, default=0)
    ap.add_argument("--log-bad", action="store_true")
    ap.add_argument("--allow-truncated-images", action="store_true",
                    help="Set PIL.ImageFile.LOAD_TRUNCATED_IMAGES=True to tolerate truncated JPEGs.")

    args = ap.parse_args()

    cfg = Config(
        data_root=args.data_root,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        arch=args.arch,
        train_frac=args.train_frac,
        seed=args.seed,
        img_size=args.img_size,
        stage=args.stage,
        unfreeze=args.unfreeze,
        bn_eval_during_probe=bool(args.bn_eval_during_probe),
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        batch_size=args.batch_size,
        workers=args.workers,
        pin_memory=not args.no_pin_memory,
        persistent_workers=not args.no_persistent_workers,
        prefetch_factor=args.prefetch_factor,
        dataloader_timeout=args.dataloader_timeout,
        mp_context=args.mp_context,
        fp16=not args.no_fp16,
        select_metric=args.select_metric,
        early_stop_patience=args.early_stop_patience,
        early_stop_metric=args.early_stop_metric,
        split_cache=args.split_cache,
        max_confusions=args.max_confusions,
        max_train_items=args.max_train_items,
        max_eval_items=args.max_eval_items,
        log_bad=bool(args.log_bad),
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    # DataLoader sanity
    if cfg.prefetch_factor < 1:
        print("[WARN] prefetch_factor < 1 is invalid; setting to 1", flush=True)
        cfg.prefetch_factor = 1

    seed_everything(cfg.seed)

    if cfg.allow_truncated_images:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        print('[INFO] PIL.ImageFile.LOAD_TRUNCATED_IMAGES=True (tolerating truncated images)', flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}", flush=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Dataset base (directory scan can take a while on VGGFace2)
    t_scan0 = time.time()
    base = ImageFolder(cfg.data_root, transform=None)
    t_scan = time.time() - t_scan0
    print(f"[INFO] ImageFolder scan_sec={t_scan:.1f}", flush=True)
    num_classes = len(base.classes)
    print(f"[INFO] identities={num_classes} images_total={len(base)} arch={cfg.arch}", flush=True)
    if cfg.workers > 0 and cfg.mp_context == "spawn" and len(base) > 500000 and sys.platform.startswith("linux"):
        print("[WARN] mp_context=spawn can be very slow/heavy for VGGFace2 due to dataset pickling. "
              "Consider --mp-context fork (default on Linux) for faster startup.", flush=True)

    label2id = {name: i for i, name in enumerate(base.classes)}
    id2label = list(base.classes)
    (out_dir / "label2id.json").write_text(json.dumps(label2id, indent=2))

    # Split cache
    cache_path = Path(cfg.split_cache) if cfg.split_cache else (out_dir / "split_indices.npz")
    if cache_path.exists():
        npz = np.load(cache_path)
        train_idx = npz["train_idx"]
        val_idx = npz["val_idx"]
        print(f"[INFO] loaded split cache: {cache_path} train={len(train_idx)} val={len(val_idx)}", flush=True)
    else:
        train_idx, val_idx = stratified_split_indices_by_class(base.targets, cfg.train_frac, cfg.seed)
        np.savez_compressed(cache_path, train_idx=train_idx, val_idx=val_idx)
        print(f"[INFO] wrote split cache: {cache_path} train={len(train_idx)} val={len(val_idx)}", flush=True)

    if cfg.max_train_items and cfg.max_train_items > 0:
        train_idx = train_idx[: int(cfg.max_train_items)]
    if cfg.max_eval_items and cfg.max_eval_items > 0:
        val_idx = val_idx[: int(cfg.max_eval_items)]

    tx_train = build_transforms("train", cfg.img_size)
    tx_eval = build_transforms("eval", cfg.img_size)

    # Wrap the same ImageFolder twice with different transforms (no re-scan of directory).
    ds_train_full = SafeImageFolder(base, transform=tx_train, log_bad=cfg.log_bad)
    ds_eval_full  = SafeImageFolder(base, transform=tx_eval,  log_bad=cfg.log_bad)

    ds_train = Subset(ds_train_full, train_idx)
    ds_eval = Subset(ds_eval_full, val_idx)

    # DataLoader kwargs (avoid passing prefetch_factor when workers==0)
    dl_common = dict(
        num_workers=cfg.workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.persistent_workers and cfg.workers > 0),
        timeout=cfg.dataloader_timeout,
        worker_init_fn=make_worker_init_fn(cfg.seed + 123),
    )
    if cfg.workers > 0:
        dl_common["prefetch_factor"] = cfg.prefetch_factor
        dl_common["multiprocessing_context"] = cfg.mp_context

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **dl_common)
    dl_eval = DataLoader(ds_eval, batch_size=cfg.batch_size, shuffle=False, drop_last=False, **dl_common)

    # Model
    model = ResNetClassifier(arch=cfg.arch, num_classes=num_classes)
    ckpt_report = load_resnet_backbone_from_simclr_checkpoint(model.backbone, cfg.checkpoint, verbose=True)
    model = model.to(device)

    # Fingerprint + config snapshot (like in-domain)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_fingerprint = {
        "command": " ".join(sys.argv),
        "timestamp": stamp,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": cfg.seed,
        "data_root": str(cfg.data_root),
        "n_total": int(len(base)),
        "n_train": int(len(train_idx)),
        "n_eval": int(len(val_idx)),
        "checkpoint": str(cfg.checkpoint),
        "checkpoint_sha256": sha256_file(cfg.checkpoint),
        "ckpt_load_report": asdict(ckpt_report),
    }
    (out_dir / "run_fingerprint.json").write_text(json.dumps(run_fingerprint, indent=2, default=str))

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.label_smoothing))

    # Best tracker (global across stages)
    best_tracker = {
        "best_metric": -float("inf"),
        "best_metric_name": cfg.select_metric,
        "best_stage": "",
        "best_stage_epoch": 0,
        "best_path": out_dir / "best_checkpoint.pt",
        "best_confusions_path": out_dir / "best_top_confusions.csv",
        "best_top1": -1.0,
        "best_top5": -1.0,
        "best_loss": float("inf"),
        "best_macro_f1": -1.0,
        "best_acc": -1.0,
        "best_per_class": None,
        "best_top_confusions": None,
    }
    history: List[Dict[str, object]] = []
    (out_dir / "history.json").write_text("[]")

    print(f"[run] out_dir={out_dir}", flush=True)

    # Stage: probe
    if cfg.stage in ("probe", "both"):
        set_backbone_requires_grad(model, train_backbone=False)
        print("[stage] PROBE: frozen backbone, train fc only.", flush=True)
        train_stage(
            stage_name="probe",
            model=model,
            dl_train=dl_train,
            dl_eval=dl_eval,
            device=device,
            criterion=criterion,
            num_classes=num_classes,
            epochs=int(cfg.probe_epochs),
            lr=float(cfg.probe_lr),
            weight_decay=float(cfg.weight_decay),
            fp16=bool(cfg.fp16),
            bn_eval=bool(cfg.bn_eval_during_probe),
            out_dir=out_dir,
            label2id=label2id,
            id2label=id2label,
            best_tracker=best_tracker,
            history=history,
            select_metric=str(cfg.select_metric),
            early_stop_patience=int(cfg.early_stop_patience),
            early_stop_metric=str(cfg.early_stop_metric),
            max_confusions=int(cfg.max_confusions),
        )

    # Stage: finetune
    if cfg.stage in ("finetune", "both"):
        set_backbone_requires_grad(model, train_backbone=True, unfreeze=cfg.unfreeze)
        print(f"[stage] FINETUNE: unfreeze={cfg.unfreeze} (plus fc).", flush=True)
        train_stage(
            stage_name="finetune",
            model=model,
            dl_train=dl_train,
            dl_eval=dl_eval,
            device=device,
            criterion=criterion,
            num_classes=num_classes,
            epochs=int(cfg.finetune_epochs),
            lr=float(cfg.finetune_lr),
            weight_decay=float(cfg.weight_decay),
            fp16=bool(cfg.fp16),
            bn_eval=False,
            out_dir=out_dir,
            label2id=label2id,
            id2label=id2label,
            best_tracker=best_tracker,
            history=history,
            select_metric=str(cfg.select_metric),
            early_stop_patience=int(cfg.early_stop_patience),
            early_stop_metric=str(cfg.early_stop_metric),
            max_confusions=int(cfg.max_confusions),
        )

    # Final eval (on current in-memory model)
    final_eval = evaluate(
        model=model,
        dl=dl_eval,
        device=device,
        criterion=criterion,
        num_classes=num_classes,
        amp=bool(cfg.fp16),
        max_confusions=int(cfg.max_confusions),
    )

    # Save final checkpoint
    torch.save({
        "stage": cfg.stage,
        "model_state": model.state_dict(),
        "final_eval": {
            "loss": float(final_eval["loss"]),
            "top1": float(final_eval["top1"]),
            "top5": float(final_eval["top5"]),
            "macro_f1": float(final_eval["macro_f1"]),
        },
        "best": {
            "metric_name": best_tracker.get("best_metric_name", None),
            "metric": best_tracker.get("best_metric", None),
            "top1": best_tracker.get("best_top1", None),
            "top5": best_tracker.get("best_top5", None),
            "loss": best_tracker.get("best_loss", None),
            "macro_f1": best_tracker.get("best_macro_f1", None),
            "stage": best_tracker.get("best_stage", None),
            "stage_epoch": best_tracker.get("best_stage_epoch", None),
        },
        "label2id": label2id,
        "id2label": id2label,
        "args": vars(args),
        "fingerprint": run_fingerprint,
    }, out_dir / "final_checkpoint.pt")

    # Write metrics.json (in-domain style)
    metrics = {
        "best": {
            "metric_name": best_tracker.get("best_metric_name", None),
            "metric": best_tracker.get("best_metric", None),
            "top1": best_tracker.get("best_top1", None),
            "top5": best_tracker.get("best_top5", None),
            "loss": best_tracker.get("best_loss", None),
            "macro_f1": best_tracker.get("best_macro_f1", None),
            "stage": best_tracker.get("best_stage", None),
            "stage_epoch": best_tracker.get("best_stage_epoch", None),
            "per_class": best_tracker.get("best_per_class", None),
            "top_confusions_csv": str(Path(best_tracker["best_confusions_path"]).name),
        },
        "final": {
            "loss": float(final_eval["loss"]),
            "top1": float(final_eval["top1"]),
            "top5": float(final_eval["top5"]),
            "macro_f1": float(final_eval["macro_f1"]),
            "per_class": final_eval["per_class"],
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save final compressed confusions too (handy)
    if final_eval.get("top_confusions") is not None and cfg.max_confusions and cfg.max_confusions > 0:
        save_top_confusions_csv(final_eval["top_confusions"], out_dir / "final_top_confusions.csv", id2label)

    print(
        f"[done] best_top1={best_tracker['best_top1']:.4f} best_stage={best_tracker['best_stage']} "
        f"final_top1={final_eval['top1']:.4f} final_macroF1={final_eval['macro_f1']:.4f}",
        flush=True,
    )
    print(f"[done] outputs in: {out_dir}", flush=True)


if __name__ == "__main__":
    main()

