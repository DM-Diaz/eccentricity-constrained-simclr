# Note: This was adapted from the run.py in the original PyTorch-SimCLR repo by sthalles.
# Mainly modified args amongst other things. Last checked 8/10/26

import argparse
import os
from typing import List, Optional

import torch
import torch.backends.cudnn as cudnn
from torchvision import models, transforms
from PIL import Image, ImageFile, UnidentifiedImageError

# Keep the original imports for backwards compatibility with the existing repo.
# If --manifest-csv is provided, then we won't use ContrastiveLearningDataset!
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset  # noqa: F401
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR

ImageFile.LOAD_TRUNCATED_IMAGES = True

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description='PyTorch SimCLR')

# Existing args (used by the original folder/STL/CIFAR presets)
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='Dataset root directory (used as frames_root when --manifest-csv is set).')
parser.add_argument('-dataset-name', default='folder',
                    choices=['stl10', 'cifar10', 'folder'],
                    help='Dataset preset: use "folder" for any ImageFolder-style tree (ignored when --manifest-csv is set).')

# New args (manifest-driven loading)
parser.add_argument('--manifest-csv', default=None, type=str,
                    help='Path to frame-level manifest CSV containing a split column.')
parser.add_argument('--include-splits', default='train,val', type=str,
                    help='Comma-separated list of split values to INCLUDE from manifest (default: train,val).')
parser.add_argument('--split-col', default='split', type=str,
                    help='Name of the split column in the manifest (default: split).')
parser.add_argument('--session-col', default='session', type=str,
                    help='Name of the session column (default: session).')
parser.add_argument('--location-col', default='location', type=str,
                    help='Name of the location column (default: location).')
parser.add_argument('--filename-col', default='filename', type=str,
                    help='Name of the filename column (default: filename).')
parser.add_argument('--ok-col', default='ok', type=str,
                    help='Optional boolean "ok" column to filter (default: ok).')
parser.add_argument('--require-ok', action='store_true',
                    help='If set, keep only rows where ok==True (recommended).')
parser.add_argument('--image-size', default=224, type=int,
                    help='Image size for SimCLR augmentations (default: 224).')
parser.add_argument('--path-template', default='{data_root}/{session}/{location}/{filename}', type=str,
                    help=("Python format template for constructing image paths from manifest columns. "
                          "Available keys: data_root, session, location, filename. "
                          "Default: {data_root}/{session}/{location}/{filename}"))
parser.add_argument('--max-rows', default=None, type=int,
                    help='Optional cap on number of rows (debug only).')

# More training args
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--mp-context', default='fork', type=str,
                    choices=['fork', 'spawn', 'forkserver'],
                    help='Multiprocessing context for DataLoader workers (default: fork). Use spawn for stability if workers crash.')
parser.add_argument('--pin-memory', action='store_true',
                    help='Pin CPU memory in DataLoader (recommended for CUDA; default: off).')
parser.add_argument('--persistent-workers', action='store_true',
                    help='Keep DataLoader workers alive between epochs (default: off).')
parser.add_argument('--prefetch-factor', default=2, type=int,
                    help='DataLoader prefetch_factor (only used if workers > 0; default: 2).')
parser.add_argument('--dataloader-timeout', default=0, type=int,
                    help='DataLoader timeout in seconds (0 = no timeout). Useful to fail fast if workers die/hang.')
parser.add_argument('--max-bad-sample-retries', default=10, type=int,
                    help='Retries for bad/missing/corrupt images inside __getitem__ before falling back.')
parser.add_argument('--log-bad-samples', action='store_true',
                    help='Print warnings when a sample fails to load (can be noisy).')
parser.add_argument('--gpu-aug', action='store_true',
                    help=('If set, perform SimCLR augmentations on the GPU (Kornia) in a batched collate_fn. '
                          'This can greatly reduce CPU/DataLoader bottlenecks. Requires kornia.'))
parser.add_argument('--gpu-aug-backend', default='kornia', choices=['kornia'],
                    help='GPU augmentation backend (default: kornia).')

# Even more args
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(',') if x.strip()]


def _get_simclr_transform(image_size: int) -> transforms.Compose:
    """
    Standard SimCLR augmentations (ImageNet-style normalization).
    This is intentionally self-contained so you don't need ImageFolder scanning.
    """
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    return transforms.Compose([
        transforms.RandomResizedCrop(size=image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=int(0.1 * image_size) // 2 * 2 + 1, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


def _get_base_tensor_transform() -> transforms.Compose:
    """Minimal CPU work: decode -> RGB -> tensor in [0,1]. No random aug, no normalization."""
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def _make_gpu_aug_collate(args):
    # Create a collate_fn that builds two SimCLR views on GPU using Kornia.

    # Assumes the dataset returns (img_tensor_cpu, dummy_label) where img_tensor_cpu is float32 in [0,1], shape [3,H,W].
    # Returns the same structure SimCLR expects: ([view1, view2], labels_tensor).

    if args.device.type != "cuda":
        raise ValueError("--gpu-aug requires CUDA (device=cuda).")

    if args.gpu_aug_backend != "kornia":
        raise ValueError(f"Unsupported --gpu-aug-backend={args.gpu_aug_backend!r}")

    try:
        import kornia.augmentation as K
    except Exception as e:
        raise ImportError(
            "GPU aug requested but Kornia is not available. Install it in your env: `pip install kornia`."
        ) from e

    image_size = int(args.image_size)
    k = int(0.1 * image_size) // 2 * 2 + 1  # match torchvision GaussianBlur kernel calc

    aug = torch.nn.Sequential(
        K.RandomResizedCrop((image_size, image_size), scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0), p=1.0),
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8),
        K.RandomGrayscale(p=0.2),
        K.RandomGaussianBlur((k, k), (0.1, 2.0), p=1.0),
    ).to(args.device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1, 3, 1, 1)

    def collate_fn(batch):
        imgs, _ = zip(*batch)  # labels are dummy
        x = torch.stack(imgs, dim=0)  # CPU float tensor
        x = x.to(args.device, non_blocking=True)

        v1 = aug(x)
        v2 = aug(x)

        v1 = (v1 - mean) / std
        v2 = (v2 - mean) / std

        labels = torch.zeros((v1.size(0),), device=args.device, dtype=torch.long)
        return [v1, v2], labels

    return collate_fn


def _make_gpu_aug_module(args):
    # Create a Kornia augmentation module to apply on GPU in the main process.

    # Returns (aug_module, mean, std) where mean/std are broadcastable tensors on the device.

    if args.device.type != "cuda":
        raise ValueError("--gpu-aug requires CUDA (device=cuda).")

    if args.gpu_aug_backend != "kornia":
        raise ValueError(f"Unsupported --gpu-aug-backend={args.gpu_aug_backend!r}")

    try:
        import kornia.augmentation as K
    except Exception as e:
        raise ImportError(
            "GPU aug requested but Kornia is not available. Install it in your env: `pip install kornia`."
        ) from e

    image_size = int(args.image_size)
    k = int(0.1 * image_size) // 2 * 2 + 1  # match torchvision GaussianBlur kernel calc

    aug = torch.nn.Sequential(
        K.RandomResizedCrop((image_size, image_size), scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0), p=1.0),
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8),
        K.RandomGrayscale(p=0.2),
        K.RandomGaussianBlur((k, k), (0.1, 2.0), p=1.0),
    ).to(args.device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1, 3, 1, 1)

    return aug, mean, std


class GpuAugLoader:
    # Wrap a CPU DataLoader and produce two SimCLR views on GPU in the MAIN process.

    # This avoids running CUDA ops inside DataLoader worker processes (unsafe).
    # Expects the underlying dataset to return (img_tensor_cpu, label) where img_tensor_cpu is float32 in [0,1].
    # Yields: ([view1, view2], labels) where views are normalized float tensors on GPU.

    def __init__(self, base_loader, aug, mean, std, device, non_blocking: bool = True):
        self.base_loader = base_loader
        self.aug = aug
        self.mean = mean
        self.std = std
        self.device = device
        self.non_blocking = non_blocking

    def __len__(self):
        return len(self.base_loader)

    def __iter__(self):
        for x, y in self.base_loader:
            x = x.to(self.device, non_blocking=self.non_blocking)
            v1 = self.aug(x)
            v2 = self.aug(x)
            v1 = (v1 - self.mean) / self.std
            v2 = (v2 - self.mean) / self.std

            if torch.is_tensor(y):
                y = y.to(self.device, non_blocking=self.non_blocking)
            else:
                y = torch.zeros((v1.size(0),), device=self.device, dtype=torch.long)
            yield [v1, v2], y


class ManifestContrastiveDataset(torch.utils.data.Dataset):

    # Frame-level dataset driven by a manifest CSV.
    # Returns: (views, dummy_label)
    #  - views is a list of length n_views, each a Tensor [C,H,W]

    def __init__(
        self,
        manifest_csv: str,
        frames_root: str,
        n_views: int,
        include_splits: List[str],
        split_col: str = "split",
        session_col: str = "session",
        location_col: str = "location",
        filename_col: str = "filename",
        ok_col: Optional[str] = "ok",
        require_ok: bool = False,
        path_template: str = "{data_root}/{session}/{location}/{filename}",
        image_size: int = 224,
        max_rows: Optional[int] = None,
        max_bad_sample_retries: int = 10,
        log_bad_samples: bool = False,
        return_unaugmented: bool = False,
    ) -> None:
        import pandas as pd

        self.manifest_csv = manifest_csv
        self.frames_root = frames_root
        self.n_views = n_views
        self.include_splits = set(include_splits)
        self.image_size = int(image_size)
        self.return_unaugmented = bool(return_unaugmented)
        self.base_transform = _get_base_tensor_transform()
        self.transform = None if self.return_unaugmented else _get_simclr_transform(self.image_size)
        self.path_template = path_template
        self.max_bad_sample_retries = int(max_bad_sample_retries)
        self.log_bad_samples = bool(log_bad_samples)
        # Fallback image (only used if repeated failures occur)
        self._fallback_img = Image.new('RGB', (self.image_size, self.image_size), (127, 127, 127))

        usecols = {split_col, session_col, filename_col}
        if location_col:
            usecols.add(location_col)
        if require_ok and ok_col:
            usecols.add(ok_col)

        df = pd.read_csv(manifest_csv, usecols=lambda c: c in usecols)

        # Filter splits
        if split_col not in df.columns:
            raise ValueError(f"Manifest missing required column '{split_col}'. Columns: {list(df.columns)}")
        df = df[df[split_col].astype(str).isin(self.include_splits)]

        # Filter ok
        if require_ok and ok_col and ok_col in df.columns:
            ok_series = df[ok_col]
            # handle bool, int, and strings like "True"
            ok_mask = ok_series.astype(str).str.lower().isin(["true", "1", "t", "yes"])
            df = df[ok_mask]

        if max_rows is not None:
            df = df.head(max_rows)

        # Build paths (avoid pandas apply for speed; use vector ops-ish)
        session = df[session_col].astype(str)

        if location_col in df.columns:
            location = df[location_col].fillna("").astype(str)
        else:
            location = ""

        filename = df[filename_col].astype(str)

        # Materialize as plain python list for fast indexing
        self.paths = [
            path_template.format(
                data_root=frames_root,
                session=s,
                location=l,
                filename=f,
            )
            for s, l, f in zip(session.tolist(),
                              location.tolist() if hasattr(location, "tolist") else [location] * len(df),
                              filename.tolist())
        ]

        if len(self.paths) == 0:
            raise ValueError(
                f"No rows left after filtering splits={include_splits} "
                f"(and require_ok={require_ok}). Check manifest + split names."
            )

    def __len__(self) -> int:
        return len(self.paths)
    def __getitem__(self, idx: int):
        # Return two augmented views for contrastive learning.

        # Robust to missing/corrupt images: resamples a different index for a few tries,
        # then falls back to a neutral image so a single bad file never kills a worker.

        n = len(self.paths)
        cur = idx % n

        for attempt in range(self.max_bad_sample_retries + 1):
            path = self.paths[cur]
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    if self.return_unaugmented:
                        x = self.base_transform(img)
                        return x, 0
                    views = [self.transform(img) for _ in range(self.n_views)]
                    return views, 0
            except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as e:
                if self.log_bad_samples and attempt == 0:
                    print(
                        f"[WARN][pid={os.getpid()}] bad sample: {path} "
                        f"({type(e).__name__}: {e})",
                        flush=True
                    )
                # Jump to another index (deterministic-ish) to avoid repeated failure on one path.
                cur = (cur + 1 + (attempt * 1315423911) % max(1, n - 1)) % n
                continue

        # Last resort: neutral fallback sample (keeps training alive)
        if self.return_unaugmented:
            x = self.base_transform(self._fallback_img)
            return x, 0
        views = [self.transform(self._fallback_img) for _ in range(self.n_views)]
        return views, 0




def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two-view training is supported. Please use --n-views 2."

    # Multiprocessing context: use 'spawn' if your DataLoader workers crash or hang.
    import torch.multiprocessing as mp
    try:
        mp.set_start_method(args.mp_context, force=True)
    except RuntimeError:
        # Start method already set in this process.
        pass

    # device
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # dataset
    if args.manifest_csv is not None:
        include_splits = _parse_csv_list(args.include_splits)

        train_dataset = ManifestContrastiveDataset(
            manifest_csv=args.manifest_csv,
            frames_root=args.data,
            n_views=args.n_views,
            include_splits=include_splits,
            split_col=args.split_col,
            session_col=args.session_col,
            location_col=args.location_col,
            filename_col=args.filename_col,
            ok_col=args.ok_col,
            require_ok=args.require_ok,
            path_template=args.path_template,
            image_size=args.image_size,
            max_rows=args.max_rows,
            max_bad_sample_retries=args.max_bad_sample_retries,
            log_bad_samples=args.log_bad_samples,
            return_unaugmented=bool(args.gpu_aug),
        )
        print(f"[Manifest dataset] n={len(train_dataset):,} rows "
              f"(splits={include_splits}, root={args.data})")
    else:
        # Original behavior (folder/cifar/stl presets)
        dataset = ContrastiveLearningDataset(args.data)
        train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
    pin_memory = bool(args.pin_memory) and (args.device.type == 'cuda')
    collate_fn = None
    use_prefetch = args.workers > 0

    # Create DataLoader with modern knobs; fall back if this torch build is older.
    if args.workers == 0:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
            drop_last=True,
            collate_fn=collate_fn
        )
    else:
        dl_kwargs = dict(
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=int(args.prefetch_factor),
            timeout=int(args.dataloader_timeout),
            collate_fn=collate_fn,
        )
        # multiprocessing_context is not supported on some older torch builds
        try:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                multiprocessing_context=args.mp_context,
                **dl_kwargs
            )
        except TypeError:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                **dl_kwargs
            )

    # If requested, apply SimCLR augmentations on GPU in the MAIN process.
    # (Do NOT do CUDA ops inside DataLoader worker processes.)
    if args.gpu_aug:
        aug, mean, std = _make_gpu_aug_module(args)
        train_loader = GpuAugLoader(train_loader, aug=aug, mean=mean, std=std,
                                    device=args.device, non_blocking=pin_memory)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
    )

    # It’s a no-op if gpu_index is negative.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
