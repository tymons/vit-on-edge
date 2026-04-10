#!/usr/bin/env python3
"""
MobileViT Training Script
--------------------------
Trains MobileViT-XXS/XS/S on ImageNet or automatically downloads ImageNette
(a free 10-class ImageNet subset) as a fallback.

Usage:
    # Full ImageNet (must be pre-downloaded to ./dataset/imagenet/)
    python train.py --model xxs --dataset-dir ./dataset

    # Auto-download ImageNette (default fallback)
    python train.py --model xxs

    # Resume from a checkpoint
    python train.py --model xxs --resume ./checkpoints/last.pth
"""

import math
import os
import argparse
import tarfile
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder, ImageNet

from mobilevit import MobileViT

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

# ImageNette: a freely available 10-class subset of ImageNet.
# Full ImageNet requires a manual download from https://image-net.org/
IMAGENETTE_URLS = {
    "full": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
    "320":  "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
    "160":  "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
}
IMAGENETTE_NUM_CLASSES = 10
IMAGENET_NUM_CLASSES   = 1000

# ImageNet normalisation constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _download_imagenette(root: Path, size: str = "320") -> Path:
    """Download and extract ImageNette if not already present."""
    url      = IMAGENETTE_URLS[size]
    filename = url.split("/")[-1]
    tgz_path = root / filename
    ds_path  = root / filename.replace(".tgz", "")

    if ds_path.exists():
        print(f"[Dataset] ImageNette already present at {ds_path}")
        return ds_path

    root.mkdir(parents=True, exist_ok=True)
    print(f"[Dataset] Downloading ImageNette ({size}) from {url} …")

    def _progress(count, block, total):
        pct = min(count * block * 100 // total, 100)
        print(f"\r  {pct:3d}%", end="", flush=True)

    urllib.request.urlretrieve(url, tgz_path, _progress)
    print()

    print(f"[Dataset] Extracting {tgz_path} …")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(root)
    tgz_path.unlink()

    print(f"[Dataset] Done — {ds_path}")
    return ds_path


def build_transforms(image_size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),  # ≈ 1.14× crop ratio
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def get_datasets(args):
    """
    Load datasets in priority order:
      1. Full ImageNet from <dataset_dir>/imagenet  (requires manual download)
      2. Auto-download ImageNette as fallback

    Returns (train_dataset, val_dataset, num_classes).
    """
    train_tf, val_tf = build_transforms(args.image_size)
    dataset_root = Path(args.dataset_dir)

    # --- 1. Try full ImageNet ---
    imagenet_root = dataset_root / "imagenet"
    if imagenet_root.exists():
        print(f"[Dataset] Found ImageNet at {imagenet_root}")
        try:
            train_ds = ImageNet(str(imagenet_root), split="train", transform=train_tf)
            val_ds   = ImageNet(str(imagenet_root), split="val",   transform=val_tf)
            return train_ds, val_ds, IMAGENET_NUM_CLASSES
        except Exception as exc:
            print(f"[Dataset] Could not load ImageNet ({exc}); falling back to ImageNette.")

    # --- 2. Auto-download ImageNette ---
    print(
        "[Dataset] Full ImageNet not found.\n"
        "[Dataset] To use it, download it from https://image-net.org/ and place it at:\n"
        f"[Dataset]   {imagenet_root}\n"
        "[Dataset] Falling back to ImageNette (10 classes, freely available) …"
    )
    imagenette_root = _download_imagenette(dataset_root, size=args.imagenette_size)
    train_ds = ImageFolder(str(imagenette_root / "train"), transform=train_tf)
    val_ds   = ImageFolder(str(imagenette_root / "val"),   transform=val_tf)
    return train_ds, val_ds, IMAGENETTE_NUM_CLASSES


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

_MODEL_CFGS = {
    "xxs": dict(dims=[64, 80, 96],        channels=[16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320],  expansion=2),
    "xs":  dict(dims=[96, 120, 144],       channels=[16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],  expansion=4),
    "s":   dict(dims=[144, 192, 240],      channels=[16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640], expansion=4),
}


def build_model(variant: str, image_size: int, num_classes: int) -> nn.Module:
    cfg = _MODEL_CFGS[variant]
    return MobileViT(
        image_size=(image_size, image_size),
        dims=cfg["dims"],
        channels=cfg["channels"],
        num_classes=num_classes,
        expansion=cfg["expansion"],
    )


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

class AverageMeter:
    """Tracks a running mean."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    """Returns top-k accuracy values (%) for each k."""
    with torch.no_grad():
        maxk      = max(topk)
        batch_sz  = target.size(0)
        _, pred   = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred      = pred.t()
        correct   = pred.eq(target.view(1, -1).expand_as(pred))
        results   = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_sz).item())
        return results


def cosine_schedule_with_warmup(optimizer, warmup_epochs: int, total_epochs: int,
                                 base_lr: float, min_lr: float = 1e-6):
    """Linear warmup then cosine annealing."""
    def _lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / base_lr, cosine)

    return optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


def save_checkpoint(state: dict, path: Path, is_best: bool, best_path: Path):
    torch.save(state, path)
    if is_best:
        import shutil
        shutil.copyfile(path, best_path)


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, device_type, epoch, args, writer, num_classes):
    model.train()
    loss_m  = AverageMeter()
    top1_m  = AverageMeter()
    top5_m  = AverageMeter()
    t0      = time.time()
    topk    = (1, min(5, num_classes))

    for step, (images, targets) in enumerate(loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device_type, enabled=args.amp):
            outputs = model(images)
            loss    = criterion(outputs, targets)

        optimizer.zero_grad(set_to_none=True)

        if args.amp and device_type == "cuda":
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        acc1, acc5 = topk_accuracy(outputs.detach(), targets, topk=topk)
        bs = images.size(0)
        loss_m.update(loss.item(), bs)
        top1_m.update(acc1, bs)
        top5_m.update(acc5, bs)

        if step % args.log_freq == 0:
            global_step = epoch * len(loader) + step
            elapsed     = time.time() - t0
            print(
                f"Epoch [{epoch:3d}/{args.epochs}] "
                f"Step [{step:4d}/{len(loader)}]  "
                f"Loss: {loss_m.avg:.4f}  "
                f"Top-1: {top1_m.avg:6.2f}%  "
                f"Top-5: {top5_m.avg:6.2f}%  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}  "
                f"[{elapsed:.0f}s]"
            )
            writer.add_scalar("train/loss",  loss_m.val,                          global_step)
            writer.add_scalar("train/top1",  top1_m.val,                          global_step)
            writer.add_scalar("train/top5",  top5_m.val,                          global_step)
            writer.add_scalar("train/lr",    optimizer.param_groups[0]["lr"],     global_step)

    return loss_m.avg, top1_m.avg


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, writer, num_classes):
    model.eval()
    loss_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    topk   = (1, min(5, num_classes))

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss    = criterion(outputs, targets)

        acc1, acc5 = topk_accuracy(outputs, targets, topk=topk)
        bs = images.size(0)
        loss_m.update(loss.item(), bs)
        top1_m.update(acc1, bs)
        top5_m.update(acc5, bs)

    print(
        f"[Val]  Epoch {epoch:3d}  "
        f"Loss: {loss_m.avg:.4f}  "
        f"Top-1: {top1_m.avg:6.2f}%  "
        f"Top-5: {top5_m.avg:6.2f}%"
    )
    writer.add_scalar("val/loss", loss_m.avg, epoch)
    writer.add_scalar("val/top1", top1_m.avg, epoch)
    writer.add_scalar("val/top5", top5_m.avg, epoch)

    return top1_m.avg


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MobileViT training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--model",     type=str, default="xxs", choices=["xxs", "xs", "s"],
                        help="MobileViT variant")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Input resolution (H=W)")

    # Dataset
    parser.add_argument("--dataset-dir",      type=str, default="./dataset",
                        help="Root directory; ImageNet expected at <dataset-dir>/imagenet")
    parser.add_argument("--imagenette-size",  type=str, default="320",
                        choices=["full", "320", "160"],
                        help="ImageNette variant to auto-download if ImageNet is absent")
    parser.add_argument("--num-classes",      type=int, default=None,
                        help="Override detected number of classes")

    # Training
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=2e-3,
                        help="Peak learning rate")
    parser.add_argument("--min-lr",       type=float, default=1e-6,
                        help="Minimum LR at end of cosine schedule")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs",type=int,   default=5)
    parser.add_argument("--grad-clip",    type=float, default=1.0,
                        help="Max gradient norm (0 = disabled)")
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    # AMP / hardware
    parser.add_argument("--amp",     action="store_true", default=True,
                        help="Use automatic mixed precision (CUDA only)")
    parser.add_argument("--no-amp",  dest="amp", action="store_false")
    parser.add_argument("--workers", type=int, default=8,
                        help="DataLoader worker processes")

    # Logging / checkpointing
    parser.add_argument("--log-freq", type=int, default=50,
                        help="Print & log every N steps")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--log-dir",  type=str, default="./runs",
                        help="TensorBoard log directory")
    parser.add_argument("--resume",   type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # Device
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.type
    print(f"[Device] {device}  (AMP: {args.amp and device_type == 'cuda'})")

    # Datasets & loaders
    train_ds, val_ds, num_classes = get_datasets(args)
    if args.num_classes is not None:
        num_classes = args.num_classes

    print(f"[Dataset] Train: {len(train_ds):,}  Val: {len(val_ds):,}  Classes: {num_classes}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device_type == "cuda"),
        drop_last=True,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device_type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    # Model
    model = build_model(args.model, args.image_size, num_classes).to(device)
    if torch.cuda.device_count() > 1:
        print(f"[Device] Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] MobileViT-{args.model.upper()}  Trainable params: {n_params / 1e6:.2f}M")

    # Loss / optimiser / scheduler / scaler
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = cosine_schedule_with_warmup(
        optimizer, args.warmup_epochs, args.epochs, args.lr, args.min_lr
    )
    scaler = torch.amp.GradScaler(device=device_type, enabled=(args.amp and device_type == "cuda"))

    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    # Checkpoint dirs
    save_dir  = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = save_dir / "last.pth"
    best_ckpt = save_dir / "best.pth"

    start_epoch = 0
    best_top1   = 0.0

    # Resume
    if args.resume:
        print(f"[Resume] Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_top1   = ckpt.get("best_top1", 0.0)
        print(f"[Resume] Epoch {ckpt['epoch']}  best Top-1: {best_top1:.2f}%")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f" Training MobileViT-{args.model.upper()} for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        epoch_t0 = time.time()

        train_loss, train_top1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, device_type, epoch, args, writer, num_classes,
        )
        scheduler.step()
        val_top1 = validate(model, val_loader, criterion, device, epoch, writer, num_classes)

        is_best   = val_top1 > best_top1
        best_top1 = max(val_top1, best_top1)

        ckpt_state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler":    scaler.state_dict(),
            "best_top1": best_top1,
            "args":      vars(args),
        }
        save_checkpoint(ckpt_state, last_ckpt, is_best, best_ckpt)

        elapsed = time.time() - epoch_t0
        tag     = "  ← best" if is_best else ""
        print(
            f"[Epoch {epoch:3d}] "
            f"train_loss={train_loss:.4f}  train_top1={train_top1:.2f}%  "
            f"val_top1={val_top1:.2f}%  best={best_top1:.2f}%  "
            f"[{elapsed:.0f}s]{tag}\n"
        )

    writer.close()
    print(f"Training complete.  Best Val Top-1: {best_top1:.2f}%")
    print(f"Best checkpoint saved to: {best_ckpt}")


if __name__ == "__main__":
    main()
