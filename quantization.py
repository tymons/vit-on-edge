"""
MobileViT INT8 Static Quantization using LiteRT-Torch (PT2E).

Pipeline:
  1. Build a PT2EQuantizer (symmetric per-channel weights, static activations).
  2. Export the model with torch.export.
  3. Insert calibration observers via prepare_pt2e.
  4. Run forward passes on an ImageNet calibration subset to populate observer
     statistics (synthetic data is used automatically when no real dataset is
     available).
  5. Fold observers → INT8 quantized graph via convert_pt2e.
  6. Lower to TFLite via litert_torch.convert.
  7. Serialize to a .tflite flatbuffer.

Usage:
    # With real ImageNet validation set:
    python quantization.py --variant s --imagenet-root /data/imagenet

    # With synthetic calibration data (no dataset required):
    python quantization.py --variant s

Dependencies:
    pip install litert-torch torchvision
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

import litert_torch
from litert_torch.quantize.pt2e_quantizer import (
    PT2EQuantizer,
    get_symmetric_quantization_config,
)
from litert_torch.quantize.quant_config import QuantConfig
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from mobilevit import mobilevit_s, mobilevit_xs, mobilevit_xxs

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class AverageMeter:
    """Tracks a running mean."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    """Return top-k accuracy values (%) for each k in *topk*."""
    with torch.no_grad():
        maxk     = max(topk)
        batch_sz = target.size(0)
        _, pred  = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred     = pred.t()
        correct  = pred.eq(target.view(1, -1).expand_as(pred))
        results  = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_sz).item())
        return results


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SIZE: int = 256          # MobileViT expects 256×256 inputs
NUM_CALIBRATION_BATCHES: int = 32
BATCH_SIZE: int = 8

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Default local dataset root and imagenette subfolder.
# Calibration uses <DATASET_DIR>/imagenette2-320/train
# Evaluation  uses <DATASET_DIR>/imagenette2-320/val
DATASET_DIR_DEFAULT  = "./dataset"
IMAGENETTE_SUBDIR    = "imagenette2-320"

# ---------------------------------------------------------------------------
# Calibration dataset helpers
# ---------------------------------------------------------------------------


def imagenet_transforms() -> T.Compose:
    """Standard ImageNet pre-processing compatible with MobileViT (256×256)."""
    return T.Compose([
        T.Resize(int(INPUT_SIZE * 256 / 224)),  # keep aspect, slightly larger
        T.CenterCrop(INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_imagenet_loader(
    root: str,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = 0,  # 0 = main-process loading; avoids os.fork/JAX conflict
) -> DataLoader:
    """DataLoader backed by a real ImageNet validation split."""
    from torchvision.datasets import ImageNet

    dataset = ImageNet(root=root, split="val", transform=imagenet_transforms())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


class SyntheticImageNet(Dataset):
    """
    Synthetic drop-in for ImageNet.

    Generates random tensors that match the shape and approximate distribution
    of ImageNet images after standard normalization.  Useful for quick tests or
    CI environments where the real dataset is unavailable.
    """

    def __init__(self, num_samples: int = NUM_CALIBRATION_BATCHES * BATCH_SIZE):
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # Approximate post-normalization distribution: ~N(0, 1)
        image = torch.randn(3, INPUT_SIZE, INPUT_SIZE)
        label = int(torch.randint(0, 1000, ()).item())
        return image, label
    

def build_calibration_loader(
    imagenet_root: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,  # 0 = main-process loading; avoids os.fork/JAX conflict
) -> DataLoader:
    """
    Return a shuffled DataLoader suitable for INT8 calibration.

    Resolution order:
      1. *imagenet_root* if supplied and valid (full ImageNet).
      2. ``./dataset/imagenette2-320/train`` (fixed local path).
      3. Synthetic random data as a last resort.
    """
    from torchvision.datasets import ImageFolder

    if imagenet_root and os.path.isdir(imagenet_root):
        print(f"[calibration] Using real ImageNet from: {imagenet_root}")
        return build_imagenet_loader(imagenet_root, batch_size=batch_size)

    candidate = Path(DATASET_DIR_DEFAULT) / IMAGENETTE_SUBDIR / "train"
    if candidate.is_dir():
        print(f"[calibration] Using calibration dataset: {candidate}")
        dataset = ImageFolder(str(candidate), transform=imagenet_transforms())
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True,
        )

    print(
        f"[calibration] '{candidate}' not found – using synthetic calibration data.\n"
        "              Pass --imagenet-root <path> or place the dataset under ./dataset/."
    )
    num_samples = NUM_CALIBRATION_BATCHES * batch_size
    dataset_s = SyntheticImageNet(num_samples=num_samples)
    return DataLoader(
        dataset_s, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=True,
    )


def build_val_loader(
    imagenet_root: Optional[str] = None,
    batch_size: int = BATCH_SIZE * 4,
    num_workers: int = 0,
) -> DataLoader:
    """
    Return a non-shuffled DataLoader for accuracy evaluation.

    Resolution order:
      1. *imagenet_root* if supplied and valid (full ImageNet).
      2. ``./dataset/imagenette2-320/val`` (fixed local path).
      3. :class:`SyntheticImageNet` as a last resort (accuracy meaningless).
    """
    from torchvision.datasets import ImageFolder

    if imagenet_root and os.path.isdir(imagenet_root):
        print(f"[eval] Using real ImageNet val split from: {imagenet_root}")
        return build_imagenet_loader(
            imagenet_root, batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
        )

    candidate = Path(DATASET_DIR_DEFAULT) / IMAGENETTE_SUBDIR / "val"
    if candidate.is_dir():
        print(f"[eval] Using validation dataset: {candidate}")
        dataset = ImageFolder(str(candidate), transform=imagenet_transforms())
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False,
        )

    print(
        f"[eval] '{candidate}' not found – accuracy evaluation will use synthetic data.\n"
        "       Results are meaningless without a real validation set.\n"
        "       Pass --imagenet-root <path> or place the dataset under ./dataset/."
    )
    dataset_s = SyntheticImageNet(num_samples=500 * batch_size)
    return DataLoader(
        dataset_s, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

CHECKPOINT_DIR_DEFAULT = "./checkpoints"


def _resolve_checkpoint_path(
    checkpoint: Optional[str],
    checkpoint_dir: str,
    silent: bool = False,
) -> Optional[str]:
    """Resolve 'best' / 'last' / None / literal path to an absolute file path."""
    ckpt_dir = Path(checkpoint_dir)

    if checkpoint is None:
        for name in ("best.pth", "last.pth"):
            candidate = ckpt_dir / name
            if candidate.exists():
                if not silent:
                    print(f"[checkpoint] Auto-detected: {candidate}")
                return str(candidate)
        if not silent:
            print(
                f"[checkpoint] No checkpoint found in '{ckpt_dir}' "
                "(best.pth / last.pth).  Using random weights."
            )
        return None
    elif checkpoint in ("best", "last"):
        path = ckpt_dir / f"{checkpoint}.pth"
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint}' not found at expected path: {path}"
            )
        return str(path)
    return checkpoint  # literal file path


def peek_num_classes(
    checkpoint: Optional[str],
    checkpoint_dir: str = CHECKPOINT_DIR_DEFAULT,
) -> Optional[int]:
    """
    Inspect a checkpoint's ``fc.weight`` tensor to determine the number of
    output classes *without* loading the full state dict into a model.

    Returns ``None`` when no checkpoint file is found.
    """
    path = _resolve_checkpoint_path(checkpoint, checkpoint_dir, silent=True)
    if path is None:
        return None

    raw = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = raw if not isinstance(raw, dict) else raw.get("model", raw.get("state_dict", raw))
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    if "fc.weight" in state_dict:
        num_classes = state_dict["fc.weight"].shape[0]
        print(f"[checkpoint] Detected num_classes={num_classes} from fc.weight")
        return num_classes
    return None


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint: Optional[str],
    checkpoint_dir: str = CHECKPOINT_DIR_DEFAULT,
) -> None:
    """
    Load weights into *model* from a checkpoint file.

    *checkpoint* can be:
      - ``None``    – auto-detect: tries ``best.pth`` then ``last.pth`` inside
                      *checkpoint_dir*; silently uses random weights if neither
                      exists.
      - ``"best"``  – loads ``<checkpoint_dir>/best.pth``
      - ``"last"``  – loads ``<checkpoint_dir>/last.pth``
      - a file path – loaded directly.

    Supported checkpoint formats
    ----------------------------
    * Dict with a ``"model"`` key  (saved by ``train.py``).
    * Dict with a ``"state_dict"`` key (common third-party format).
    * Raw ``state_dict`` (plain mapping of parameter tensors).

    DataParallel ``module.`` prefixes are stripped automatically.
    """
    path = _resolve_checkpoint_path(checkpoint, checkpoint_dir)
    if path is None:
        return  # no checkpoint – keep random weights

    print(f"[checkpoint] Loading from '{path}' …")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        # Print train.py metadata when present
        meta = []
        if "epoch" in ckpt:
            meta.append(f"epoch={ckpt['epoch']}")
        if "best_top1" in ckpt:
            meta.append(f"best_top1={ckpt['best_top1']:.2f}%")
        if meta:
            print(f"[checkpoint] Metadata: {', '.join(meta)}")
    else:
        state_dict = ckpt  # raw state_dict

    # Strip DataParallel 'module.' prefix
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[checkpoint] Warning – {len(missing)} missing key(s): {missing[:3]} …")
    if unexpected:
        print(f"[checkpoint] Warning – {len(unexpected)} unexpected key(s): {unexpected[:3]} …")
    print("[checkpoint] Weights loaded successfully.")


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    desc: str = "eval",
) -> tuple[float, float]:
    """
    Evaluate a float32 *model* on *loader*.

    Inputs are NCHW tensors (standard torchvision layout).
    Returns ``(top1, top5)`` accuracy percentages.
    """
    model = model.eval().to(device)
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    num_classes = None

    for i, (images, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images  = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        if num_classes is None:
            num_classes = outputs.shape[1]
        k = min(5, num_classes)
        acc1, acc5 = topk_accuracy(outputs, targets, topk=(1, k))
        bs = images.size(0)
        top1_m.update(acc1, bs)
        top5_m.update(acc5, bs)
        if (i + 1) % 50 == 0:
            print(f"  [{desc}] batch {i + 1}  top-1: {top1_m.avg:.2f}%  top-5: {top5_m.avg:.2f}%")

    return top1_m.avg, top5_m.avg


@torch.no_grad()
def evaluate_quantized_pt2e(
    pt2e_model: "torch.fx.GraphModule",
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> tuple[float, float]:
    """
    Evaluate a PT2E-quantized graph module on *loader*.

    The exported model has NHWC input (via ``litert_torch.to_channel_last_io``)
    and a static ``batch=1`` dimension.  NCHW images from *loader* are permuted
    to ``[1, H, W, C]`` and fed one at a time.

    Returns ``(top1, top5)`` accuracy percentages.
    """
    pt2e_model = pt2e_model.eval().to(device)
    top1_m     = AverageMeter()
    top5_m     = AverageMeter()
    num_classes = None

    for i, (images, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        # Feed one image at a time (static batch=1 in exported graph)
        batch_outputs = []
        for img in images:                                        # [C, H, W]
            nhwc = img.permute(1, 2, 0).unsqueeze(0).to(device)  # [1, H, W, C]
            out  = pt2e_model(nhwc)                               # [1, num_classes]
            batch_outputs.append(out)
        outputs = torch.cat(batch_outputs, dim=0)                 # [B, num_classes]
        targets = targets.to(device)
        if num_classes is None:
            num_classes = outputs.shape[1]
        k = min(5, num_classes)
        acc1, acc5 = topk_accuracy(outputs, targets, topk=(1, k))
        bs = images.size(0)
        top1_m.update(acc1, bs)
        top5_m.update(acc5, bs)
        if (i + 1) % 10 == 0:
            print(f"  [int8-eval] batch {i + 1}  top-1: {top1_m.avg:.2f}%  top-5: {top5_m.avg:.2f}%")

    return top1_m.avg, top5_m.avg


# ---------------------------------------------------------------------------
# INT8 static quantization
# ---------------------------------------------------------------------------


def _run_calibration(
    model: torch.fx.GraphModule,
    loader: DataLoader,
    num_batches: int,
    device: torch.device,
) -> None:
    """Forward-pass over calibration data to populate observer statistics."""
    # The exported GraphModule has a static batch dimension of 1 (einops
    # rearrange inside MobileViT forces specialization).  We iterate through
    # each loaded batch and feed images one at a time.
    # Images from the DataLoader are NCHW; permute to NHWC to match the
    # channel-last model wrapper's expected input layout.
    seen = 0
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            for img in images:  # img: (3, H, W)
                # NCHW → NHWC: [1, H, W, C]
                nhwc = img.permute(1, 2, 0).unsqueeze(0).to(device)
                model(nhwc)
                seen += 1
            if (i + 1) % 8 == 0:
                print(f"  calibration batch {i + 1}/{num_batches}  ({seen} samples)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize MobileViT to INT8 using LiteRT-Torch (PT2E).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Model ---------------------------------------------------------------
    parser.add_argument(
        "--variant",
        choices=["xxs", "xs", "s"],
        default="s",
        help="MobileViT variant to quantize.",
    )

    # ---- Checkpoint ----------------------------------------------------------
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH|best|last",
        help=(
            "Checkpoint to load weights from.  Accepted values:\n"
            "  'best'   → <checkpoint-dir>/best.pth\n"
            "  'last'   → <checkpoint-dir>/last.pth\n"
            "  <path>   → arbitrary .pth file\n"
            "  (omit)   → auto-detect best.pth then last.pth"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=CHECKPOINT_DIR_DEFAULT,
        metavar="DIR",
        help="Directory to search for best.pth / last.pth.",
    )

    # ---- Dataset -------------------------------------------------------------
    parser.add_argument(
        "--imagenet-root",
        default=None,
        metavar="PATH",
        help=(
            "Root directory of the ImageNet dataset (must contain a 'val/' sub-folder "
            "in the standard ImageFolder layout).  Used for both calibration and "
            "accuracy evaluation.  When omitted, synthetic data is used instead."
        ),
    )

    # ---- Accuracy evaluation -------------------------------------------------
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Limit accuracy evaluation to at most N batches per model "
            "(float32 and INT8).  Handy for quick sanity checks.  "
            "Default: full validation set."
        ),
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        default=False,
        help="Skip accuracy evaluation entirely (calibrate and export only).",
    )

    # ---- Quantization --------------------------------------------------------
    parser.add_argument(
        "--output",
        default="mobilevit_int8.tflite",
        help="Destination path for the quantized .tflite model.",
    )
    parser.add_argument(
        "--num-cal-batches",
        type=int,
        default=NUM_CALIBRATION_BATCHES,
        help="Number of calibration batches.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Calibration (and evaluation) batch size.",
    )

    return parser.parse_args()

def quantize_mobilevit_int8_tflite(
    model: torch.nn.Module,
    output_path: str,
    cal_loader: Optional[DataLoader] = None,
    num_calibration_batches: int = NUM_CALIBRATION_BATCHES,
) -> str:
    """
    Quantize *model* to INT8 via the TFLite converter path and export to
    a ``.tflite`` flatbuffer.

    Parameters
    ----------
    model:
        Float32 MobileViT ``nn.Module``.
    output_path:
        Destination ``.tflite`` file path.
    cal_loader:
        DataLoader that yields ``(images, labels)`` NCHW batches.  When
        provided, real ImageNet samples are used for calibration.  Falls
        back to synthetic data when ``None``.
    num_calibration_batches:
        How many batches to consume for calibration.

    Returns
    -------
    str
        The resolved absolute path to the exported ``.tflite`` file.
    """
    import numpy as np
    import tensorflow as tf  # noqa: F401  (needed for tf.lite.Optimize)

    # litert_torch.convert traces on CPU; move model there regardless of where
    # it was placed during float32 evaluation (e.g. CUDA after evaluate_model).
    model = model.eval().cpu()

    # Wrap with NHWC I/O so litert_torch emits native NHWC CONV2D ops instead
    # of inserting NCHW↔NHWC TRANSPOSE nodes that fall back to CPU on Ethos-U.
    nhwc_model = litert_torch.to_channel_last_io(model, args=[0])

    # Sample args and representative dataset must both use NHWC [B, H, W, C].
    # litert_torch requires float32 I/O (Q/DQ nodes are internal); do NOT set
    # inference_input_type / inference_output_type to tf.int8.
    sample_args = (torch.randn(1, INPUT_SIZE, INPUT_SIZE, 3),)

    def representative_dataset():
        """Yield NHWC [1, H, W, C] float32 numpy arrays for calibration."""
        if cal_loader is not None:
            seen = 0
            for i, (images, _) in enumerate(cal_loader):
                if i >= num_calibration_batches:
                    break
                for img in images:  # img: NCHW [C, H, W]
                    # NCHW → NHWC: [1, H, W, C]
                    nhwc = img.permute(1, 2, 0).unsqueeze(0).numpy().astype(np.float32)
                    yield [nhwc]
                    seen += 1
                if (i + 1) % 8 == 0:
                    print(f"  [calibration] batch {i + 1}/{num_calibration_batches}  ({seen} images)")
        else:
            print("  [calibration] No loader provided – using synthetic calibration data.")
            rng = np.random.default_rng(seed=42)
            for _ in range(num_calibration_batches * BATCH_SIZE):
                data = rng.standard_normal((1, INPUT_SIZE, INPUT_SIZE, 3))
                yield [data.astype(np.float32)]

    tfl_converter_flags = {
        "optimizations": [tf.lite.Optimize.DEFAULT],
        "representative_dataset": representative_dataset,
    }

    print(f"[quantize] Converting with TFLite calibration ({num_calibration_batches} batches) …")
    tfl_drq_model = litert_torch.convert(
        nhwc_model, sample_args, _ai_edge_converter_flags=tfl_converter_flags
    )

    out = str(Path(output_path).resolve())
    tfl_drq_model.export(out)
    size_mb = Path(out).stat().st_size / 1024 / 1024
    print(f"\n✓  Quantized (TFLite INT8) model saved → '{out}'  ({size_mb:.2f} MB)")
    return out


def evaluate_tflite_model(
    tflite_path: str,
    loader: DataLoader,
    max_batches: Optional[int] = None,
) -> tuple[float, float]:
    """
    Evaluate a TFLite model on *loader* using the TFLite interpreter.

    The TFLite model expects NHWC ``[1, H, W, C]`` float32 input (as produced
    by ``quantize_mobilevit_int8_tflite``).  Images from *loader* (NCHW) are
    permuted and fed one at a time to satisfy the static ``batch=1`` shape.

    Returns ``(top1, top5)`` accuracy percentages.
    """
    import numpy as np
    import tensorflow as tf

    print(f"[tflite-eval] Loading interpreter from '{tflite_path}' …")
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_det  = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_idx  = in_det["index"]
    out_idx = out_det["index"]

    top1_m = AverageMeter()
    top5_m = AverageMeter()
    num_classes: Optional[int] = None

    for i, (images, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        batch_logits = []
        for img in images:  # img: NCHW [C, H, W]
            # NCHW → NHWC [1, H, W, C] float32 numpy
            nhwc = img.permute(1, 2, 0).unsqueeze(0).numpy().astype(np.float32)
            interp.set_tensor(in_idx, nhwc)
            interp.invoke()
            out = interp.get_tensor(out_idx)  # [1, num_classes]
            batch_logits.append(torch.from_numpy(out))

        outputs = torch.cat(batch_logits, dim=0)  # [B, num_classes]
        if num_classes is None:
            num_classes = outputs.shape[1]
        k = min(5, num_classes)
        acc1, acc5 = topk_accuracy(outputs, targets, topk=(1, k))
        bs = images.size(0)
        top1_m.update(acc1, bs)
        top5_m.update(acc5, bs)
        if (i + 1) % 10 == 0:
            print(
                f"  [tflite-eval] batch {i + 1}  "
                f"top-1: {top1_m.avg:.2f}%  top-5: {top5_m.avg:.2f}%"
            )

    return top1_m.avg, top5_m.avg
    


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ---- Build model ---------------------------------------------------------
    # Peek at the checkpoint first so we build the model with the correct
    # number of output classes (e.g. 10 for ImageNette, 1000 for ImageNet).
    _DIMS: dict[str, list[int]] = {
        "xxs": [64, 80, 96],
        "xs":  [96, 120, 144],
        "s":   [144, 192, 240],
    }
    _CHANNELS: dict[str, list[int]] = {
        "xxs": [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320],
        "xs":  [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
        "s":   [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
    }
    _EXPANSION: dict[str, int] = {"xxs": 2, "xs": 4, "s": 4}
    num_classes = peek_num_classes(args.checkpoint, args.checkpoint_dir) or 1000
    print(f"[model] Building MobileViT-{args.variant.upper()} (num_classes={num_classes}) …")
    from mobilevit import MobileViT
    model = MobileViT(
        image_size=(256, 256),
        dims=_DIMS[args.variant],
        channels=_CHANNELS[args.variant],
        num_classes=num_classes,
        expansion=_EXPANSION[args.variant],
    )

    # ---- Load checkpoint -----------------------------------------------------
    load_checkpoint(model, args.checkpoint, args.checkpoint_dir)

    # ---- Accuracy evaluation: float32 ----------------------------------------
    val_loader = None
    float_top1: Optional[float] = None
    float_top5: Optional[float] = None

    if not args.skip_eval:
        val_loader = build_val_loader(
            imagenet_root=args.imagenet_root,
            batch_size=args.batch_size * 4,
        )
        print("\n[accuracy] ── Float32 model ──────────────────────────────────")
        float_top1, float_top5 = evaluate_model(
            model, val_loader, device,
            max_batches=args.eval_batches,
            desc="float32",
        )
        print(f"[accuracy] Float32  Top-1: {float_top1:.2f}%  Top-5: {float_top5:.2f}%")

    # ---- Calibration data ----------------------------------------------------
    cal_loader = build_calibration_loader(
        imagenet_root=args.imagenet_root,
        batch_size=args.batch_size,
    )

    # ---- Quantize (TFLite path) + export -------------------------------------
    print("\n[quantize] ── INT8 TFLite quantization ───────────────────────────")
    tflite_path = quantize_mobilevit_int8_tflite(
        model=model,
        output_path=args.output,
        cal_loader=cal_loader,
        num_calibration_batches=args.num_cal_batches,
    )

    # ---- Accuracy evaluation: INT8 TFLite ------------------------------------
    if not args.skip_eval and val_loader is not None:
        print("\n[accuracy] ── INT8 TFLite model ──────────────────────────────")
        int8_top1, int8_top5 = evaluate_tflite_model(
            tflite_path, val_loader,
            max_batches=args.eval_batches,
        )
        print(f"[accuracy] INT8     Top-1: {int8_top1:.2f}%  Top-5: {int8_top5:.2f}%")

        # ---- Accuracy summary ------------------------------------------------
        assert float_top1 is not None and float_top5 is not None
        drop1 = float_top1 - int8_top1
        drop5 = float_top5 - int8_top5
        print()
        print("=" * 57)
        print(f"  {'Model':<22s}  {'Top-1':>8s}  {'Top-5':>8s}")
        print(f"  {'-'*22}  {'-'*8}  {'-'*8}")
        print(f"  {'Float32':<22s}  {float_top1:>7.2f}%  {float_top5:>7.2f}%")
        print(f"  {'INT8 (TFLite)':<22s}  {int8_top1:>7.2f}%  {int8_top5:>7.2f}%")
        print(f"  {'Drop':<22s}  {drop1:>+7.2f}%  {drop5:>+7.2f}%")
        print("=" * 57)


if __name__ == "__main__":
    main()