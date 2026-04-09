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
# Constants
# ---------------------------------------------------------------------------

INPUT_SIZE: int = 256          # MobileViT expects 256×256 inputs
NUM_CALIBRATION_BATCHES: int = 32
BATCH_SIZE: int = 8

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

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
    num_workers: int = 0,  # 0 = main-process loading; avoids os.fork/JAX conflict
) -> DataLoader:
    """DataLoader backed by a real ImageNet validation split."""
    from torchvision.datasets import ImageNet

    dataset = ImageNet(root=root, split="val", transform=imagenet_transforms())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
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
    Return a DataLoader suitable for calibration.

    When *imagenet_root* points to a valid ImageNet directory the real dataset
    is used; otherwise a :class:`SyntheticImageNet` dataset is returned as a
    seamless fallback.
    """
    if imagenet_root and os.path.isdir(imagenet_root):
        print(f"[calibration] Using real ImageNet from: {imagenet_root}")
        return build_imagenet_loader(imagenet_root, batch_size=batch_size)

    print(
        "[calibration] ImageNet root not found – using synthetic calibration data.\n"
        "              Pass --imagenet-root <path> to use a real dataset."
    )
    num_samples = NUM_CALIBRATION_BATCHES * batch_size
    dataset = SyntheticImageNet(num_samples=num_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )


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


def quantize_mobilevit_int8(
    model: torch.nn.Module,
    loader: DataLoader,
    output_path: str = "mobilevit_int8.tflite",
    num_calibration_batches: int = NUM_CALIBRATION_BATCHES,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Quantize *model* to INT8 using the PT2E + LiteRT-Torch pipeline.

    Parameters
    ----------
    model:
        A MobileViT ``nn.Module`` in floating-point precision.
    loader:
        DataLoader that yields ``(images, labels)`` batches used for
        calibration.  Only *images* are fed to the model.
    output_path:
        Destination path for the serialized ``.tflite`` flatbuffer.
    num_calibration_batches:
        How many batches from *loader* to consume during calibration.
    device:
        Torch device to run calibration on (``"cpu"`` is safe; use
        ``"cuda"`` for faster calibration when a GPU is available).
    """
    model = model.eval().to(device)

    # Wrap with channel-last (NHWC) I/O before PT2E export.
    # PyTorch Conv2d is NCHW; without this, litert_torch inserts a
    # TRANSPOSE before and after every Conv2d to satisfy TFLite's NHWC
    # requirement.  PT2E Q/DQ nodes prevent the converter from fusing those
    # transposes into the conv, so they land on CPU.  Tracing the model
    # in NHWC format (input: [B, H, W, C]) lets litert_torch emit native
    # NHWC CONV2D ops directly, eliminating all layout-conversion transposes.
    nhwc_model   = litert_torch.to_channel_last_io(model, args=[0])
    # Sample args now use NHWC layout [B, H, W, C]
    sample_args  = (torch.randn(1, INPUT_SIZE, INPUT_SIZE, 3, device=device),)
    #   • is_per_channel=True  → per-output-channel weight quantization
    #                            (better accuracy than per-tensor)
    #   • is_dynamic=False     → static quantization: activation scales are
    #                            determined once from calibration data and
    #                            baked into the model (INT8 weights + INT8
    #                            activations).
    # ------------------------------------------------------------------
    print("[1/5] Configuring PT2E quantizer (INT8 static, per-channel) …")
    quant_cfg = get_symmetric_quantization_config(
        is_per_channel=True,
        is_dynamic=False,
    )
    quantizer = PT2EQuantizer().set_global(quant_cfg)

    # ------------------------------------------------------------------
    # Step 2 – Export the model with torch.export
    # ------------------------------------------------------------------
    print("[2/5] Exporting model with torch.export …")
    # Export the NHWC-wrapped model. MobileViT's shape arithmetic forces a
    # static batch=1; calibration feeds images one at a time.
    exported_program = torch.export.export(nhwc_model, sample_args)
    pt2e_model: torch.fx.GraphModule = exported_program.module()

    # ------------------------------------------------------------------
    # Step 3 – Insert observer / fake-quant nodes
    # ------------------------------------------------------------------
    print("[3/5] Inserting calibration observers …")
    pt2e_model = prepare_pt2e(pt2e_model, quantizer)

    # ------------------------------------------------------------------
    # Step 4 – Calibration
    # ------------------------------------------------------------------
    print(f"[4/5] Running calibration ({num_calibration_batches} batches) …")
    _run_calibration(pt2e_model, loader, num_batches=num_calibration_batches, device=device)

    # ------------------------------------------------------------------
    # Step 5 – Convert: fold observers → INT8 quantized graph
    # ------------------------------------------------------------------
    print("[5/5] Converting to INT8 quantized graph …")
    pt2e_model = convert_pt2e(pt2e_model, fold_quantize=False)

    # ------------------------------------------------------------------
    # Step 6 – Lower to LiteRT / TFLite
    # ------------------------------------------------------------------
    print("[6/6] Lowering to LiteRT (TFLite) …")
    edge_model = litert_torch.convert(
        pt2e_model,
        sample_args,
        quant_config=QuantConfig(pt2e_quantizer=quantizer),
    )

    # ------------------------------------------------------------------
    # Step 7 – Serialize
    # ------------------------------------------------------------------
    edge_model.export(output_path)
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"\n✓  Quantized model saved → '{output_path}'  ({size_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize MobileViT to INT8 using LiteRT-Torch (PT2E).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant",
        choices=["xxs", "xs", "s"],
        default="s",
        help="MobileViT variant to quantize.",
    )
    parser.add_argument(
        "--imagenet-root",
        default=None,
        metavar="PATH",
        help=(
            "Root directory of the ImageNet dataset (must contain a 'val/' sub-folder "
            "in the standard ImageFolder layout).  When omitted, synthetic calibration "
            "data is used instead."
        ),
    )
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
        help="Calibration batch size.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        metavar="PATH",
        help="Optional path to a .pth checkpoint.  The file may contain a raw "
             "state_dict or a dict with a 'state_dict' key.",
    )
    return parser.parse_args()

def quantize_mobilevit_int8_tflite(model: torch.nn.Module, output_path: str):
    import tensorflow as tf
    import numpy as np

    model = model.eval()

    # Wrap with NHWC I/O so litert_torch emits native NHWC CONV2D ops instead
    # of inserting NCHW↔NHWC TRANSPOSE nodes that fall back to CPU on Ethos-U.
    nhwc_model = litert_torch.to_channel_last_io(model, args=[0])

    # Sample args and representative dataset must both use NHWC [B, H, W, C].
    # litert_torch requires float32 I/O (Q/DQ nodes are internal); do NOT set
    # inference_input_type / inference_output_type to tf.int8.
    sample_args = (torch.randn(1, INPUT_SIZE, INPUT_SIZE, 3),)

    def representative_dataset():
        rng = np.random.default_rng(seed=42)
        for _ in range(100):
            # NHWC: [1, H, W, C] — matches the NHWC-wrapped model's input
            data = rng.standard_normal((1, INPUT_SIZE, INPUT_SIZE, 3))
            yield [data.astype(np.float32)]

    tfl_converter_flags = {
        'optimizations': [tf.lite.Optimize.DEFAULT],
        'representative_dataset': representative_dataset,
    }

    tfl_drq_model = litert_torch.convert(
        nhwc_model, sample_args, _ai_edge_converter_flags=tfl_converter_flags
    )

    tfl_drq_model.export(output_path)
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"\n✓  Quantized (tflite-based) model saved → '{output_path}'  ({size_mb:.2f} MB)")
    


def main() -> None:
    args = parse_args()

    # ---- Model ---------------------------------------------------------------
    factory = {"xxs": mobilevit_xxs, "xs": mobilevit_xs, "s": mobilevit_s}
    print(f"[model] Building MobileViT-{args.variant.upper()} …")
    model = factory[args.variant]()

    if args.weights:
        print(f"[model] Loading weights from '{args.weights}' …")
        ckpt = torch.load(args.weights, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict)

    # ---- Calibration data ----------------------------------------------------
    loader = build_calibration_loader(
        imagenet_root=args.imagenet_root,
        batch_size=args.batch_size,
    )

    # ---- Quantize + export ---------------------------------------------------
    # quantize_mobilevit_int8(
    #     model=model,
    #     loader=loader,
    #     output_path=args.output,
    #     num_calibration_batches=args.num_cal_batches,
    # )

    quantize_mobilevit_int8_tflite(
        model=model,
        output_path=args.output
    )


if __name__ == "__main__":
    main()