# MobileViT — Edge Deployment Study

PyTorch implementation of [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178) (arXiv 2021), extended with a progressive optimisation pipeline targeting MCU deployment on ARM Cortex-M + Ethos-U NPU.

---

## Repository layout

```
mobilevit-pytorch/
├── archs/
│   ├── optim_0/          # Baseline reference implementation
│   │   ├── mobilevit.py
│   │   └── CHANGES.md
│   ├── optim_1/          # Activation replacement (SiLU → ReLU6)
│   │   ├── mobilevit.py
│   │   └── CHANGES.md
│   ├── optim_2/          # NPU-compatible tensor reshaping
│   │   ├── mobilevit.py
│   │   └── CHANGES.md
│   └── optim_3/          # Attention head-batch merge
│       ├── mobilevit.py
│       └── CHANGES.md
├── dataset/              # Datasets (ImageNette auto-downloaded here)
├── train.py              # Training script
├── quantization.py       # TFLite export & INT8 quantization script
├── mobilevit.py          # Original reference model (unused by scripts)
└── requirements.txt
```

Each `archs/optim_*/` folder is self-contained: training writes checkpoints to `optim_*/runs/` and quantization writes TFLite models to `optim_*/out/`.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs: `litert-torch`, `torchvision`, `tensorboard`.

---

## Architecture variants — `archs/optim_*/`

Four progressive optimisation steps are studied, each documented in its own `CHANGES.md`:

| Variant | Based on | Key change | `CHANGES.md` |
|---------|----------|------------|--------------|
| `optim_0` | — | Baseline reference implementation | [archs/optim_0/CHANGES.md](archs/optim_0/CHANGES.md) |
| `optim_1` | `optim_0` | All **SiLU → ReLU6** activations | [archs/optim_1/CHANGES.md](archs/optim_1/CHANGES.md) |
| `optim_2` | `optim_1` | **Einops removed**; patch fold/unfold replaced with `pixel_unshuffle` / `pixel_shuffle` (`SPACE_TO_DEPTH` / `DEPTH_TO_SPACE`); attention tensors kept ≤ 4D | [archs/optim_2/CHANGES.md](archs/optim_2/CHANGES.md) |
| `optim_3` | `optim_2` | Attention **head dimension merged into batch**; `torch.matmul` replaced with `torch.bmm`; eliminates unsupported `[0,2,1,3]` TRANSPOSE on Ethos-U | [archs/optim_3/CHANGES.md](archs/optim_3/CHANGES.md) |

### Rationale summary

The optimisation chain targets the ARM **Ethos-U55/U65** NPU command stream and the **TFLite Micro** runtime:

- **optim_1** — `SiLU` is not a native Ethos-U operator and forces a CPU fallback for every activation layer. `ReLU6` is fused directly into MAC arrays and its bounded `[0, 6]` range simplifies INT8 quantization.
- **optim_2** — `einops.rearrange` lowers to `STABLEHLO_SCATTER` in the MLIR → TFLite path, which has no TFLite runtime kernel. `pixel_unshuffle` / `pixel_shuffle` export as the native `SPACE_TO_DEPTH` / `DEPTH_TO_SPACE` ops and keep all tensors ≤ 4D (required by the Ethos-U command set).
- **optim_3** — The `[0,2,1,3]` permutation introduced by splitting attention heads (`transpose(1,2)`) generates a standalone `TRANSPOSE` op that Ethos-U55/U65 cannot accelerate. Merging heads into the batch axis (`B×p×heads`) eliminates this permutation entirely; the only remaining transpose is encoded as `BATCH_MATMUL(adj_y=True)`, which is fully NPU-native.

---

## Training — `train.py`

### Basic usage

```bash
# Train the baseline (optim_0), XXS variant — ImageNette auto-downloaded
python train.py --arch optim_0 --model xxs

# Train the fully optimised variant
python train.py --arch optim_3 --model xxs

# Resume from the last checkpoint
python train.py --arch optim_2 --model xxs --resume archs/optim_2/runs/last.pth
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--arch` | `optim_0` | Architecture to load from `archs/<arch>/mobilevit.py` |
| `--model` | `xxs` | MobileViT size: `xxs`, `xs`, or `s` |
| `--epochs` | `100` | Number of training epochs |
| `--batch-size` | `16` | Batch size |
| `--lr` | `2e-3` | Peak learning rate (cosine schedule with warmup) |
| `--dataset-dir` | `./dataset` | Root dataset directory |
| `--imagenette-size` | `320` | ImageNette variant to download if ImageNet is absent |
| `--save-dir` | `archs/<arch>/runs` | Checkpoint output directory |
| `--log-dir` | `archs/<arch>/runs` | TensorBoard log directory |
| `--resume` | — | Path to a `.pth` checkpoint to resume from |
| `--amp` / `--no-amp` | on | Automatic mixed precision (CUDA only) |

### Dataset resolution order

1. Full ImageNet at `<dataset-dir>/imagenet/` — manual download required from [image-net.org](https://image-net.org/)
2. ImageNette 2 — auto-downloaded to `<dataset-dir>/imagenette2-320/` if ImageNet is not found

### Outputs

All outputs are written to `archs/<arch>/runs/`:

```
archs/optim_2/runs/
├── last.pth      # Checkpoint saved after every epoch
└── best.pth      # Checkpoint of the epoch with highest val Top-1
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir archs/optim_2/runs
```

---

## Quantization & TFLite export — `quantization.py`

For each run the script produces **two TFLite models** and an accuracy summary:

| File | Description |
|------|-------------|
| `mobilevit_<variant>_float.tflite` | Float32 TFLite (no quantization) |
| `mobilevit_<variant>_int8.tflite` | INT8 post-training quantization via representative dataset |
| `accuracy_summary.txt` | Three-model accuracy comparison table |

All files are written to `archs/<arch>/out/`.

### Basic usage

```bash
# Export both TFLite models for optim_3, XXS — uses archs/optim_3/runs/best.pth automatically
python quantization.py --arch optim_3 --variant xxs

# With a real ImageNet validation set for accurate evaluation
python quantization.py --arch optim_3 --variant xxs --imagenet-root /data/imagenet

# Skip accuracy evaluation (export only)
python quantization.py --arch optim_3 --variant xxs --skip-eval

# Override output directory
python quantization.py --arch optim_1 --variant xxs --output-dir ./my_outputs
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--arch` | `optim_0` | Architecture to load from `archs/<arch>/mobilevit.py` |
| `--variant` | `xxs` | MobileViT size: `xxs`, `xs`, or `s` |
| `--checkpoint` | auto | `best`, `last`, or a `.pth` file path |
| `--checkpoint-dir` | `archs/<arch>/runs` | Directory to search for `best.pth` / `last.pth` |
| `--output-dir` | `archs/<arch>/out` | Output directory for TFLite files and summary |
| `--imagenet-root` | — | Real ImageNet root for calibration & evaluation |
| `--num-cal-batches` | `32` | Number of calibration batches for INT8 quantization |
| `--batch-size` | `8` | Calibration / evaluation batch size |
| `--eval-batches` | full set | Limit evaluation to N batches (for quick checks) |
| `--skip-eval` | off | Skip accuracy evaluation; export only |

### Checkpoint resolution

The script resolves `--checkpoint` in this order:

1. `best` / `last` keyword → `<checkpoint-dir>/best.pth` or `last.pth`
2. Explicit file path passed to `--checkpoint`
3. Auto-detect: tries `best.pth` then `last.pth` in `<checkpoint-dir>`
4. No file found → random weights (noted in the summary)

### Calibration dataset resolution

1. `--imagenet-root` if supplied
2. `./dataset/imagenette2-320/train/` if present
3. Synthetic random data as last resort (accuracy results will be meaningless)

### Accuracy summary

After evaluation the script prints and saves a table to `accuracy_summary.txt`:

```
===========================================================
  Accuracy summary — MobileViT-XXS (optim_3)
===========================================================
  Weights : best.pth  (archs/optim_3/runs/best.pth)
===========================================================
  Model                       Top-1      Top-5
  --------------------------  --------  --------
  Float32 (PyTorch)            87.32%    98.10%
  Float32 (TFLite)             87.31%    98.10%
  INT8 (TFLite)                86.95%    97.88%
  --------------------------  --------  --------
  Drop (Float32 TFLite)        -0.01%    0.00%
  Drop (INT8 TFLite)           -0.37%   -0.22%
===========================================================
```

---

## MobileViT size variants

| Variant | `dims` | Parameters |
|---------|--------|------------|
| `xxs` | [64, 80, 96] | ~1.3 M |
| `xs` | [96, 120, 144] | ~2.3 M |
| `s` | [144, 192, 240] | ~5.6 M |

All variants use 256 × 256 input resolution.

---

## Citation

```bibtex
@article{mehta2021mobilevit,
  title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
  author={Mehta, Sachin and Rastegari, Mohammad},
  journal={arXiv preprint arXiv:2110.02178},
  year={2021}
}
```

## Credits

Original MobileViT implementation adapted from [MobileNetV2](https://github.com/tonylins/pytorch-mobilenet-v2) and [ViT](https://github.com/lucidrains/vit-pytorch).
