# optim_0 — Baseline MobileViT (Reference Implementation)

This is the unmodified reference implementation of MobileViT, serving as the
baseline for all subsequent edge-deployment optimizations.  No changes have
been made relative to the original paper.

## Architecture Overview

MobileViT is a lightweight, hybrid vision transformer designed for mobile
applications.  It combines the local feature extraction capabilities of
MobileNetV2-style inverted residual blocks (MV2) with the global context
modelling of a Vision Transformer (ViT).

### Backbone stages

| Stage | Block | Stride | Role |
|-------|-------|--------|------|
| Stem  | `conv_nxn_bn` 3×3 | 2 | Initial feature extraction |
| MV2-1 | `MV2Block` | 1 | Channel expansion, no downsampling |
| MV2-2/3/4 | `MV2Block` | 2 / 1 / 1 | Downsampling + inverted residuals |
| MViT-1 | `MobileViTBlock` | — | Local + global at 1/8 resolution |
| MV2-5 | `MV2Block` | 2 | Downsampling |
| MViT-2 | `MobileViTBlock` | — | Local + global at 1/16 resolution |
| MV2-6 | `MV2Block` | 2 | Downsampling |
| MViT-3 | `MobileViTBlock` | — | Local + global at 1/32 resolution |
| Head  | `conv_1x1_bn` + `AvgPool` + `Linear` | — | Classification |

### MobileViTBlock internals

Each `MobileViTBlock` implements the unfold → transformer → fold pipeline:

1. **Local representation** — two successive convolutions (`conv_nxn_bn`,
   `conv_1x1_bn`) encode local spatial context into a channel embedding.
2. **Patch unfolding** — `einops.rearrange` reshapes the feature map into
   non-overlapping patches of size `ph × pw`, producing a sequence of patch
   tokens for each spatial location.
3. **Global representation** — a multi-head self-attention `Transformer`
   (depth L ∈ {2, 4, 3} per stage, 4 heads, dim_head = 8) processes the
   unfolded patches, allowing every spatial location to attend to all others.
4. **Patch folding** — the attended tokens are rearranged back to the spatial
   feature map layout.
5. **Fusion** — the locally encoded skip connection is concatenated with the
   globally refined map and fused through a final `conv_nxn_bn`.

### Activations

All convolutional blocks use **SiLU** (Swish, σ(x)·x).  The `FeedForward`
MLP inside the transformer also uses SiLU.

### Tensor operations

Patch folding and unfolding rely on `einops.rearrange`, which internally
transposes tensors and may produce **5-dimensional** intermediate tensors
(shape `[B, p, h, n, d]` inside the attention module).

### Variants

| Variant | dims | channels | Parameters |
|---------|------|----------|------------|
| XXS | [64, 80, 96] | [16,16,24,24,48,48,64,64,80,80,320] | ~1.3 M |
| XS  | [96, 120, 144] | [16,32,48,48,64,64,80,80,96,96,384] | ~2.3 M |
| S   | [144, 192, 240] | [16,32,64,64,96,96,128,128,160,160,640] | ~5.6 M |
