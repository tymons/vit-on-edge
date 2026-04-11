# optim_1 — Activation Replacement: SiLU → ReLU6

Based on: **optim_0** (baseline reference implementation)

## Summary

All **SiLU** (Swish) activation functions are replaced with **ReLU6** throughout
the entire network.  No other changes are made; the macro-architecture,
parameter count, and tensor shapes are identical to optim_0.

## Changed locations

| Location | Before | After |
|----------|--------|-------|
| `conv_1x1_bn()` | `nn.SiLU()` | `nn.ReLU6(inplace=True)` |
| `conv_nxn_bn()` | `nn.SiLU()` | `nn.ReLU6(inplace=True)` |
| `FeedForward.net` (MLP inside transformer) | `nn.SiLU()` | `nn.ReLU6(inplace=True)` |
| `MV2Block.conv` — both expansion paths | `nn.SiLU()` | `nn.ReLU6(inplace=True)` |

## Rationale

### Hardware compatibility

ARM Ethos-U NPUs (U55, U65) and most embedded NPU accelerators expose a
native, zero-overhead **ReLU6** operator that is fused directly into MAC
arrays.  SiLU ($\sigma(x) \cdot x$, where $\sigma$ is the sigmoid function)
requires an approximation or a look-up table that is not available in the
Ethos-U command stream.  As a result, TFLite-Micro falls back to a **CPU
kernel** for every SiLU invocation, breaking the NPU execution pipeline and
introducing DRAM round-trips for every activation layer.

### Quantization friendliness

ReLU6 clips outputs to $[0, 6]$, providing a bounded, well-defined range
that simplifies INT8 fixed-point quantization.  The quantization scale can
be set to $6 / 255 \approx 0.0235$ with zero offset at zero, which minimises
rounding error.  SiLU, being unbounded from below, forces the quantizer to
estimate a wider dynamic range, increasing quantization noise.

### Latency impact

Convolutional layers with fused activations (Conv + BN + Act) are the most
frequent operations in the MV2 backbone.  Keeping the activation NPU-native
allows the entire backbone to remain on the accelerator, removing any
CPU–NPU context switch overhead per layer.
