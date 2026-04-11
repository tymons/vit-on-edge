# optim_2 — NPU-Compatible Tensor Reshaping (Einops Removal)

Based on: **optim_1** (SiLU → ReLU6 activation replacement)

## Summary

All uses of `einops.rearrange` are eliminated and replaced with explicit
PyTorch primitives that map directly to TFLite operators natively supported
by the Ethos-U NPU.  Two components are affected: the patch folding/unfolding
pipeline in `MobileViTBlock` and the multi-head projection in `Attention`.
No changes are made to layer types, channel widths, or parameter count.

## Changed locations

### 1. `MobileViTBlock.forward` — patch unfolding and folding

| | Before (optim_1) | After (optim_2) |
|---|---|---|
| Import | `from einops import rearrange` | `import torch.nn.functional as F` |
| Unfold | `rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=ph, pw=pw)` | `F.pixel_unshuffle(x, ph)` + reshape + permute |
| Fold | `rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', ...)` | permute + reshape + `F.pixel_shuffle(x, ph)` |
| Max tensor rank | 5D (implicit inside einops) | ≤ 4D at every step |

**Unfold sequence (SPACE_TO_DEPTH):**
```
[B, D, H, W]
  → pixel_unshuffle(ph)   # [B, D·ph·pw, nh, nw]  — SPACE_TO_DEPTH native op
  → flatten(2)            # [B, D·ph·pw, nh·nw]
  → reshape(B, D, ph·pw, nh·nw)  # [B, D, ph·pw, nh·nw]
  → permute(0,2,3,1)      # [B, ph·pw, nh·nw, D]  → transformer input
```

**Fold sequence (DEPTH_TO_SPACE):**
```
[B, ph·pw, nh·nw, D]
  → permute(0,3,1,2)      # [B, D, ph·pw, nh·nw]
  → reshape(B, D·ph·pw, nh, nw)  # [B, D·ph·pw, nh, nw]
  → pixel_shuffle(ph)     # [B, D, H, W]  — DEPTH_TO_SPACE native op
```

### 2. `Attention.forward` — multi-head self-attention projection

| | Before (optim_1) | After (optim_2) |
|---|---|---|
| Q/K/V split | `einops.rearrange` → 5D `[B, p, h, n, d]` | `.reshape()` + `.transpose(1,2)` → 4D `[B·p, h, n, d]` |
| Attention matmul | `torch.matmul` on 5D | `torch.matmul` on 4D |
| Output merge | `einops.rearrange` | `.transpose(1,2).reshape(...)` |

The patch and batch dimensions are merged (`B·p`) before the head split,
keeping all intermediate tensors at most 4D.

## Rationale

### `einops.rearrange` and `torch.fold` are not supported on Ethos-U

`einops.rearrange` compiles to `torch.Tensor.view` / `permute` / `contiguous`
combinations.  When exported to TFLite via MLIR, the patch-folding pattern
(`b d (h ph) (w pw) -> b (ph pw) (h w) d`) is lowered to
`STABLEHLO_SCATTER`, which **has no TFLite runtime kernel** and cannot be
offloaded to the Ethos-U command stream.  The entire MobileViTBlock would
therefore be executed on the Cortex-M CPU, negating the benefit of the NPU.

### `pixel_unshuffle` / `pixel_shuffle` map to native TFLite ops

`F.pixel_unshuffle` exports to the `SPACE_TO_DEPTH` TFLite built-in operator,
and `F.pixel_shuffle` exports to `DEPTH_TO_SPACE`.  Both operators are part of
the TFLite built-in op set supported by the Ethos-U55 and U65 NPU drivers.
This allows the entire patch rearrangement to remain inside the NPU compute
pipeline with no CPU fallback.

### Tensor rank constraint (≤ 4D)

The Ethos-U NPU command stream operates exclusively on tensors with rank ≤ 4.
The implicit 5D intermediate tensors produced by einops inside the attention
module (`[B, p, h, n, d]`) force the compiler to insert dimension-collapsing
reshape operators that are not schedulable on the NPU.  Constraining every
intermediate tensor to ≤ 4D ensures all operations can be expressed as native
TFLite operators that the Vela compiler can map to the NPU.
