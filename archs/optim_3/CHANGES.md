# optim_3 — Attention Head-Batch Merge (Eliminating Unsupported Transposes)

Based on: **optim_2** (NPU-compatible tensor reshaping, einops removal)

## Summary

The multi-head self-attention in `Attention.forward` is restructured so that
the head dimension is fully absorbed into the batch dimension before any
matrix multiplication.  This eliminates the `[0, 2, 1, 3]` permutation
(`transpose(1, 2)`) that was required in optim_2 to move heads before the
sequence length dimension, and replaces `torch.matmul` with `torch.bmm` on
purely 3D tensors.  All other components — `MobileViTBlock`, `MV2Block`,
convolutional helpers, and the top-level `MobileViT` — are unchanged from
optim_2.

## Changed location

### `Attention.forward` — head dimension handling

| | Before (optim_2) | After (optim_3) |
|---|---|---|
| Batch variable | `Bp = B * p` | `BpH = B * p * heads` |
| Q/K/V shape after reshape | `[Bp, n, heads, dh]` then `.transpose(1,2)` → `[Bp, heads, n, dh]` | `[BpH, n, dh]` directly, no transpose |
| Attention matmul | `torch.matmul` on 4D `[Bp, heads, n, n]` | `torch.bmm` on 3D `[BpH, n, n]` |
| Value matmul | `torch.matmul` on 4D `[Bp, heads, n, dh]` | `torch.bmm` on 3D `[BpH, n, dh]` |
| Output merge | `.transpose(1,2).reshape(Bp, n, heads*dh)` | `.reshape(Bp, n, heads*dh)` — no transpose |

**Attention computation (optim_3):**
```
# x: [B, p, n, D]  →  Bp = B*p,  BpH = Bp*heads
qkv  = to_qkv(x.reshape(Bp, n, -1))       # [Bp, n, 3·H·dh]
q, k, v = qkv.chunk(3, dim=-1)            # each [Bp, n, H·dh]

q = q.reshape(BpH, n, dh)                 # [BpH, n, dh]
k = k.reshape(BpH, n, dh)
v = v.reshape(BpH, n, dh)

dots = torch.bmm(q, k.transpose(-1,-2))   # [BpH, n, n]  — adj_y=True ✓
attn = softmax(dots)
out  = torch.bmm(attn, v)                 # [BpH, n, dh]

out  = out.reshape(Bp, n, H*dh)           # merge heads — no permute ✓
```

## Rationale

### The `[0, 2, 1, 3]` permutation is not schedulable on Ethos-U

In optim_2 the Q, K, V tensors were reshaped to `[Bp, n, heads, dh]` and
then transposed with `.transpose(1, 2)` to reach `[Bp, heads, n, dh]`.  This
corresponds to the permutation vector `[0, 2, 1, 3]`.  The Vela compiler
translates arbitrary permutations to a `TRANSPOSE` TFLite op.  While
`TRANSPOSE` is a valid TFLite built-in, Ethos-U55/U65 **cannot accelerate
general-permutation transposes** in the current microNPU command set; the op
is delegated to the Cortex-M CPU, causing an NPU pipeline stall and DRAM
round-trip for every attention layer.

### Merging heads into the batch dimension removes the problematic permutation

By merging `heads` directly into the leading batch axis
(`BpH = B × p × heads`) before any matmul, Q, K, and V remain 3D
`[BpH, n, dh]` at all times.  The only transpose needed is
`k.transpose(-1, -2)`, which permutes the last two axes `[0, 1]→[0, 1]`
`[..., n, dh]→[..., dh, n]`.  TFLite and Vela represent this as
`BATCH_MATMUL` with the `adj_y=True` flag rather than a standalone
`TRANSPOSE` operator, keeping the entire attention computation inside the
NPU accelerated path.

### `torch.bmm` vs `torch.matmul`

`torch.bmm` is the explicit 3D batched matrix-multiply primitive.  It exports
directly to the TFLite `BATCH_MATMUL` built-in, which is natively supported
by Ethos-U.  Using `torch.matmul` on higher-rank tensors can produce
`STABLEHLO_DOT_GENERAL` during MLIR lowering, which may not resolve to
`BATCH_MATMUL` and risks a CPU fallback in edge runtimes.
