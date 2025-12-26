<!-- AI SLOP GENERATION -->

# I-JEPA Masking & Patching Explained

## Architecture

```
Image
  │
  ├────────────────────────────┐
  │                            │
  ▼                            ▼
Context Encoder            Target Encoder (EMA)
(~50 patches)              (all 64 patches)
  │                            │
  ▼                            │
context_emb [N_ctx, D]         │
  │                            │
  ▼                            │
Predictor                      │
[N_ctx + N_tgt, D]             │
  │                            │
  ▼                            ▼
predictions [N_tgt, D]    targets [N_tgt, D]
  │                            │
  └───────────┬────────────────┘
              ▼
         Smooth L1 Loss
```

---

## 1. No Padding Tokens

Images are resized to resolutions that are exact multiples of the patch size, so the ViT sees a clean patch grid and a fixed sequence length.

| Resolution | Patch Size | Grid | Patches |
|------------|------------|------|---------|
| 224×224    | 16×16      | 14×14| 196     |
| 224×224    | 14×14      | 16×16| 256     |
| 448×448    | 16×16      | 28×28| 784     |
| 64×64      | 8×8        | 8×8  | 64      |

No ragged edges → no padding needed.

## 2. Masks Are Index Lists, Not 2D Grids

A "block" is just a set of patch indices. The 2D shape doesn't matter to the model.

```python
# A 2x2 target block might be:
target_indices = [5, 6, 9, 10]

# Context (Swiss cheese after removing targets):
context_indices = [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]
```

The model sees a **list of tokens**, not a 2D mask tensor. Geometry is encoded in positional embeddings.

## 3. Context vs Target: No Overlap in Practice

The sampling process:

1. Sample 4 target blocks (small rectangles, scale ~0.15-0.2)
2. Sample 1 large context block (scale ~0.85-1.0)
3. **Remove overlap**: `context = context_block - union(target_blocks)`

After removal, context and target indices are **disjoint**.

## 4. What Each Network Sees

| Network | Input | Output |
|---------|-------|--------|
| **Target Encoder** | ALL patches `[N, D]` | Full representations `[N, D]` |
| **Context Encoder** | Only context patches `[N_ctx, D]` | Context representations `[N_ctx, D]` |
| **Predictor** | Context + mask tokens `[N_ctx + N_tgt, D]` | Predictions at target positions `[N_tgt, D]` |

## 5. The `apply_masks` Operation

The key function that selects patches by index:

```python
def apply_masks(x, indices):
    """x: [N, D], indices: [K] -> [K, D]"""
    return x[indices]
```

No padding, no attention masks. Just indexing.

## 6. How the Predictor Works

The predictor combines:
1. **Context tokens**: Encoded patches + positional embeddings at context positions
2. **Mask tokens**: One learned vector repeated for each target, + positional embeddings at target positions

```python
# Context tokens with positions
ctx_with_pos = context_embeddings + pos_embed[context_indices]

# Mask tokens: learned vector + target positions
mask_tokens = tile(mask_token, n_targets) + pos_embed[target_indices]

# Predictor input
pred_input = concat([ctx_with_pos, mask_tokens])  # [N_ctx + N_tgt, D]

# After transformer
pred_output = predictor(pred_input)  # [N_ctx + N_tgt, D]

# Extract predictions (last N_tgt positions)
predictions = pred_output[N_ctx:]  # [N_tgt, D]
```

The mask tokens say: "Predict what should be **at these positions**."

## 7. 2D Positional Embeddings

I-JEPA uses **2D sinusoidal** positional embeddings (not learned). Row and column are encoded separately:

```python
pos_embed[idx] = concat([sin/cos(row), sin/cos(col)])
```

This is how the predictor knows the spatial relationship between context and target patches.

## 8. Loss Computation

```python
# Target encoder output at target positions (no gradients)
targets = stop_gradient(target_encoder(image)[target_indices])

# Predictor output
predictions = predictor(context_embeddings, context_indices, target_indices)

# Loss (Smooth L1 in original, often with normalization)
loss = smooth_l1_loss(predictions, targets)
```

## 9. EMA Update

Only the context encoder and predictor are trained with gradients. The target encoder is updated via exponential moving average:

```python
target_params = momentum * target_params + (1 - momentum) * encoder_params
```

Momentum schedule: 0.996 → 1.0 over training.

## 10. Summary: The Key Insight

**Shapes don't matter once you have indices.**

A "Swiss cheese" context block with holes where targets were removed is just:
```python
context_indices = [0, 1, 2, 4, 8, 11, 12, 15, ...]
```

The model processes a bag of tokens. The 2D structure is recovered through positional embeddings, not through the sequence order or shape of the input.

## 11. Variable Sequence Lengths

**Q: The context encoder sees ~50 patches one time, ~48 another. How does that work?**

Transformers handle variable lengths natively. Self-attention just computes pairwise scores between whatever tokens are present. No padding needed for a single image.

```python
# Image 1: context has 50 patches
context_encoder(patches[[0,1,2,4,5,...]])  # [50, D] in, [50, D] out

# Image 2: context has 48 patches
context_encoder(patches[[0,1,3,4,6,...]])  # [48, D] in, [48, D] out
```

**Batching is where it gets tricky.** To stack into one tensor, you need the same shape. I-JEPA solves this by **truncating to the minimum** context size in the batch:

```python
# From src/masks/multiblock.py:164
min_keep_enc = min(len(mask) for mask in all_context_masks)
collated_masks_enc = [mask[:min_keep_enc] for mask in all_context_masks]
```

If batch has context sizes [50, 48, 52], they all get truncated to 48. Losing 2-4 patches from a ~50 patch context is negligible.

If transformers can handle arbitrary lengths, why do LLMs have max context lengths?

Several reasons:

### Learned Positional Embeddings

Many LLMs use a learned embedding table:

```python
self.pos_embed = nn.Embedding(max_len, dim)  # e.g., max_len=2048
```

Position 2049? No embedding exists. You literally can't index beyond what was allocated.

| Positional Embedding Type | Fixed length? |
|---------------------------|---------------|
| Learned (GPT-2, early LLaMA) | Yes - table has fixed size |
| Sinusoidal (original Transformer, I-JEPA) | No - computed on the fly |
| RoPE, ALiBi | No - computed on the fly |

I-JEPA uses **sinusoidal** embeddings, so no hard limit from positions.

### Training Distribution

Even with RoPE/sinusoidal, models only *trained* on sequences up to length N. Beyond that:
- Attention patterns don't generalize
- Performance degrades, often catastrophically

This is why "context extension" is an active research area.

### Memory: O(N²) Attention

| Context | Attention memory |
|---------|------------------|
| 2K tokens | 4M entries |
| 8K tokens | 64M entries |
| 32K tokens | 1B entries |
| 128K tokens | 16B entries |

### KV Cache During Inference

At inference, you cache keys/values for all previous tokens:

```
Token 1: cache K₁, V₁
Token 2: cache K₁, K₂, V₁, V₂
...
Token N: cache K₁...Kₙ, V₁...Vₙ  # grows linearly with N
```

This is often the real bottleneck - not compute, but memory.

### Summary

| Constraint | Why |
|------------|-----|
| Learned positions | Lookup table has fixed size |
| Training distribution | Model doesn't generalize beyond training length |
| Attention memory | O(N²) blows up |
| KV cache | Linear growth, still huge at long contexts |

Transformers *can* handle arbitrary lengths architecturally - it's positions, training, and memory that impose limits.

---

## Code References (jepax)

| File | What |
|------|------|
| `jepax/masks.py` | `MaskCollator` - samples context/target indices |
| `jepax/model/ijepa.py` | `IJEPAEncoder`, `IJEPAPredictor`, `ema_update` |
| `jepax/model/ijepa.py:16` | `get_2d_sincos_pos_embed` |
| `experiments/train_ijepa.py` | Full training loop (--dataset cifar10\|cifar100\|celeba\|imagenet) |

## Code References (original PyTorch I-JEPA)

| File | What |
|------|------|
| `src/masks/multiblock.py` | `MaskCollator` - block sampling |
| `src/masks/utils.py:11` | `apply_masks` - the gather operation |
| `src/models/vision_transformer.py:220` | `VisionTransformerPredictor` |
| `src/train.py:295-313` | Forward pass and loss |
| `src/train.py:333-336` | EMA update |
