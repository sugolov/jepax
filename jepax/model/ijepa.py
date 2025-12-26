"""I-JEPA model components in JAX/Equinox."""

import math
from typing import Optional

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> Array:
    """Generate 2D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension (must be divisible by 4)
        grid_size: Height/width of the patch grid

    Returns:
        pos_embed: [grid_size*grid_size, embed_dim]
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sincos"

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # [2, grid_size, grid_size]
    grid = np.stack(grid, axis=0).reshape(2, -1)  # [2, grid_size*grid_size]

    # Half dimensions for height, half for width
    half_dim = embed_dim // 2
    omega = np.arange(half_dim // 2, dtype=np.float32)
    omega = 1.0 / (10000 ** (omega / (half_dim // 2)))

    # Height embeddings
    pos_h = grid[1][:, None] * omega[None, :]  # [N, half_dim//2]
    emb_h = np.concatenate([np.sin(pos_h), np.cos(pos_h)], axis=1)  # [N, half_dim]

    # Width embeddings
    pos_w = grid[0][:, None] * omega[None, :]  # [N, half_dim//2]
    emb_w = np.concatenate([np.sin(pos_w), np.cos(pos_w)], axis=1)  # [N, half_dim]

    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # [N, embed_dim]
    return jnp.array(pos_embed)


class PatchEmbed(eqx.Module):
    """Patch embedding using linear projection."""

    proj: eqx.nn.Linear
    patch_size: int = eqx.field(static=True)

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, *, key: PRNGKeyArray):
        self.patch_size = patch_size
        self.proj = eqx.nn.Linear(patch_size * patch_size * in_channels, embed_dim, key=key)

    def __call__(self, x: Float[Array, "H W C"]) -> Float[Array, "N D"]:
        """Patchify and embed image.

        Args:
            x: Image [H, W, C]

        Returns:
            patches: [num_patches, embed_dim]
        """
        x = einops.rearrange(
            x,
            "(h ph) (w pw) c -> (h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        return jax.vmap(self.proj)(x)


class MLP(eqx.Module):
    """MLP block for transformer."""

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, dim: int, mlp_ratio: float = 4.0, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = eqx.nn.Linear(dim, hidden_dim, key=k1)
        self.fc2 = eqx.nn.Linear(hidden_dim, dim, key=k2)

    def __call__(self, x: Array) -> Array:
        x = jax.vmap(self.fc1)(x)
        x = jax.nn.gelu(x)
        x = jax.vmap(self.fc2)(x)
        return x


class Attention(eqx.Module):
    """Multi-head self-attention."""

    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear

    def __init__(self, dim: int, num_heads: int = 8, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = eqx.nn.Linear(dim, dim * 3, key=k1)
        self.proj = eqx.nn.Linear(dim, dim, key=k2)

    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N D"]:
        N, D = x.shape

        # QKV projection
        qkv = jax.vmap(self.qkv)(x)  # [N, 3*D]
        qkv = qkv.reshape(N, 3, self.num_heads, self.head_dim)
        qkv = jnp.transpose(qkv, (1, 2, 0, 3))  # [3, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(0, 2, 1)) * self.scale  # [num_heads, N, N]
        attn = jax.nn.softmax(attn, axis=-1)

        # Apply attention to values
        out = attn @ v  # [num_heads, N, head_dim]
        out = jnp.transpose(out, (1, 0, 2)).reshape(N, D)  # [N, D]

        return jax.vmap(self.proj)(out)


class TransformerBlock(eqx.Module):
    """Transformer block with pre-norm."""

    norm1: eqx.nn.LayerNorm
    attn: Attention
    norm2: eqx.nn.LayerNorm
    mlp: MLP

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, key=k1)
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, key=k2)

    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N D"]:
        x = x + self.attn(jax.vmap(self.norm1)(x))
        x = x + self.mlp(jax.vmap(self.norm2)(x))
        return x


class IJEPAEncoder(eqx.Module):
    """Vision Transformer encoder for I-JEPA.

    Can process all patches or a subset specified by mask indices.
    """

    patch_embed: PatchEmbed
    pos_embed: Array  # [num_patches, embed_dim] - frozen sinusoidal
    blocks: list
    norm: eqx.nn.LayerNorm
    embed_dim: int = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, depth + 1)

        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        grid_size = img_size // patch_size

        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size, key=keys[0])
        self.pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, mlp_ratio, key=k)
            for k in keys[1:]
        ]
        self.norm = eqx.nn.LayerNorm(embed_dim)

    def __call__(
        self,
        x: Float[Array, "H W C"],
        mask_indices: Optional[Array] = None
    ) -> Float[Array, "N D"]:
        """Forward pass.

        Args:
            x: Input image [H, W, C]
            mask_indices: Optional indices of patches to keep [K]. If None, use all patches.

        Returns:
            embeddings: [K, embed_dim] if mask_indices provided, else [num_patches, embed_dim]
        """
        # Patchify and embed
        x = self.patch_embed(x)  # [num_patches, embed_dim]

        # Add positional embeddings
        x = x + self.pos_embed

        # Apply mask (select subset of patches)
        if mask_indices is not None:
            x = x[mask_indices]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = jax.vmap(self.norm)(x)

        return x


class IJEPAPredictor(eqx.Module):
    """Predictor module for I-JEPA.

    Takes context embeddings and predicts target embeddings using mask tokens.
    """

    embed_proj: eqx.nn.Linear  # encoder_dim -> predictor_dim
    mask_token: Array  # [predictor_dim] - learned mask token
    pos_embed: Array  # [num_patches, predictor_dim] - frozen sinusoidal
    blocks: list
    norm: eqx.nn.LayerNorm
    output_proj: eqx.nn.Linear  # predictor_dim -> encoder_dim
    predictor_dim: int = eqx.field(static=True)

    def __init__(
        self,
        num_patches: int,
        encoder_dim: int = 384,
        predictor_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, depth + 4)

        self.predictor_dim = predictor_dim
        grid_size = int(math.sqrt(num_patches))

        self.embed_proj = eqx.nn.Linear(encoder_dim, predictor_dim, key=keys[0])
        self.mask_token = jax.random.normal(keys[1], (predictor_dim,)) * 0.02
        self.pos_embed = get_2d_sincos_pos_embed(predictor_dim, grid_size)

        self.blocks = [
            TransformerBlock(predictor_dim, num_heads, mlp_ratio, key=k)
            for k in keys[2:-1]
        ]
        self.norm = eqx.nn.LayerNorm(predictor_dim)
        self.output_proj = eqx.nn.Linear(predictor_dim, encoder_dim, key=keys[-1])

    def __call__(
        self,
        context_embeddings: Float[Array, "N_ctx D_enc"],
        context_indices: Array,  # [N_ctx]
        target_indices: Array,   # [N_tgt]
    ) -> Float[Array, "N_tgt D_enc"]:
        """Predict target embeddings from context.

        Args:
            context_embeddings: Encoded context patches [N_ctx, encoder_dim]
            context_indices: Which patch positions the context embeddings came from
            target_indices: Which patch positions to predict

        Returns:
            predictions: Predicted embeddings for target positions [N_tgt, encoder_dim]
        """
        n_ctx = context_embeddings.shape[0]
        n_tgt = target_indices.shape[0]

        # Project context to predictor dimension
        ctx = jax.vmap(self.embed_proj)(context_embeddings)  # [N_ctx, predictor_dim]

        # Add positional embeddings to context
        ctx = ctx + self.pos_embed[context_indices]

        # Create mask tokens for target positions
        mask_tokens = jnp.tile(self.mask_token[None, :], (n_tgt, 1))  # [N_tgt, predictor_dim]
        mask_tokens = mask_tokens + self.pos_embed[target_indices]

        # Concatenate context and mask tokens
        x = jnp.concatenate([ctx, mask_tokens], axis=0)  # [N_ctx + N_tgt, predictor_dim]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = jax.vmap(self.norm)(x)

        # Extract predictions (last N_tgt tokens)
        preds = x[n_ctx:]  # [N_tgt, predictor_dim]

        # Project back to encoder dimension
        preds = jax.vmap(self.output_proj)(preds)  # [N_tgt, encoder_dim]

        return preds


class IJEPA(eqx.Module):
    """Full I-JEPA model with encoder, predictor, and target encoder (EMA)."""

    encoder: IJEPAEncoder
    predictor: IJEPAPredictor
    # Note: target_encoder is maintained separately as EMA copy

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 384,
        encoder_depth: int = 12,
        predictor_dim: int = 192,
        predictor_depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        *,
        key: PRNGKeyArray,
    ):
        k1, k2 = jax.random.split(key)
        num_patches = (img_size // patch_size) ** 2

        self.encoder = IJEPAEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            key=k1,
        )

        self.predictor = IJEPAPredictor(
            num_patches=num_patches,
            encoder_dim=embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            key=k2,
        )


def ema_update(online: eqx.Module, target: eqx.Module, momentum: float) -> eqx.Module:
    """Update target network with exponential moving average.

    target = momentum * target + (1 - momentum) * online
    """
    def update_leaf(t, o):
        if eqx.is_array(t):
            return momentum * t + (1 - momentum) * o
        return t

    return jax.tree.map(update_leaf, target, online)
