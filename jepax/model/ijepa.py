import math
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Key

from .vit import PatchEmbedding
from .transformer import TransformerBlock


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> Array:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, -1)

    half_dim = embed_dim // 2
    omega = np.arange(half_dim // 2, dtype=np.float32)
    omega = 1.0 / (10000 ** (omega / (half_dim // 2)))

    pos_h = grid[1][:, None] * omega[None, :]
    emb_h = np.concatenate([np.sin(pos_h), np.cos(pos_h)], axis=1)
    pos_w = grid[0][:, None] * omega[None, :]
    emb_w = np.concatenate([np.sin(pos_w), np.cos(pos_w)], axis=1)

    return jnp.array(np.concatenate([emb_h, emb_w], axis=1))


class IJEPAEncoder(eqx.Module):
    """ViT encoder"""

    patch_embed: PatchEmbedding
    pos_embed: Array
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
        key: Key[Array, ""],
    ):
        keys = jax.random.split(key, depth + 1)
        grid_size = img_size // patch_size

        self.embed_dim = embed_dim
        self.num_patches = grid_size**2
        self.patch_embed = PatchEmbedding(
            in_channels, embed_dim, patch_size, key=keys[0]
        )
        self.pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
        self.blocks = [
            TransformerBlock(
                embed_dim,
                num_heads,
                causal=False,
                mlp_ratio=mlp_ratio,
                p_drop=0.0,
                key=k,
            )
            for k in keys[1:]
        ]
        self.norm = eqx.nn.LayerNorm(embed_dim)

    def __call__(
        self, x: Float[Array, "H W C"], mask_indices: Optional[Array] = None
    ) -> Float[Array, "N D"]:
        x = self.patch_embed(x) + jax.lax.stop_gradient(self.pos_embed)
        if mask_indices is not None:
            x = x[mask_indices]
        for block in self.blocks:
            x = block(x)
        return jax.vmap(self.norm)(x)


class IJEPAPredictor(eqx.Module):
    """Predictor: context + mask tokens -> target predictions."""

    embed_proj: eqx.nn.Linear
    mask_token: Array
    pos_embed: Array
    blocks: list
    norm: eqx.nn.LayerNorm
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        num_patches: int,
        encoder_dim: int = 384,
        predictor_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        *,
        key: Key[Array, ""],
    ):
        keys = jax.random.split(key, depth + 4)
        grid_size = int(math.sqrt(num_patches))

        self.embed_proj = eqx.nn.Linear(encoder_dim, predictor_dim, key=keys[0])
        self.mask_token = jax.random.normal(keys[1], (predictor_dim,)) * 0.02
        self.pos_embed = get_2d_sincos_pos_embed(predictor_dim, grid_size)
        self.blocks = [
            TransformerBlock(
                predictor_dim,
                num_heads,
                causal=False,
                mlp_ratio=mlp_ratio,
                p_drop=0.0,
                key=k,
            )
            for k in keys[2:-1]
        ]
        self.norm = eqx.nn.LayerNorm(predictor_dim)
        self.output_proj = eqx.nn.Linear(predictor_dim, encoder_dim, key=keys[-1])

    def __call__(self, ctx_emb: Array, ctx_idx: Array, tgt_idx: Array) -> Array:
        n_ctx = ctx_emb.shape[0]
        ctx = jax.vmap(self.embed_proj)(ctx_emb) + jax.lax.stop_gradient(
            self.pos_embed[ctx_idx]
        )
        mask_tokens = jnp.tile(
            self.mask_token[None, :], (tgt_idx.shape[0], 1)
        ) + jax.lax.stop_gradient(self.pos_embed[tgt_idx])

        x = jnp.concatenate([ctx, mask_tokens], axis=0)
        for block in self.blocks:
            x = block(x)
        x = jax.vmap(self.norm)(x)
        return jax.vmap(self.output_proj)(x[n_ctx:])


class IJEPA(eqx.Module):
    encoder: IJEPAEncoder
    predictor: IJEPAPredictor

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
        key: Key[Array, ""],
    ):
        k1, k2 = jax.random.split(key)
        num_patches = (img_size // patch_size) ** 2

        self.encoder = IJEPAEncoder(
            img_size,
            patch_size,
            in_channels,
            embed_dim,
            encoder_depth,
            num_heads,
            mlp_ratio,
            key=k1,
        )
        self.predictor = IJEPAPredictor(
            num_patches,
            embed_dim,
            predictor_dim,
            predictor_depth,
            num_heads,
            mlp_ratio,
            key=k2,
        )


@eqx.filter_jit
def ema_update(online: eqx.Module, target: eqx.Module, momentum: float) -> eqx.Module:
    """target = momentum * target + (1 - momentum) * online"""
    return jax.tree.map(
        lambda t, o: momentum * t + (1 - momentum) * o if eqx.is_array(t) else t,
        target,
        online,
    )
