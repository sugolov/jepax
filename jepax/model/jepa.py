import jax
from jax import numpy as jnp
import numpy as np
import einops

import equinox as eqx
from equinox.nn import Linear

from typing import Optional
from jaxtyping import Float, Array, PRNGKeyArray, Key

from jepax.model.masker import set_token_mask
from jepax.model.transformer import Transformer, PositionalEncoding, PositionalEncoding2D
from jepax.model.vit import PatchEmbedding

class IJEPAEncoder(eqx.Module):
    embed: PatchEmbedding
    transformer: Transformer

    def __init__(
        self,
        num_channels: int,
        patch_size: int,
        dim: int,
        num_layers: int,
        num_head: int,
        mlp_ratio: float = 3.0,
        p_drop: float = 0.1,
        seq_len: int = 2048,
        *,
        key: PRNGKeyArray
    ):
        k1, k2, k3 = jax.random.split(key, 3)

        self.embed = PatchEmbedding(num_channels, dim, patch_size, k1)
        self.transformer = Transformer(
            dim=dim, 
            num_layers=num_layers,
            num_head=num_head,
            mlp_ratio=mlp_ratio,
            p_drop=p_drop,
            seq_len=seq_len,
            key=k2
        )
        self.ctx_mask = jax.random.normal(k3, (1, dim))


    def __call__(self, key, x: Array, mask=None, train=True):
        x = self.embed(x)
        if mask:
            x = jnp.where(mask[..., None], x, self.ctx_mask)

        x = self.transformer(x, key=key, train=train)

        return x
    
class IJEPAPredictor(eqx.Module):
    in_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    transformer: Transformer

    pred_token: Array
    mask_token: Array

    pe: PositionalEncoding2D

    def __init__(
        self,
        latent_dim: int,
        dim: int,
        grid_size: int,
        num_layers: int,
        num_head: int,
        mlp_ratio: float = 3.0,
        p_drop: float = 0.1,
        seq_len: int = 2048,
        *,
        key: PRNGKeyArray
    ):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        self.in_proj = eqx.nn.Linear(dim, latent_dim, key=k1)
        self.transformer = Transformer(
            dim=latent_dim, 
            num_layers=num_layers,
            num_head=num_head,
            mlp_ratio=mlp_ratio,
            p_drop=p_drop,
            seq_len=seq_len,
            key=k2
        )
        self.mask_token = jax.random.normal(k3, (1, latent_dim))
        self.pred_token = jax.random.normal(k4, (1, latent_dim))
        self.out_proj = eqx.nn.Linear(dim, latent_dim, key=k5)
        self.pe = PositionalEncoding2D(dim=latent_dim, seq_len=seq_len, grid_size=grid_size)

    #@partial(jax.jit, static_argnames=('num_pad', 'fill_value'))
    def _get_pred_idx(self, mask_pred, num_pad=64, fill_value=-1):
        """Returns grid and flattened indices for predictor tokens
        - mask_idx: indices of tokens from context to be predicted
        - mask_idx_pos: (i,j) grid corresponding to mask_idx
        """
        # flatten last 2 dims 
        mask_pred_flat = mask_pred.reshape(*mask_pred.shape[:1], -1)

        mask_idx = jnp.stack(jnp.where(mask_pred_flat, size=num_pad, fill_value=fill_value)[1:])[0]
        mask_idx_pos = jnp.stack(jnp.where(mask_pred, size=num_pad, fill_value=fill_value)[1:])
        return mask_idx, mask_idx_pos


    #@partial(jax.jit, static_argnames=('num_pad', 'num_prev'))
    def _get_pred_attn_mask(self, mask_idx, num_prev=0):
        # NOTE: mask is 0 for tokens we attend
        attn_mask = jnp.concatenate([jnp.zeros(num_prev), mask_idx < 0])
        return jnp.repeat(attn_mask[None, :], num_prev + len(mask_idx), axis=0).astype(bool)

    def __call__(self, key, z: Float[Array, "T D"], mask_pred: Float[Array, "Bm T D"], num_pad: int, train=True):
        # project tokens
        T = z.shape[0]
        z = jax.vmap(self.in_proj)(z) # vmap proj over token

        # mask_idx:         flattened token indices
        # mask_idx_pos:     (i,j) grid thats good for pos encoding
        mask_idx, mask_idx_pos = self._get_pred_idx(mask_pred, num_pad=num_pad)
        mask_full = mask_pred.any(axis=0).flatten()

        # set mask tokens
        z = jnp.where(mask_full[:, None], self.mask_token, z)
        
        pe_pred = self.pe._get_pe_from_grid(mask_idx_pos)     # positional embedding
        x_pred = jnp.repeat(self.pred_token, num_pad, axis=0) # repeat token for pad length
        x_pred = x_pred + pe_pred

        # concatenate predicted array with current x
        z = jnp.concatenate([z, x_pred], axis=0)

        # mask out padded values in x_pred in forward pass
        attn_mask = self._get_pred_attn_mask(mask_idx, num_prev=T)
        z = self.transformer(z, key=key, train=train, use_pe=False, attn_mask=attn_mask)
        return z, mask_idx
    
class IJEPA(eqx.Module):
    encoder: IJEPAEncoder
    predictor: IJEPAPredictor

    def __init__(self,
        encoder: IJEPAEncoder,
        predictor: IJEPAPredictor):
        self.encoder = encoder
        self.predictor = predictor

    def __call__(self, key: Key, x: Array, mask_ctx, mask_pred, train=True):
        k1, k2, k3 = jax.random.split(key, 3)
        mask_enc = ~(~mask_ctx | jnp.any(mask_pred, axis=0))

        z = self.encoder(x, mask_enc, train=train, key=k1)
        z_full = self.encoder(x, train=train, key=k2)
        z_pred, mask_idx = self.predictor(x, mask_pred=mask_pred, key=k3)

        return z, z_full, z_pred, mask_idx



