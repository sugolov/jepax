import jax
from jax import numpy as jnp
import numpy as np
import einops

import equinox as eqx
from equinox.nn import Linear

from typing import Optional
from jaxtyping import Float, Array, PRNGKeyArray

from jepax.model.masker import set_token_mask
from jepax.model.transformer import Transformer, PositionalEncoding
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


    def __call__(self, key, x, mask=None, train=True):
        x = self.embed(x)
        if mask:
            x = set_token_mask(x, mask, self.ctx_mask)

        x = self.transformer(x, key=key, train=train)
        return x
    
class IJEPAPredictor(eqx.Module):
    in_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    transformer: Transformer
    num_mask: int = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        dim: int,
        num_layers: int,
        num_head: int,
        num_mask: int,
        mlp_ratio: float = 3.0,
        p_drop: float = 0.1,
        seq_len: int = 2048,
        *,
        key: PRNGKeyArray
    ):
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.num_mask = num_mask

        self.in_proj = eqx.nn.Linear(latent_dim, dim, key=k1)
        self.transformer = Transformer(
            dim=dim, 
            num_layers=num_layers,
            num_head=num_head,
            mlp_ratio=mlp_ratio,
            p_drop=p_drop,
            seq_len=seq_len,
            key=k2
        )
        self.mask_token = jax.random.normal(k3, (1, dim))
        self.out_proj = eqx.nn.Linear(dim, latent_dim, key=k4)
        self.pe = PositionalEncoding(dim=dim, seq_len=seq_len)

    def __call__(self, key, x: Float[Array, "T D"], mask_pred: Float[Array, "Bm T D"], train=True):
        x = self.pe(x)
        num_mask = mask_pred.shape[0]
        x = jnp.repeat(x[None], num_mask, axis=0)

        x = self.transformer(x, key=key, train=train, use_pe=False)

        return x
