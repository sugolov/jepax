import jax
from jax import numpy as jnp
import numpy as np
import einops

import equinox as eqx
from equinox.nn import Linear

from typing import Optional
from jaxtyping import Float, Array, PRNGKeyArray

from jepax.model.masker import IJEPAMasker, set_token_mask
from jepax.model.transformer import Transformer

class PatchEmbedding(eqx.Module):
    linear: eqx.nn.Embedding
    patch_size: int

    def __init__(
        self,
        input_channels: int,
        output_shape: int,
        patch_size: int,
        key: PRNGKeyArray,
    ):
        self.patch_size = patch_size

        self.linear = eqx.nn.Linear(
            self.patch_size**2 * input_channels,
            output_shape,
            key=key,
        )

    def __call__(
        self, x: Float[Array, "channels height width"]
    ) -> Float[Array, "num_patches embedding_dim"]:
        x = einops.rearrange(
            x,
            "(h ph) (w pw) c -> (h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        x = jax.vmap(self.linear)(x)

        return x

class ViTclassifier(eqx.Module):
    embed: PatchEmbedding
    transformer: Transformer
    clf: Linear
    cls_token: jax.Array


    def __init__(
        self,
        num_channels: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        num_layers: int,
        num_head: int,
        mlp_ratio: float = 3.0,
        p_drop: float = 0.1,
        seq_len: int = 2048,
        *,
        key: PRNGKeyArray
    ):
        k1, k2, k3, k4 = jax.random.split(key, 4)
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
        self.clf = eqx.nn.Linear(dim, num_classes, key=k3)
        self.cls_token = jax.random.normal(k4, (1, dim))

    def __call__(self, x, key, train=True):
        x = self.embed(x)
        x = jnp.concatenate([self.cls_token, x], axis=0)
        x = self.transformer(x, key=key, train=train)
        logits = self.clf(x[0])
        return logits
    
class JEPAViT(eqx.Module):
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


    def __call__(self, x, key, mask, ctx_mask=False, train=True):
        x = self.embed(x)

        if ctx_mask:
            x = set_token_mask(x, mask.flatten(), self.ctx_mask)

        x = self.transformer(x, key=key, train=train)

        return x
    

class JEPAViT(eqx.Module):
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


    def __call__(self, x, key, mask, ctx_mask=False, train=True):
        x = self.embed(x)

        if ctx_mask:
            x = set_token_mask(x, mask.flatten(), self.ctx_mask)

        x = self.transformer(x, key=key, train=train)

        return x