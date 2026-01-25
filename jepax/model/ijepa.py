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

# Encoder configs (same as ViT)
ijepa_encoder_configs = {
    "vit-ti": {"dim": 192, "num_layers": 12, "num_head": 3, "mlp_ratio": 4.0},
    "vit-s": {"dim": 384, "num_layers": 12, "num_head": 6, "mlp_ratio": 4.0},
    "vit-b": {"dim": 768, "num_layers": 12, "num_head": 12, "mlp_ratio": 4.0},
    "vit-l": {"dim": 1024, "num_layers": 24, "num_head": 16, "mlp_ratio": 4.0},
    "vit-h": {"dim": 1280, "num_layers": 32, "num_head": 16, "mlp_ratio": 4.0},
    "test": {"dim": 64, "num_layers": 2, "num_head": 2, "mlp_ratio": 2.0},
}

# Predictor configs
ijepa_predictor_configs = {
    "pred-ti": {"latent_dim": 96, "num_layers": 6, "num_head": 3, "mlp_ratio": 4.0},
    "pred-s": {"latent_dim": 192, "num_layers": 6, "num_head": 6, "mlp_ratio": 4.0},
    "pred-b": {"latent_dim": 384, "num_layers": 6, "num_head": 12, "mlp_ratio": 4.0},
    "pred-l": {"latent_dim": 512, "num_layers": 12, "num_head": 16, "mlp_ratio": 4.0},
    "pred-h": {"latent_dim": 640, "num_layers": 12, "num_head": 16, "mlp_ratio": 4.0},
    "test": {"latent_dim": 32, "num_layers": 2, "num_head": 2, "mlp_ratio": 2.0},
}

# Combined IJEPA configs (encoder_name, predictor_name)
ijepa_configs = {
    "ijepa-ti": ("vit-ti", "pred-ti"),
    "ijepa-s": ("vit-s", "pred-s"),
    "ijepa-b": ("vit-b", "pred-b"),
    "ijepa-l": ("vit-l", "pred-l"),
    "ijepa-h": ("vit-h", "pred-h"),
    "ijepa-test": ("test", "test"),
}


def get_encoder_config(
    name: str,
    num_channels: int = 3,
    patch_size: int = 16,
    img_size: int = 224,
    p_drop: float = 0.0,
    seq_len: int = 256,
):
    if name not in ijepa_encoder_configs:
        raise ValueError(f"Unknown encoder config: {name}. Choose from {list(ijepa_encoder_configs.keys())}")
    return {
        **ijepa_encoder_configs[name],
        "num_channels": num_channels,
        "patch_size": patch_size,
        "img_size": img_size,
        "p_drop": p_drop,
        "seq_len": seq_len,
    }


def get_predictor_config(
    name: str,
    enc_dim: int,
    grid_size: int,
    p_drop: float = 0.0,
    seq_len: int = 256,
):
    if name not in ijepa_predictor_configs:
        raise ValueError(f"Unknown predictor config: {name}. Choose from {list(ijepa_predictor_configs.keys())}")
    return {
        **ijepa_predictor_configs[name],
        "dim": enc_dim,  # predictor input dim must match encoder output
        "grid_size": grid_size,
        "p_drop": p_drop,
        "seq_len": seq_len,
    }


def get_ijepa_config(name: str):
    if name not in ijepa_configs:
        raise ValueError(f"Unknown IJEPA config: {name}. Choose from {list(ijepa_configs.keys())}")
    return ijepa_configs[name]


def get_ijepa_model(
    name: str,
    *,
    key: PRNGKeyArray,
    num_channels: int = 3,
    patch_size: int = 16,
    img_size: int = 224,
    p_drop: float = 0.1,
    seq_len: int = 256,
):
    enc_name, pred_name = get_ijepa_config(name)
    
    k1, k2 = jax.random.split(key)
    grid_size = img_size // patch_size
    
    enc_config = get_encoder_config(
        enc_name,
        num_channels=num_channels,
        patch_size=patch_size,
        img_size=img_size,
        p_drop=p_drop,
        seq_len=seq_len,
    )
    
    pred_config = get_predictor_config(
        pred_name,
        enc_dim=enc_config["dim"],
        grid_size=grid_size,
        p_drop=p_drop,
        seq_len=seq_len,
    )
    
    encoder = IJEPAEncoder(**enc_config, key=k1)
    predictor = IJEPAPredictor(**pred_config, key=k2)
    
    return IJEPA(encoder=encoder, predictor=predictor), enc_config["dim"]


# Convenience for custom encoder/predictor combos
def get_ijepa_model_custom(
    enc_name: str,
    pred_name: str,
    *,
    key: PRNGKeyArray,
    num_channels: int = 3,
    patch_size: int = 16,
    img_size: int = 224,
    p_drop: float = 0.0,
    seq_len: int = 256,
):
    k1, k2 = jax.random.split(key)
    grid_size = img_size // patch_size
    
    enc_config = get_encoder_config(
        enc_name,
        num_channels=num_channels,
        patch_size=patch_size,
        img_size=img_size,
        p_drop=p_drop,
        seq_len=seq_len,
    )
    
    pred_config = get_predictor_config(
        pred_name,
        enc_dim=enc_config["dim"],
        grid_size=grid_size,
        p_drop=p_drop,
        seq_len=seq_len,
    )
    
    encoder = IJEPAEncoder(**enc_config, key=k1)
    predictor = IJEPAPredictor(**pred_config, key=k2)
    
    return IJEPA(encoder=encoder, predictor=predictor)

class IJEPAEncoder(eqx.Module):
    embed: PatchEmbedding
    transformer: Transformer
    mask_token: Array

    def __init__(
        self,
        num_channels: int,
        patch_size: int,
        img_size: int,
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
            grid_size=img_size // patch_size,
            pe_type="2d",
            key=k2,
            causal=False
        )
        self.mask_token = jax.random.normal(k3, (1, dim))


    def __call__(self, key, x, mask=None, train=True, get_intermediates=False):
        x = self.embed(x)

        if mask is not None:
            x = set_token_mask(x, mask, self.mask_token)


        out = self.transformer(
            x, key=key, 
            train=train, 
            get_intermediates=get_intermediates
        )
        # returns a list of intermediates if get_intermediates is true
        return out
    
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
            key=k2,
            causal=False
        )
        self.mask_token = jax.random.normal(k3, (1, latent_dim))
        self.pred_token = jax.random.normal(k4, (1, latent_dim))
        self.out_proj = eqx.nn.Linear(latent_dim, dim, key=k5)
        self.pe = PositionalEncoding2D(dim=latent_dim, seq_len=seq_len, grid_size=grid_size)

    #@partial(jax.jit, static_argnames=('num_pad', 'fill_value'))
    def _get_pred_idx(self, mask_pred, num_pad=64, fill_value=-1):
        """Returns grid and flattened indices for predictor tokens
        - mask_idx: indices of tokens from context to be predicted
        - mask_idx_pos: (i,j) grid corresponding to mask_idx

        Args:
            mask_pred (_type_): _description_
            num_pad (int, optional): _description_. Defaults to 64.
            fill_value (int, optional): _description_. Defaults to -1.

        Returns:
            _type_: _description_
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
        z = jax.vmap(self.out_proj)(z)[T:]

        return z, mask_idx
    
class IJEPA(eqx.Module):
    encoder: IJEPAEncoder
    predictor: IJEPAPredictor

    def __init__(self,
        encoder: IJEPAEncoder,
        predictor: IJEPAPredictor):
        self.encoder = encoder
        self.predictor = predictor

    def __call__(self, key: Key, x: Array, mask_ctx, mask_pred, num_pad=256, train=True):
        k1, k2, k3 = jax.random.split(key, 3)
        mask_enc = ~(~mask_ctx | jnp.any(mask_pred, axis=0))

        z = self.encoder(k1, x, mask_enc, train=train)
        # z_full = jax.lax.stop_gradient(self.encoder(k2, x, train=train)) # stop grad
        
        z_pred, mask_idx = self.predictor(k3, z, num_pad=num_pad, mask_pred=mask_pred)

        # return z, z_full, z_pred, mask_idx
        return z, z_pred, mask_idx