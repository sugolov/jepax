from typing import Optional

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Key

from jepax.model.transformer import (
    PositionalEncoding2D,
    Transformer,
)
from jepax.model.vit import PatchEmbedding

ijepa_encoder_configs = {
    "vit-ti": {"dim": 192, "num_layers": 12, "num_head": 3, "mlp_ratio": 4.0},
    "vit-s": {"dim": 384, "num_layers": 12, "num_head": 6, "mlp_ratio": 4.0},
    "vit-b": {"dim": 768, "num_layers": 12, "num_head": 12, "mlp_ratio": 4.0},
    "vit-l": {"dim": 1024, "num_layers": 24, "num_head": 16, "mlp_ratio": 4.0},
    "vit-h": {"dim": 1280, "num_layers": 32, "num_head": 16, "mlp_ratio": 4.0},
    "test": {"dim": 64, "num_layers": 2, "num_head": 2, "mlp_ratio": 2.0},
}

ijepa_predictor_configs = {
    "pred-ti": {"latent_dim": 96, "num_layers": 6, "num_head": 3, "mlp_ratio": 4.0},
    "pred-s": {"latent_dim": 192, "num_layers": 6, "num_head": 6, "mlp_ratio": 4.0},
    "pred-b": {"latent_dim": 384, "num_layers": 6, "num_head": 12, "mlp_ratio": 4.0},
    "pred-l": {"latent_dim": 512, "num_layers": 12, "num_head": 16, "mlp_ratio": 4.0},
    "pred-h": {"latent_dim": 640, "num_layers": 12, "num_head": 16, "mlp_ratio": 4.0},
    "test": {"latent_dim": 32, "num_layers": 2, "num_head": 2, "mlp_ratio": 2.0},
}

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
        raise ValueError(
            f"Unknown encoder config: {name}. "
            f"Choose from {list(ijepa_encoder_configs.keys())}"
        )
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
        raise ValueError(
            f"Unknown predictor config: {name}. "
            f"Choose from {list(ijepa_predictor_configs.keys())}"
        )
    return {
        **ijepa_predictor_configs[name],
        "dim": enc_dim,  # predictor input dim must match encoder output
        "grid_size": grid_size,
        "p_drop": p_drop,
        "seq_len": seq_len,
    }


def get_ijepa_config(name: str):
    if name not in ijepa_configs:
        raise ValueError(
            f"Unknown IJEPA config: {name}. Choose from {list(ijepa_configs.keys())}"
        )
    return ijepa_configs[name]


def fix_init_weight(model):
    def rescale_block(block, layer_id):
        scale = jnp.sqrt(2.0 * (layer_id + 1))
        block = eqx.tree_at(
            lambda b: b.attn.out_proj.weight,
            block,
            block.attn.out_proj.weight / scale,
        )
        block = eqx.tree_at(
            lambda b: b.ff.linear2.weight,
            block,
            block.ff.linear2.weight / scale,
        )
        return block

    # Rescale encoder blocks
    enc_blocks = model.encoder.transformer.blocks
    new_blocks = [rescale_block(b, i) for i, b in enumerate(enc_blocks)]
    model = eqx.tree_at(lambda m: m.encoder.transformer.blocks, model, new_blocks)
    # Rescale predictor blocks
    pred_blocks = model.predictor.transformer.blocks
    new_pred_blocks = [rescale_block(b, i) for i, b in enumerate(pred_blocks)]
    model = eqx.tree_at(
        lambda m: m.predictor.transformer.blocks, model, new_pred_blocks
    )
    return model


def get_ijepa_model(
    name: str,
    *,
    key: Key[Array, ""],
    num_channels: int = 3,
    patch_size: int = 16,
    img_size: int = 224,
    p_drop: float = 0.1,
    seq_len: int = 256,
    gradient_checkpointing: bool = False,
    attn_implementation: Optional[str] = None,
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

    encoder = IJEPAEncoder(
        **enc_config,
        gradient_checkpointing=gradient_checkpointing,
        attn_implementation=attn_implementation,
        key=k1,
    )
    predictor = IJEPAPredictor(
        **pred_config,
        gradient_checkpointing=gradient_checkpointing,
        attn_implementation=attn_implementation,
        key=k2,
    )
    model = IJEPA(encoder=encoder, predictor=predictor)
    model = fix_init_weight(model)
    return model, enc_config["dim"]


def _sincos_embed(positions, dim):
    """Compute sinusoidal embedding from integer positions [N] -> [N, dim]."""
    div_term = jnp.exp(
        jnp.arange(0, dim, 2, dtype=jnp.float32) * (-jnp.log(10000.0) / dim)
    )
    angles = positions[:, None].astype(jnp.float32) * div_term[None, :]
    pe = jnp.stack([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    return pe.reshape(positions.shape[0], dim)


def compute_2d_pe(flat_indices, grid_size, dim, dtype=jnp.float32):
    """Compute 2D sinusoidal PE from flat patch indices."""
    half = dim // 2
    cols = flat_indices % grid_size
    rows = flat_indices // grid_size
    pe = jnp.concatenate(
        [_sincos_embed(cols, half), _sincos_embed(rows, half)], axis=-1
    )
    return jax.lax.stop_gradient(pe.astype(dtype))


def mask_to_indices(mask: Array, max_len: int) -> tuple[Array, int]:
    """Convert boolean mask to padded indices array.

    Args:
        mask: boolean mask [N] where True = keep
        max_len: pad indices to this length

    Returns:
        indices: [max_len] indices of True positions, padded with 0s
        n_keep: number of True positions
    """
    indices = jnp.where(mask, size=max_len, fill_value=0)[0]
    n_keep = jnp.sum(mask)
    return indices, n_keep


class IJEPAEncoder(eqx.Module):
    embed: PatchEmbedding
    transformer: Transformer
    pe: PositionalEncoding2D
    dim: int = eqx.field(static=True)
    seq_len: int = eqx.field(static=True)

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
        gradient_checkpointing: bool = False,
        attn_implementation: Optional[str] = None,
        *,
        key: Key[Array, ""],
    ):
        k1, k2 = jax.random.split(key, 2)
        grid_size = img_size // patch_size

        self.embed = PatchEmbedding(num_channels, dim, patch_size, k1)
        self.transformer = Transformer(
            dim=dim,
            num_layers=num_layers,
            num_head=num_head,
            mlp_ratio=mlp_ratio,
            p_drop=p_drop,
            seq_len=seq_len,
            key=k2,
            causal=False,
            gradient_checkpointing=gradient_checkpointing,
            attn_implementation=attn_implementation,
        )
        self.pe = PositionalEncoding2D(dim=dim, seq_len=seq_len, grid_size=grid_size)
        self.dim = dim
        self.seq_len = seq_len

    def __call__(self, key, x, mask=None, train=True, get_intermediates=False):
        """
        Args:
            x: image [H, W, C]
            mask: bool mask [N_patches], True = visible context (flat)
        """
        x = self.embed(x)  # [N_patches, D]
        n_patches = x.shape[0]

        x = self.pe(x)

        if mask is not None:
            mask_flat = mask.reshape(-1)
            indices, n_keep = mask_to_indices(mask_flat, n_patches)

            x_gathered = x[indices]

            # Zero out padding positions
            valid_mask = jnp.arange(n_patches) < n_keep
            x_gathered = jnp.where(valid_mask[:, None], x_gathered, 0.0)

            # Attention mask: only attend to first n_keep positions
            attn_mask = valid_mask[:, None] & valid_mask[None, :]
        else:
            x_gathered = x
            attn_mask = None
            n_keep = n_patches
            indices = jnp.arange(n_patches)

        out = self.transformer(
            x_gathered,
            attn_mask=attn_mask,
            key=key,
            train=train,
            use_pe=False,
            get_intermediates=get_intermediates,
        )

        if get_intermediates:
            out, intermediates = out
            return out, intermediates, indices, n_keep
        return out, indices, n_keep


class IJEPAPredictor(eqx.Module):
    in_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    transformer: Transformer
    pred_token: Array
    pe: PositionalEncoding2D
    norm: eqx.nn.LayerNorm
    dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    grid_size: int = eqx.field(static=True)

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
        gradient_checkpointing: bool = False,
        attn_implementation: Optional[str] = None,
        *,
        key: Key[Array, ""],
    ):
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.in_proj = eqx.nn.Linear(dim, latent_dim, key=k1)
        self.transformer = Transformer(
            dim=latent_dim,
            num_layers=num_layers,
            num_head=num_head,
            mlp_ratio=mlp_ratio,
            p_drop=p_drop,
            seq_len=seq_len,
            key=k2,
            causal=False,
            gradient_checkpointing=gradient_checkpointing,
            attn_implementation=attn_implementation,
        )
        self.pred_token = jax.random.normal(k3, (1, latent_dim)) * 0.02
        self.out_proj = eqx.nn.Linear(latent_dim, dim, key=k4)
        self.pe = PositionalEncoding2D(
            dim=latent_dim, seq_len=seq_len, grid_size=grid_size
        )
        self.norm = eqx.nn.LayerNorm(latent_dim)
        self.dim = dim
        self.latent_dim = latent_dim
        self.grid_size = grid_size

    def __call__(
        self,
        key,
        ctx_emb: Float[Array, "N D"],
        ctx_indices: Array,
        n_ctx: int,
        tgt_indices: Array,
        n_tgt: int,
        seq_len: int,
        train=True,
    ):
        """
        Args:
            ctx_emb: context embeddings from encoder [seq_len, D], first n_ctx valid
            ctx_indices: original patch indices for context [seq_len]
            n_ctx: number of valid context tokens
            tgt_indices: target patch indices [seq_len], first n_tgt valid
            n_tgt: number of target tokens
            seq_len: sequence length for padding
        """
        # Project context to predictor dimension
        ctx_proj = jax.vmap(self.in_proj)(ctx_emb)  # [seq_len, latent_dim]
        dtype = ctx_proj.dtype

        ctx_pos = compute_2d_pe(
            ctx_indices, self.grid_size, self.latent_dim, dtype=dtype
        )
        ctx_proj = ctx_proj + ctx_pos

        # Create pred_tokens with target positional embeddings
        tgt_pos = compute_2d_pe(
            tgt_indices, self.grid_size, self.latent_dim, dtype=dtype
        )
        pred_tokens = self.pred_token + tgt_pos

        total_valid = n_ctx + n_tgt
        combined = jnp.zeros((seq_len, self.latent_dim), dtype=dtype)

        # Place context tokens (first n_ctx positions)
        ctx_mask = jnp.arange(seq_len) < n_ctx
        combined = jnp.where(ctx_mask[:, None], ctx_proj, combined)

        # Place pred_tokens (positions n_ctx to n_ctx + n_tgt)
        tgt_mask = (jnp.arange(seq_len) >= n_ctx) & (jnp.arange(seq_len) < total_valid)
        # Shift pred_tokens indices to get the right ones
        tgt_shifted_idx = jnp.clip(jnp.arange(seq_len) - n_ctx, 0, seq_len - 1)
        pred_tokens_shifted = pred_tokens[tgt_shifted_idx]
        combined = jnp.where(tgt_mask[:, None], pred_tokens_shifted, combined)

        # only attend to first total_valid positions
        valid_mask = jnp.arange(seq_len) < total_valid
        attn_mask = valid_mask[:, None] & valid_mask[None, :]

        out = self.transformer(
            combined, attn_mask=attn_mask, key=key, train=train, use_pe=False
        )
        out = jax.vmap(self.norm)(out)
        out = jax.vmap(self.out_proj)(out)

        # Return predictions at target positions (n_ctx to n_ctx + n_tgt)
        return out, n_ctx, total_valid


class IJEPA(eqx.Module):
    encoder: IJEPAEncoder
    predictor: IJEPAPredictor

    def __init__(self, encoder: IJEPAEncoder, predictor: IJEPAPredictor):
        self.encoder = encoder
        self.predictor = predictor

    def __call__(self, key: Key, x: Array, mask_ctx, mask_pred, train=True):
        """
        Args:
            x: image [H, W, C]
            mask_ctx: context mask [N_patches], True = context (flat)
            mask_pred: target masks [M, N_patches], True = target position (flattened)
        """
        if key is not None:
            k1, k2 = jax.random.split(key, 2)
        else:
            k1 = k2 = None

        # Encoder mask: context minus targets
        mask_tgt = jnp.any(mask_pred, axis=0)  # [N_patches]
        mask_enc = mask_ctx & ~mask_tgt

        # Encode visible context patches
        z_enc, ctx_indices, n_ctx = self.encoder(k1, x, mask_enc, train=train)

        # Get target indices
        tgt_flat = mask_tgt.flatten()
        n_patches = tgt_flat.shape[0]
        tgt_indices, n_tgt = mask_to_indices(tgt_flat, n_patches)

        z_pred, pred_start, pred_end = self.predictor(
            k2,
            ctx_emb=z_enc,
            ctx_indices=ctx_indices,
            n_ctx=n_ctx,
            tgt_indices=tgt_indices,
            n_tgt=n_tgt,
            seq_len=n_patches,
            train=train,
        )

        return z_pred, tgt_indices, n_tgt, pred_start, pred_end
