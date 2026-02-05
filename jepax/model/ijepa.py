import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from jepax.model.transformer import (
    PositionalEncoding2D,
    Transformer,
)
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
        "dim": enc_dim,
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
    model = IJEPA(encoder=encoder, predictor=predictor)

    return model, enc_config["dim"]


class IJEPAEncoder(eqx.Module):
    """I-JEPA Encoder - processes only visible patches via gather."""

    embed: PatchEmbedding
    transformer: Transformer
    pe: PositionalEncoding2D
    dim: int = eqx.field(static=True)
    grid_size: int = eqx.field(static=True)

    def __init__(
        self,
        num_channels: int,
        patch_size: int,
        img_size: int,
        dim: int,
        num_layers: int,
        num_head: int,
        mlp_ratio: float = 4.0,
        p_drop: float = 0.0,
        seq_len: int = 256,
        *,
        key: PRNGKeyArray,
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
        )
        self.pe = PositionalEncoding2D(dim=dim, seq_len=seq_len, grid_size=grid_size)
        self.dim = dim
        self.grid_size = grid_size

    def get_pos_embed(self):
        """Get full positional embeddings [N_patches, D]."""
        return self.pe._get_pe_from_grid(self.pe.grid)

    def __call__(self, key, x, indices=None, train=True, get_intermediates=False):
        """
        Args:
            x: image [H, W, C]
            indices: patch indices to keep [N_keep], or None for all patches
            get_intermediates: if True, return (out, intermediates) for concat probing

        Returns:
            out: encoder output [N_keep, D] or [N_patches, D]
            intermediates: list of layer outputs (if get_intermediates=True)
        """
        x = self.embed(x)  # [N_patches, D]
        pos_emb = self.get_pos_embed()  # [N_patches, D]

        if indices is not None:
            # Gather only visible patches and their positions
            x = x[indices]  # [N_keep, D]
            pos_emb = pos_emb[indices]  # [N_keep, D]

        x = x + pos_emb

        return self.transformer(x, key=key, train=train, use_pe=False, get_intermediates=get_intermediates)


class IJEPAPredictor(eqx.Module):
    """I-JEPA Predictor - concat [context + pos, mask_token + tgt_pos]."""

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
        mlp_ratio: float = 4.0,
        p_drop: float = 0.0,
        seq_len: int = 256,
        *,
        key: PRNGKeyArray,
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
        )
        self.pred_token = jax.random.normal(k3, (latent_dim,)) * 0.02
        self.out_proj = eqx.nn.Linear(latent_dim, dim, key=k4)
        self.pe = PositionalEncoding2D(dim=latent_dim, seq_len=seq_len, grid_size=grid_size)
        self.norm = eqx.nn.LayerNorm(latent_dim)
        self.dim = dim
        self.latent_dim = latent_dim
        self.grid_size = grid_size

    def get_pos_embed(self):
        """Get full positional embeddings [N_patches, latent_dim]."""
        return self.pe._get_pe_from_grid(self.pe.grid)

    def __call__(
        self,
        key,
        ctx_emb: Float[Array, "N_ctx D"],
        ctx_indices: Array,
        tgt_indices: Array,
        train=True,
    ):
        """
        Args:
            ctx_emb: context embeddings from encoder [N_ctx, D]
            ctx_indices: original patch indices for context [N_ctx]
            tgt_indices: target patch indices [N_tgt]

        Returns:
            preds: predictions for target positions [N_tgt, D]
        """
        n_ctx = ctx_emb.shape[0]
        n_tgt = tgt_indices.shape[0]

        # Project context to predictor dimension
        ctx_proj = jax.vmap(self.in_proj)(ctx_emb)  # [N_ctx, latent_dim]

        # Add positional embeddings for context positions
        pos_emb = self.get_pos_embed()  # [N_patches, latent_dim]
        ctx_pos = pos_emb[ctx_indices]  # [N_ctx, latent_dim]
        ctx_proj = ctx_proj + ctx_pos

        # Create pred_tokens with target positional embeddings
        tgt_pos = pos_emb[tgt_indices]  # [N_tgt, latent_dim]
        pred_tokens = self.pred_token + tgt_pos  # [N_tgt, latent_dim] (broadcast)

        # Concatenate: [context, pred_tokens]
        combined = jnp.concatenate([ctx_proj, pred_tokens], axis=0)  # [N_ctx + N_tgt, latent_dim]

        # Run transformer (no extra positional encoding - already added)
        out = self.transformer(combined, key=key, train=train, use_pe=False)

        # Apply layer norm
        out = jax.vmap(self.norm)(out)

        # Project back to encoder dimension
        out = jax.vmap(self.out_proj)(out)

        # Return only predictions for target positions (last N_tgt)
        preds = out[n_ctx:]  # [N_tgt, D]

        return preds


class IJEPA(eqx.Module):
    encoder: IJEPAEncoder
    predictor: IJEPAPredictor

    def __init__(self, encoder: IJEPAEncoder, predictor: IJEPAPredictor):
        self.encoder = encoder
        self.predictor = predictor

    def __call__(self, key, x: Array, ctx_indices: Array, tgt_indices: Array, train=True):
        """
        Args:
            x: image [H, W, C]
            ctx_indices: context patch indices [N_ctx]
            tgt_indices: target patch indices [N_tgt]

        Returns:
            preds: predictions for target positions [N_tgt, D]
        """
        k1, k2 = jax.random.split(key, 2)

        # Encode visible context patches only
        z_ctx = self.encoder(k1, x, indices=ctx_indices, train=train)  # [N_ctx, D]

        # Predict target representations
        preds = self.predictor(
            k2,
            ctx_emb=z_ctx,
            ctx_indices=ctx_indices,
            tgt_indices=tgt_indices,
            train=train,
        )  # [N_tgt, D]

        return preds
