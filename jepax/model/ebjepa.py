"""EB-JEPA: Energy-Based JEPA model."""

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Key

from jepax.config import EBJEPAModelConfig
from jepax.model.ijepa import get_encoder_config, IJEPAEncoder
from jepax.model.resnet import build_resnet_backbone, ResNetBackbone


class Projector(eqx.Module):
    """3-layer MLP projector with optional normalization."""

    linear1: eqx.nn.Linear
    norm1: eqx.Module | None
    linear2: eqx.nn.Linear
    norm2: eqx.Module | None
    linear3: eqx.nn.Linear
    norm_type: str = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        key: Key[Array, ""],
        norm_type: str = "bn",
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        self.linear1 = eqx.nn.Linear(in_dim, hidden_dim, key=k1)
        self.linear2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=k2)
        self.linear3 = eqx.nn.Linear(hidden_dim, out_dim, key=k3)
        self.norm_type = norm_type

        if norm_type == "bn":
            self.norm1 = eqx.nn.BatchNorm(hidden_dim, axis_name="batch", mode="batch")
            self.norm2 = eqx.nn.BatchNorm(hidden_dim, axis_name="batch", mode="batch")
        elif norm_type == "ln":
            self.norm1 = eqx.nn.LayerNorm(hidden_dim)
            self.norm2 = eqx.nn.LayerNorm(hidden_dim)
        else:
            self.norm1 = None
            self.norm2 = None

    def __call__(
        self,
        x: Float[Array, " D"],
        state: eqx.nn.State | None = None,
    ) -> tuple[Float[Array, " P"], eqx.nn.State | None]:
        x = self.linear1(x)
        if self.norm_type == "bn" and self.norm1 is not None:
            x, state = self.norm1(x, state)
        elif self.norm_type == "ln" and self.norm1 is not None:
            x = self.norm1(x)
        x = jax.nn.relu(x)

        x = self.linear2(x)
        if self.norm_type == "bn" and self.norm2 is not None:
            x, state = self.norm2(x, state)
        elif self.norm_type == "ln" and self.norm2 is not None:
            x = self.norm2(x)
        x = jax.nn.relu(x)

        return self.linear3(x), state


class EBEncoder(eqx.Module):
    encoder: IJEPAEncoder
    norm: eqx.nn.LayerNorm

    def __call__(self, key, x, mask=None, train=True, get_intermediates=False):
        result = self.encoder(
            key, x, mask=mask, train=train, get_intermediates=get_intermediates
        )
        if get_intermediates:
            out, intermediates, indices, n_keep = result
            out = jax.vmap(self.norm)(out)
            return out, intermediates, indices, n_keep
        out, indices, n_keep = result
        out = jax.vmap(self.norm)(out)
        return out, indices, n_keep


class EBJEPA(eqx.Module):
    encoder: eqx.Module
    projector: Projector
    uses_resnet: bool = eqx.field(static=True)

    def __call__(
        self,
        key: Key[Array, ""],
        x: Float[Array, "C H W"],
        state: eqx.nn.State | None = None,
        *,
        train: bool = True,
    ) -> tuple[Float[Array, " D"], Float[Array, " P"], eqx.nn.State | None]:
        if self.uses_resnet:
            out, state = self.encoder(key, x, state)
        else:
            out, _, _ = self.encoder(key, x, mask=None, train=train)
        features = jnp.mean(out, axis=0)  # [N_patches, D] -> [D]
        projections, state = self.projector(features, state)
        return features, projections, state


def get_ebjepa_model(
    model_cfg: EBJEPAModelConfig,
    *,
    key: Key[Array, ""],
    img_size: int = 32,
    gradient_checkpointing: bool = False,
) -> tuple[EBJEPA, int]:
    """Create an EB-JEPA model. Returns (model, embed_dim)."""
    k1, k2 = jax.random.split(key)

    uses_resnet = model_cfg.type == "resnet"
    if uses_resnet:
        encoder, embed_dim = build_resnet_backbone(
            model_cfg.resnet.variant, key=k1, small_input=(img_size <= 64),
        )
    else:
        vit_cfg = model_cfg.vit
        enc_config = get_encoder_config(
            vit_cfg.name,
            num_channels=model_cfg.num_channels,
            patch_size=vit_cfg.patch_size,
            img_size=img_size,
            p_drop=model_cfg.p_drop,
            seq_len=vit_cfg.seq_len,
        )
        ijepa_encoder = IJEPAEncoder(
            **enc_config,
            gradient_checkpointing=gradient_checkpointing,
            attn_implementation=vit_cfg.attn_implementation,
            key=k1,
        )
        norm = eqx.nn.LayerNorm(enc_config["dim"])
        encoder = EBEncoder(encoder=ijepa_encoder, norm=norm)
        embed_dim = enc_config["dim"]

    ph = model_cfg.proj_hidden_dim
    po = model_cfg.proj_output_dim
    projector = Projector(embed_dim, ph, po, key=k2, norm_type=model_cfg.proj_norm)
    model = EBJEPA(encoder=encoder, projector=projector, uses_resnet=uses_resnet)
    return model, embed_dim
