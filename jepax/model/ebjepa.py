"""EB-JEPA: Energy-Based JEPA model.

SimCLR-style SSL: two augmented views -> shared ViT encoder -> projector MLP
"""

from typing import Optional

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Key

from jepax.model.ijepa import get_encoder_config, IJEPAEncoder


ebjepa_configs = {
    "ebjepa-ti": {"encoder": "vit-ti", "proj_hidden": 2048, "proj_output": 2048},
    "ebjepa-s": {"encoder": "vit-s", "proj_hidden": 2048, "proj_output": 2048},
    "ebjepa-b": {"encoder": "vit-b", "proj_hidden": 2048, "proj_output": 2048},
    "ebjepa-l": {"encoder": "vit-l", "proj_hidden": 2048, "proj_output": 2048},
    "ebjepa-h": {"encoder": "vit-h", "proj_hidden": 2048, "proj_output": 2048},
    "ebjepa-test": {"encoder": "test", "proj_hidden": 64, "proj_output": 64},
}


class Projector(eqx.Module):
    """3-layer MLP: Linear->ReLU->Linear->ReLU->Linear (no final bias)."""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    linear3: eqx.nn.Linear

    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, *, key: Key[Array, ""]
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        self.linear1 = eqx.nn.Linear(in_dim, hidden_dim, key=k1)
        self.linear2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=k2)
        self.linear3 = eqx.nn.Linear(hidden_dim, out_dim, use_bias=False, key=k3)

    def __call__(self, x: Float[Array, " D"]) -> Float[Array, " P"]:
        x = jax.nn.relu(self.linear1(x))
        x = jax.nn.relu(self.linear2(x))
        return self.linear3(x)


class EBJEPA(eqx.Module):
    encoder: IJEPAEncoder
    projector: Projector

    def __call__(
        self,
        key: Key[Array, ""],
        x: Float[Array, "C H W"],
        train: bool = True,
    ) -> tuple[Float[Array, " D"], Float[Array, " P"]]:
        out, _, _ = self.encoder(key, x, mask=None, train=train)
        features = jnp.mean(out, axis=0)  # [N_patches, D] -> [D]
        projections = self.projector(features)
        return features, projections


def get_ebjepa_model(
    name: str,
    *,
    key: Key[Array, ""],
    img_size: int = 32,
    patch_size: int = 8,
    seq_len: int = 16,
    num_channels: int = 3,
    p_drop: float = 0.0,
    proj_hidden_dim: int | None = None,
    proj_output_dim: int | None = None,
    gradient_checkpointing: bool = False,
    attn_implementation: Optional[str] = None,
) -> tuple[EBJEPA, int]:
    """Create an EB-JEPA model. Returns (model, embed_dim)."""
    if name not in ebjepa_configs:
        raise ValueError(
            f"Unknown EB-JEPA config: {name}. Choose from {list(ebjepa_configs.keys())}"
        )
    cfg = ebjepa_configs[name]
    enc_name = cfg["encoder"]
    ph = proj_hidden_dim or cfg["proj_hidden"]
    po = proj_output_dim or cfg["proj_output"]

    k1, k2 = jax.random.split(key)

    enc_config = get_encoder_config(
        enc_name,
        num_channels=num_channels,
        patch_size=patch_size,
        img_size=img_size,
        p_drop=p_drop,
        seq_len=seq_len,
    )

    encoder = IJEPAEncoder(
        **enc_config,
        gradient_checkpointing=gradient_checkpointing,
        attn_implementation=attn_implementation,
        key=k1,
    )
    projector = Projector(enc_config["dim"], ph, po, key=k2)
    model = EBJEPA(encoder=encoder, projector=projector)
    return model, enc_config["dim"]
