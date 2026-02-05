import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Key

from jepax.model.transformer import Transformer


vit_classifier_configs = {
    "vit-ti": {"dim": 192, "num_layers": 12, "num_head": 3, "mlp_ratio": 4.0},
    "vit-s": {"dim": 384, "num_layers": 12, "num_head": 6, "mlp_ratio": 4.0},
    "vit-b": {"dim": 768, "num_layers": 12, "num_head": 12, "mlp_ratio": 4.0},
    "vit-l": {"dim": 1024, "num_layers": 24, "num_head": 16, "mlp_ratio": 4.0},
    "vit-h": {"dim": 1280, "num_layers": 32, "num_head": 16, "mlp_ratio": 4.0},
}


def get_vit_config(
    name: str, num_classes: int = 10, num_channels: int = 3, patch_size: int = 16
) -> dict:
    if name not in vit_classifier_configs:
        raise ValueError(
            f"Unknown config: {name}. "
            f"Choose from {list(vit_classifier_configs.keys())}"
        )

    return {
        **vit_classifier_configs[name],
        "num_classes": num_classes,
        "num_channels": num_channels,
        "patch_size": patch_size,
    }


def get_vit_clf_model(
    name: str, num_classes: int = 10, *, key: Key[Array, ""], **kwargs
) -> "ViTclassifier":
    config = get_vit_config(name, num_classes, **kwargs)
    return ViTclassifier(**config, key=key)


class PatchEmbedding(eqx.Module):
    """Linear patch embedding layer."""

    linear: eqx.nn.Linear
    patch_size: int

    def __init__(
        self,
        input_channels: int,
        output_shape: int,
        patch_size: int,
        key: Key[Array, ""],
    ):
        self.patch_size = patch_size
        self.linear = eqx.nn.Linear(
            self.patch_size**2 * input_channels,
            output_shape,
            key=key,
        )

    def __call__(
        self, x: Float[Array, "C H W"]
    ) -> Float[Array, "N D"]:
        # Native JAX patchify (avoids einops sharding issues)
        c, h, w = x.shape
        ps = self.patch_size
        n_h, n_w = h // ps, w // ps
        # [C, H, W] -> [C, n_h, ps, n_w, ps] -> [n_h, n_w, C, ps, ps] -> [N, C*ps*ps]
        x = x.reshape(c, n_h, ps, n_w, ps)
        x = jnp.transpose(x, (1, 3, 0, 2, 4))
        x = x.reshape(n_h * n_w, c * ps * ps)
        x = jax.vmap(self.linear)(x)
        return x


class ViTclassifier(eqx.Module):
    """Vision Transformer for classification."""

    embed: PatchEmbedding
    transformer: Transformer
    clf: eqx.nn.Linear
    cls_token: jax.Array

    def __init__(
        self,
        num_channels: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        num_layers: int,
        num_head: int,
        mlp_ratio: float = 4.0,
        p_drop: float = 0.1,
        seq_len: int = 2048,
        *,
        key: Key[Array, ""],
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
            key=k2,
        )
        self.clf = eqx.nn.Linear(dim, num_classes, key=k3)
        self.cls_token = jax.random.normal(k4, (1, dim))

    def __call__(
        self,
        x: Float[Array, "C H W"],
        key: Key[Array, ""],
        train: bool = True,
    ) -> Float[Array, " K"]:
        x = self.embed(x)
        x = jnp.concatenate([self.cls_token, x], axis=0)
        x = self.transformer(x, key=key, train=train)
        logits = self.clf(x[0])
        return logits
