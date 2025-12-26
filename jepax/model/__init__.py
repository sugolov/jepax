from jepax.model.vit import ViTclassifier
from jaxtyping import PRNGKeyArray


vit_classifier_configs = {
    "vit-ti": {"dim": 192, "num_layers": 12, "num_head": 3, "mlp_ratio": 4.0},
    "vit-s": {"dim": 384, "num_layers": 12, "num_head": 6, "mlp_ratio": 4.0},
    "vit-b": {"dim": 768, "num_layers": 12, "num_head": 12, "mlp_ratio": 4.0},
    "vit-l": {"dim": 1024, "num_layers": 24, "num_head": 16, "mlp_ratio": 4.0},
    "vit-h": {"dim": 1280, "num_layers": 32, "num_head": 16, "mlp_ratio": 4.0},
}


def get_vit_config(
    name: str, num_classes: int = 10, num_channels: int = 3, patch_size: int = 16
):
    if name not in vit_classifier_configs:
        raise ValueError(
            f"Unknown config: {name}. Choose from {list(vit_classifier_configs.keys())}"
        )

    return {
        **vit_classifier_configs[name],
        "num_classes": num_classes,
        "num_channels": num_channels,
        "patch_size": patch_size,
    }


def get_vit_clf_model(name: str, num_classes: int = 10, *, key: PRNGKeyArray, **kwargs):
    config = get_vit_config(name, num_classes, **kwargs)
    return ViTclassifier(**config, key=key)
