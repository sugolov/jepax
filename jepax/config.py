from dataclasses import dataclass, field

import dacite
import yaml


@dataclass
class DataConfig:
    dataset: str = "imagenet"
    batch_size: int = 2048
    num_workers: int = 16
    prefetch_factor: int = 4


@dataclass
class IJEPAModelConfig:
    name: str = "ijepa-b"
    patch_size: int = 14
    seq_len: int = 256
    num_channels: int = 3
    p_drop: float = 0.0
    attn_implementation: str | None = None


@dataclass
class IJEPATrainConfig:
    epochs: int = 300
    lr: float = 1e-3
    start_lr: float = 2e-4
    final_lr: float = 1e-6
    warmup_epochs: int = 15
    wd: float = 0.04
    final_wd: float = 0.4
    ema_start: float = 0.996
    ema_end: float = 1.0
    seed: int = 42
    normalize_targets: bool = True
    gradient_checkpointing: bool = False


@dataclass
class MaskConfig:
    n_pred_masks: int = 4
    pred_scale: tuple[float, float] = (0.15, 0.2)
    pred_aspect: tuple[float, float] = (0.75, 1.5)
    ctx_scale: tuple[float, float] = (0.85, 1.0)
    ctx_aspect: float = 1.0


@dataclass
class EvalConfig:
    bn_mode: str = "ema"
    interval: int = 30
    epochs: int = 50
    batch_size: int = 16384
    train_samples: int | None = None
    val_samples: int | None = None
    n_concat: int = 4
    optim: str = "lars"
    lr: float = 0.1
    wd: float = 0.0
    modes: list[str] | None = None


@dataclass
class IJEPAConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: IJEPAModelConfig = field(default_factory=IJEPAModelConfig)
    train: IJEPATrainConfig = field(default_factory=IJEPATrainConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_ijepa_config(path: str) -> IJEPAConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return dacite.from_dict(IJEPAConfig, raw, config=dacite.Config(cast=[tuple]))


@dataclass
class ViTBackboneConfig:
    name: str = "vit-s"
    patch_size: int = 8
    seq_len: int = 16
    attn_implementation: str | None = None


@dataclass
class ResNetBackboneConfig:
    variant: str = "resnet18"


@dataclass
class EBJEPAModelConfig:
    type: str = "vit"
    num_channels: int = 3
    p_drop: float = 0.0
    proj_hidden_dim: int = 2048
    proj_output_dim: int = 2048
    proj_norm: str = "bn"  # "bn", "ln", or "none"
    vit: ViTBackboneConfig = field(default_factory=ViTBackboneConfig)
    resnet: ResNetBackboneConfig = field(default_factory=ResNetBackboneConfig)


@dataclass
class EBJEPATrainConfig:
    epochs: int = 300
    lr: float = 0.3
    start_lr: float = 3e-5
    final_lr: float = 0.0
    warmup_epochs: int = 10
    wd: float = 1e-4
    seed: int = 42
    optimizer: str = "lars"
    gradient_checkpointing: bool = False


@dataclass
class LossConfig:
    type: str = "vicreg"
    std_coeff: float = 1.0
    cov_coeff: float = 80.0
    bcs_lmbd: float = 10.0
    bcs_num_slices: int = 256


@dataclass
class AugConfig:
    random_crop_scale: tuple[float, float] = (0.2, 1.0)
    color_jitter_prob: float = 0.8
    grayscale_prob: float = 0.2
    hflip_prob: float = 0.5
    solarize_prob: float = 0.0


@dataclass
class EBJEPAConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: EBJEPAModelConfig = field(default_factory=EBJEPAModelConfig)
    train: EBJEPATrainConfig = field(default_factory=EBJEPATrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_ebjepa_config(path: str) -> EBJEPAConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return dacite.from_dict(EBJEPAConfig, raw, config=dacite.Config(cast=[tuple]))
