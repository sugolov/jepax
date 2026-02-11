from dataclasses import dataclass, field, asdict
import yaml
import dacite


@dataclass
class DataConfig:
    dataset: str = "imagenet"
    batch_size: int = 2048
    num_workers: int = 16
    prefetch_factor: int = 4


@dataclass
class ModelConfig:
    name: str = "ijepa-b"
    patch_size: int = 14
    seq_len: int = 256
    num_channels: int = 3
    p_drop: float = 0.0


@dataclass
class TrainConfig:
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


@dataclass
class MaskConfig:
    n_pred_masks: int = 4
    pred_scale: tuple[float, float] = (0.15, 0.2)
    pred_aspect: tuple[float, float] = (0.75, 1.5)
    ctx_scale: tuple[float, float] = (0.85, 1.0)
    ctx_aspect: float = 1.0


@dataclass
class EvalConfig:
    mode: str = "last"
    interval: int = 30
    epochs: int = 50
    batch_size: int = 16384
    train_samples: int | None = None
    val_samples: int | None = None
    n_concat: int = 4
    optim: str = "lars"
    lr: float = 0.1
    wd: float = 0.0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return dacite.from_dict(Config, raw, config=dacite.Config(cast=[tuple]))


def to_dict(cfg: Config) -> dict:
    return asdict(cfg)