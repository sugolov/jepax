import yaml
from types import SimpleNamespace


def _to_namespace(d):
    """Recursively convert dict to SimpleNamespace for dot access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_namespace(x) for x in d]
    return d


def load_config(path: str) -> SimpleNamespace:
    """Load YAML config with dot-access support."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw)


def to_dict(cfg: SimpleNamespace) -> dict:
    """Convert SimpleNamespace back to dict."""
    if isinstance(cfg, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(cfg).items()}
    if isinstance(cfg, list):
        return [to_dict(x) for x in cfg]
    return cfg
