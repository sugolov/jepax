"""Smoke tests for IJEPA linear-probe evaluation.

Runnable with: pytest tests/test_ijepa_eval.py -v

Minimal: every test just verifies the function runs and returns the right
shape. No convergence claims.

Note: there's a single module-level _ENCODER reused across all tests.
Creating a fresh encoder per test triggers eqx.filter_jit cache-comparison
errors because PositionalEncoding2D stores a numpy grid as a static field.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jepax.model.ijepa import IJEPAEncoder
from jepax.train.eval_ijepa import (
    _get_reps_concat,
    _get_reps_last,
    _train_probe_paper,
    BNLinearProbe,
    evaluate_linear_probe,
    extract_features,
    LinearProbe,
)

IMG_SHAPE = (3, 16, 16)
ENC_DIM = 32

# Shared encoder for all tests that need one.
_ENCODER = IJEPAEncoder(
    num_channels=3,
    patch_size=4,
    img_size=16,
    dim=ENC_DIM,
    num_layers=4,
    num_head=2,
    mlp_ratio=2.0,
    p_drop=0.0,
    seq_len=16,
    key=jax.random.key(0),
)


def _images(batch_size, key=None):
    if key is None:
        key = jax.random.key(1)
    return jax.random.normal(key, (batch_size,) + IMG_SHAPE)


def _dict_loader(images, labels, batch_size):
    return [
        {"image": images[i : i + batch_size], "label": labels[i : i + batch_size]}
        for i in range(0, len(images), batch_size)
    ]


def _tuple_loader(images, labels, batch_size):
    return [
        (images[i : i + batch_size], labels[i : i + batch_size])
        for i in range(0, len(images), batch_size)
    ]


def _synth(n, dim, num_classes, seed=0):
    """Random features + integer labels (not separable, just well-shaped)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    y = rng.integers(0, num_classes, size=(n,)).astype(np.int32)
    return X, y


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def test_linear_probe_runs():
    """LinearProbe maps [in_dim] -> [out_dim]."""
    probe = LinearProbe(in_dim=8, out_dim=3, key=jax.random.key(0))
    out = probe(jnp.zeros(8))
    assert out.shape == (3,)


def test_bn_linear_probe_runs():
    """BNLinearProbe vmapped over a batch returns ([B, out_dim], state)."""
    probe, state = eqx.nn.make_with_state(BNLinearProbe)(
        in_dim=8, out_dim=3, key=jax.random.key(0)
    )
    feats = jnp.zeros((4, 8))
    logits, _ = jax.vmap(
        probe, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(feats, state)
    assert logits.shape == (4, 3)


# ---------------------------------------------------------------------------
# Feature pooling
# ---------------------------------------------------------------------------


def test_get_reps_last_runs():
    """_get_reps_last returns mean-pooled features of shape [B, D]."""
    out = _get_reps_last(_ENCODER, _images(2), jax.random.key(0))
    assert out.shape == (2, ENC_DIM)


def test_get_reps_concat_runs():
    """_get_reps_concat returns (last [B, D], concat [B, n_concat * D])."""
    last, concat = _get_reps_concat(_ENCODER, _images(2), jax.random.key(0), n_concat=2)
    assert last.shape == (2, ENC_DIM)
    assert concat.shape == (2, 2 * ENC_DIM)


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------


def test_extract_features_dict_batch():
    """Loader yielding {'image', 'label'} dicts is supported."""
    imgs = np.array(_images(4))
    labels = np.arange(4, dtype=np.int32)
    last, concat, lab = extract_features(
        _ENCODER, _dict_loader(imgs, labels, 2), jax.random.key(0), n_concat=2
    )
    assert last.shape == (4, ENC_DIM)
    assert concat.shape == (4, 2 * ENC_DIM)
    assert lab.shape == (4,)


def test_extract_features_tuple_batch():
    """Loader yielding (images, labels) tuples is supported."""
    imgs = np.array(_images(4))
    labels = np.arange(4, dtype=np.int32)
    last, _, lab = extract_features(
        _ENCODER, _tuple_loader(imgs, labels, 2), jax.random.key(0), n_concat=1
    )
    assert last.shape == (4, ENC_DIM)
    assert lab.shape == (4,)


def test_extract_features_n_concat_one_returns_none_concat():
    """n_concat=1 skips the concat branch → concat output is None."""
    imgs = np.array(_images(2))
    _, concat, _ = extract_features(
        _ENCODER,
        _dict_loader(imgs, np.zeros(2, dtype=np.int32), 2),
        jax.random.key(0),
        n_concat=1,
    )
    assert concat is None


def test_extract_features_max_samples():
    """max_samples caps returned features and stops iteration early."""
    imgs = np.array(_images(10))
    labels = np.arange(10, dtype=np.int32)
    last, _, lab = extract_features(
        _ENCODER,
        _dict_loader(imgs, labels, 4),
        jax.random.key(0),
        max_samples=5,
        n_concat=1,
    )
    assert last.shape == (5, ENC_DIM)
    assert lab.shape == (5,)


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------


def test_train_probe_runs():
    """AdamW path runs and returns top1, top5 in [0, 1]."""
    np.random.seed(0)
    X_tr, y_tr = _synth(64, dim=8, num_classes=10, seed=0)
    X_va, y_va = _synth(32, dim=8, num_classes=10, seed=1)
    top1, top5 = _train_probe_paper(
        X_tr,
        y_tr,
        X_va,
        y_va,
        input_dim=8,
        num_classes=10,
        key=jax.random.key(0),
        n_epochs=1,
        lr=0.01,
        batch_size=16,
        optim="adamw",
    )
    assert 0.0 <= top1 <= 1.0
    assert 0.0 <= top5 <= 1.0


def test_train_probe_bn_runs():
    """use_bn=True path runs without error."""
    np.random.seed(0)
    X_tr, y_tr = _synth(64, dim=8, num_classes=10, seed=0)
    X_va, y_va = _synth(32, dim=8, num_classes=10, seed=1)
    top1, _ = _train_probe_paper(
        X_tr,
        y_tr,
        X_va,
        y_va,
        input_dim=8,
        num_classes=10,
        key=jax.random.key(0),
        n_epochs=1,
        lr=0.01,
        batch_size=16,
        optim="adamw",
        use_bn=True,
    )
    assert 0.0 <= top1 <= 1.0


def test_train_probe_sgd_runs():
    """optim='sgd' path runs without error."""
    np.random.seed(0)
    X_tr, y_tr = _synth(32, dim=8, num_classes=10, seed=0)
    X_va, y_va = _synth(16, dim=8, num_classes=10, seed=1)
    top1, _ = _train_probe_paper(
        X_tr,
        y_tr,
        X_va,
        y_va,
        input_dim=8,
        num_classes=10,
        key=jax.random.key(0),
        n_epochs=1,
        lr=0.01,
        batch_size=16,
        optim="sgd",
    )
    assert 0.0 <= top1 <= 1.0


def test_train_probe_lars_runs():
    """optim='lars' path runs without error."""
    np.random.seed(0)
    X_tr, y_tr = _synth(32, dim=8, num_classes=10, seed=0)
    X_va, y_va = _synth(16, dim=8, num_classes=10, seed=1)
    top1, _ = _train_probe_paper(
        X_tr,
        y_tr,
        X_va,
        y_va,
        input_dim=8,
        num_classes=10,
        key=jax.random.key(0),
        n_epochs=1,
        lr=0.1,
        batch_size=16,
        optim="lars",
    )
    assert 0.0 <= top1 <= 1.0


# ---------------------------------------------------------------------------
# evaluate_linear_probe
# ---------------------------------------------------------------------------


def test_evaluate_linear_probe_runs():
    """Default modes produces last_* and concat_* keys."""
    rng = np.random.default_rng(0)
    tr = rng.normal(size=(16,) + IMG_SHAPE).astype(np.float32)
    tr_y = rng.integers(0, 10, size=(16,)).astype(np.int32)
    va = rng.normal(size=(8,) + IMG_SHAPE).astype(np.float32)
    va_y = rng.integers(0, 10, size=(8,)).astype(np.int32)

    np.random.seed(0)
    results = evaluate_linear_probe(
        encoder=_ENCODER,
        embed_dim=ENC_DIM,
        train_loader=_dict_loader(tr, tr_y, 8),
        val_loader=_dict_loader(va, va_y, 8),
        num_classes=10,
        key=jax.random.key(0),
        n_concat=2,
        n_epochs=1,
        lr=0.01,
        batch_size=8,
        optim="adamw",
        verbose=False,
    )
    assert "last_top1" in results
    assert "concat_top1" in results


def test_evaluate_linear_probe_modes_filter_runs():
    """modes=['last'] filters to only that config's keys."""
    rng = np.random.default_rng(0)
    tr = rng.normal(size=(16,) + IMG_SHAPE).astype(np.float32)
    tr_y = rng.integers(0, 10, size=(16,)).astype(np.int32)
    va = rng.normal(size=(8,) + IMG_SHAPE).astype(np.float32)
    va_y = rng.integers(0, 10, size=(8,)).astype(np.int32)

    np.random.seed(0)
    results = evaluate_linear_probe(
        encoder=_ENCODER,
        embed_dim=ENC_DIM,
        train_loader=_dict_loader(tr, tr_y, 8),
        val_loader=_dict_loader(va, va_y, 8),
        num_classes=10,
        key=jax.random.key(0),
        n_concat=2,
        n_epochs=1,
        lr=0.01,
        batch_size=8,
        optim="adamw",
        verbose=False,
        modes=["last"],
    )
    assert "last_top1" in results
    assert not any(k.startswith("concat") for k in results)
