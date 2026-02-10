"""Linear probe evaluation for I-JEPA.

Paper protocol: test (last-layer / concat-4) x (with / without BN), report best.
Uses LARS with step-wise LR decay (÷10 every 15 epochs) following MAE.
"""
import gc
import warnings

import equinox as eqx
import jax
import numpy as np
import optax
from jax import numpy as jnp

warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------------------------------------------------------
# Feature extraction (shared)
# -----------------------------------------------------------------------------


@eqx.filter_jit
def _get_reps_last(encoder, images, key):
    """Get mean-pooled last-layer representations."""
    keys = jax.random.split(key, images.shape[0])

    def encode(k, img):
        out, _, _ = encoder(k, img, mask=None, train=False)
        return out.mean(axis=0)

    return jax.vmap(encode)(keys, images)


@eqx.filter_jit
def _get_reps_concat(encoder, images, key, n_concat=4):
    """Get concatenated avg-pooled representations from last n layers."""
    keys = jax.random.split(key, images.shape[0])

    def encode(k, img):
        out, intermediates, _, _ = encoder(
            k, img, mask=None, train=False, get_intermediates=True
        )
        last = out.mean(axis=0)
        inter = jnp.stack(intermediates[-n_concat:])  # (n_concat, T, D)
        inter = inter.mean(axis=1).flatten()  # (n_concat * D,)
        return last, inter

    return jax.vmap(encode)(keys, images)


def extract_features(encoder, loader, key, max_samples=None, n_concat=4):
    """Extract features from encoder."""
    last_list, concat_list, labels_list = [], [], []
    n_seen = 0

    for batch in loader:
        batch_imgs = batch["image"]
        batch_labels = batch["label"]
        key, subkey = jax.random.split(key)

        if n_concat > 1:
            last, concat = _get_reps_concat(encoder, batch_imgs, subkey, n_concat)
            concat_list.append(np.array(concat))
        else:
            last = _get_reps_last(encoder, batch_imgs, subkey)

        last_list.append(np.array(last))
        labels_list.append(np.array(batch_labels))

        n_seen += len(batch_imgs)
        if max_samples and n_seen >= max_samples:
            break

    last = np.concatenate(last_list)[:max_samples]
    labels = np.concatenate(labels_list)[:max_samples]
    concat = np.concatenate(concat_list)[:max_samples] if concat_list else None

    gc.collect()
    return last, concat, labels


# -----------------------------------------------------------------------------
# Paper mode: LARS/Adam + optional BatchNorm
# -----------------------------------------------------------------------------


class LinearProbe(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, in_dim: int, out_dim: int, *, key):
        self.linear = eqx.nn.Linear(in_dim, out_dim, key=key)

    def __call__(self, x):
        return self.linear(x)


class BNLinearProbe(eqx.Module):
    bn: eqx.nn.BatchNorm
    linear: eqx.nn.Linear

    def __init__(self, in_dim: int, out_dim: int, *, key):
        self.bn = eqx.nn.BatchNorm(in_dim, axis_name="batch")
        self.linear = eqx.nn.Linear(in_dim, out_dim, key=key)

    def __call__(self, x, state):
        x, state = self.bn(x, state)
        return self.linear(x), state


def _train_probe_paper(
    train_feats,
    train_labels,
    val_feats,
    val_labels,
    input_dim,
    num_classes,
    key,
    n_epochs,
    lr,
    batch_size,
    optim="lars",
    weight_decay=0.0,
    use_bn=False,
):
    """Train probe with LARS/Adam optimizer (paper style)."""
    key, init_key = jax.random.split(key)

    if use_bn:
        probe, state = eqx.nn.make_with_state(BNLinearProbe)(
            input_dim, num_classes, key=init_key
        )
    else:
        probe = LinearProbe(input_dim, num_classes, key=init_key)
        state = None

    if optim.lower() == "lars":
        optimizer = optax.lars(lr, weight_decay=weight_decay)
    elif optim.lower() in ["adam", "adamw"]:
        optimizer = optax.adamw(lr, weight_decay=weight_decay)
    else:
        optimizer = optax.sgd(lr)

    opt_state = optimizer.init(eqx.filter(probe, eqx.is_array))

    @eqx.filter_jit
    def step_bn(probe, state, opt_state, feats, labels):
        def loss_fn(probe, state):
            logits, state = jax.vmap(
                probe, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
            )(feats, state)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            return loss, state

        (loss, state), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(probe, state)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(probe, eqx.is_array))
        probe = eqx.apply_updates(probe, updates)
        return probe, state, opt_state, loss

    @eqx.filter_jit
    def step_linear(probe, opt_state, feats, labels):
        def loss_fn(probe):
            logits = jax.vmap(probe)(feats)
            return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

        loss, grads = eqx.filter_value_and_grad(loss_fn)(probe)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(probe, eqx.is_array))
        probe = eqx.apply_updates(probe, updates)
        return probe, opt_state, loss

    n_train = len(train_feats)
    for _ in range(n_epochs):
        perm = np.random.permutation(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            if len(idx) < 2:
                continue  # BatchNorm needs at least 2 samples
            feats_batch = jnp.array(train_feats[idx])
            labels_batch = jnp.array(train_labels[idx])
            if use_bn:
                probe, state, opt_state, _ = step_bn(probe, state, opt_state, feats_batch, labels_batch)
            else:
                probe, opt_state, _ = step_linear(probe, opt_state, feats_batch, labels_batch)

    # Evaluate
    val_feats_jnp = jnp.array(val_feats)
    if use_bn:
        inference_probe = eqx.nn.inference_mode(probe)
        logits, _ = jax.vmap(
            inference_probe, in_axes=(0, None), out_axes=(0, None)
        )(val_feats_jnp, state)
    else:
        logits = jax.vmap(probe)(val_feats_jnp)

    top1 = float((logits.argmax(-1) == val_labels).mean())
    top5_idx = jnp.argsort(logits, axis=-1)[:, -5:]
    top5 = float(jnp.any(top5_idx == val_labels[:, None], axis=-1).mean())

    return top1, top5


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------


def evaluate_linear_probe(
    encoder,
    embed_dim,
    train_loader,
    val_loader,
    num_classes,
    key,
    n_concat=4,
    n_epochs=50,
    lr=0.1,
    batch_size=16384,
    optim="lars",
    weight_decay=0.0,
    max_train_samples=None,
    max_val_samples=None,
    verbose=True,
    **kwargs,
):
    """Evaluate linear probe across all configurations (last, last_bn, concat, concat_bn)."""
    key, k1, k2 = jax.random.split(key, 3)

    if verbose:
        print("Probe: extracting train features")
    train_last, train_concat, train_labels = extract_features(
        encoder, train_loader, k1, max_train_samples, n_concat
    )

    if verbose:
        print("Probe: extracting val features")
    val_last, val_concat, val_labels = extract_features(
        encoder, val_loader, k2, max_val_samples, n_concat
    )

    if verbose:
        shapes = f"train {train_last.shape}, val {val_last.shape}"
        if train_concat is not None:
            shapes += f" (concat: {train_concat.shape})"
        print(f"Probe: {shapes}")

    results = {}

    configs = [
        ("last", train_last, val_last, embed_dim, False),
        ("last_bn", train_last, val_last, embed_dim, True),
    ]
    if train_concat is not None:
        configs.extend([
            ("concat", train_concat, val_concat, embed_dim * n_concat, False),
            ("concat_bn", train_concat, val_concat, embed_dim * n_concat, True),
        ])

    for name, tr_f, val_f, dim, use_bn in configs:
        if verbose:
            print(f"Probe: training {name}")
        key, subkey = jax.random.split(key)
        t1, t5 = _train_probe_paper(
            tr_f, train_labels, val_f, val_labels,
            dim, num_classes, subkey,
            n_epochs, lr, batch_size, optim, weight_decay, use_bn,
        )
        results[f"{name}_top1"] = t1
        results[f"{name}_top5"] = t5
        if verbose:
            print(f"  {name}: top1={t1*100:.2f}%, top5={t5*100:.2f}%")

    return results
