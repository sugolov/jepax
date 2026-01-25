import gc

import numpy as np
import jax
from jax import numpy as jnp
import equinox as eqx
import optax
from tqdm import tqdm

from functools import partial


class LinearProbe(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, in_dim: int, out_dim: int, *, key):
        self.linear = eqx.nn.Linear(in_dim, out_dim, key=key)

    def __call__(self, x):
        return self.linear(x)


@eqx.filter_jit
def get_representations(encoder, images, key, *, n_concat=1):
    """Get mean-pooled representations for a batch of images."""
    keys = jax.random.split(key, images.shape[0])

    def encode_single(k, img):
        out = encoder(k, img, mask=None, train=False)
        z = out.mean(axis=0)
        return z, z

    def encode_multiple(k, img):
        z, out = encoder(k, img, mask=None, train=False, get_intermediates=True)
        z = z.mean(axis=0)
        out = jnp.stack(out[-n_concat:]) # (n_last, T, D)
        out = out.mean(axis=1) # (n_last, D)
        out = out.flatten() # (n_last * D,)
        return z, out
    
    assert n_concat > 0, "last layer probing idx must be >= 0"
    encode = encode_single if n_concat == 1 else encode_multiple

    return jax.vmap(encode)(keys, images)

def extract_features(encoder, loader, key, max_samples=None, n_concat=4):
    """Extract both last-layer and concat features in single pass."""
    last_list, concat_list, labels_list = [], [], []
    n_seen = 0
    
    for batch in loader:
        batch_imgs = batch["image"]
        batch_labels = batch["label"]
        
        key, subkey = jax.random.split(key)
        last_reps, concat_reps = get_representations(
            encoder, 
            batch_imgs, 
            subkey, 
            n_concat=n_concat
        )
        
        last_list.append(np.array(last_reps))
        concat_list.append(np.array(concat_reps))
        labels_list.append(np.array(batch_labels))
        
        n_seen += len(batch_imgs)
        if max_samples and n_seen >= max_samples:
            break

    last = np.concatenate(last_list)[:max_samples]
    concat = np.concatenate(concat_list)[:max_samples]
    labels = np.concatenate(labels_list)[:max_samples]

    gc.collect()
    
    return last, concat, labels

@eqx.filter_jit
def linear_probe_step(probe, optimizer, opt_state, reps, labels):
    def loss_fn(probe):
        logits = jax.vmap(probe)(reps)
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    loss, grads = eqx.filter_value_and_grad(loss_fn)(probe)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(probe, eqx.is_array))
    probe = eqx.apply_updates(probe, updates)
    return probe, opt_state, loss


def train_and_eval_probe(
    train_reps, train_labels, val_reps, val_labels,
    input_dim, num_classes, key, n_epochs, lr, batch_size, optim="adam", 
    weight_decay=5e-4, verbose=False
):
    """Train a single linear probe and return (top1, top5)."""
    probe = LinearProbe(input_dim, num_classes, key=key)

    if optim.lower() in ["adam", "adamw"]:
        optimizer = optax.adamw(lr, weight_decay=weight_decay)
    elif optim.lower() == "lars":
        optimizer = optax.lars(lr, weight_decay=weight_decay)
    else:
        optimizer = optax.sgd(lr)

    opt_state = optimizer.init(eqx.filter(probe, eqx.is_array))
    
    n_train = len(train_reps)
    epoch_iter = tqdm(range(n_epochs), desc="    Probe", leave=False) if verbose else range(n_epochs)
    
    for _ in epoch_iter:
        perm = np.random.permutation(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            probe, opt_state, _ = linear_probe_step(
                probe, optimizer, opt_state,
                jnp.array(train_reps[idx]),
                jnp.array(train_labels[idx])
            )
    
    # Evaluate
    logits = jax.vmap(probe)(jnp.array(val_reps))
    
    top1 = float((logits.argmax(-1) == val_labels).mean())
    top5_preds = jnp.argsort(logits, axis=-1)[:, -5:]
    top5 = float(jnp.any(top5_preds == val_labels[:, None], axis=-1).mean())
    
    return top1, top5

def evaluate_linear_probe(
    encoder,
    embed_dim,
    train_loader,
    val_loader,
    num_classes,
    key,
    n_concat=4,
    n_epochs=50,
    lr=0.01,
    batch_size=512,
    optim = "adam",
    weight_decay = 5e-4,
    max_train_samples=None,
    max_val_samples=None,
    verbose=True,
):
    """Train linear probes and evaluate. Returns dict with last/concat/best results."""
    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    
    if verbose:
        print("Probe: extracting train features")
    train_last, train_concat, train_labels = extract_features(
        encoder, train_loader, k1, max_train_samples, n_concat=n_concat
    )
    
    if verbose:
        print("Probe: extracting val features")
    val_last, val_concat, val_labels = extract_features(
        encoder, val_loader, k2, max_val_samples, n_concat=n_concat
    )
    
    if verbose:
        print(f"Probe: train {train_last.shape} / {train_concat.shape}, val {val_last.shape} / {val_concat.shape}")
    
    if verbose:
        print("Probe: training last-layer probe")
    top1_last, top5_last = train_and_eval_probe(
        train_last, train_labels, val_last, val_labels,
        embed_dim, num_classes, k3, n_epochs, lr, batch_size, 
        optim=optim, weight_decay=weight_decay, verbose=verbose
    )
    
    result = {
        "last": (top1_last, top5_last),
        "best": (top1_last, top5_last),
    }
    
    if n_concat > 1:
        if verbose:
            print(f"Probe: training concat-{n_concat} probe")
        top1_concat, top5_concat = train_and_eval_probe(
            train_concat, train_labels, val_concat, val_labels,
            embed_dim * n_concat, num_classes, k4, n_epochs, lr, batch_size,
            optim=optim, weight_decay=weight_decay, verbose=verbose
        )
        result["concat"] = (top1_concat, top5_concat)
        result["best"] = (max(top1_last, top1_concat), max(top5_last, top5_concat))
    
    return result