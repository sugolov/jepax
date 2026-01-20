# eval_ijepa.py
import numpy as np
import jax
from jax import numpy as jnp
import equinox as eqx
import optax
from tqdm import tqdm


class LinearProbe(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, in_dim: int, out_dim: int, *, key):
        self.linear = eqx.nn.Linear(in_dim, out_dim, key=key)

    def __call__(self, x):
        return self.linear(x)


@eqx.filter_jit
def get_representations(encoder, images, key):
    """Get mean-pooled representations for a batch of images."""
    keys = jax.random.split(key, images.shape[0])
    
    def encode_single(k, img):
        out = encoder(k, img, mask=None, train=False)
        return out.mean(axis=0)

    return jax.vmap(encode_single)(keys, images)


@eqx.filter_jit
def linear_probe_step(probe, optimizer, opt_state, reps, labels):
    def loss_fn(probe):
        logits = jax.vmap(probe)(reps)
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    loss, grads = eqx.filter_value_and_grad(loss_fn)(probe)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(probe, eqx.is_array))
    probe = eqx.apply_updates(probe, updates)
    return probe, opt_state, loss


def extract_features(encoder, loader, key, max_samples=None):
    """Extract features from encoder."""
    reps_list, labels_list = [], []
    n_seen = 0
    
    for batch_imgs, batch_labels in loader:
        key, subkey = jax.random.split(key)
        reps = get_representations(encoder, batch_imgs, subkey)
        reps_list.append(np.array(reps))
        labels_list.append(np.array(batch_labels))
        n_seen += len(batch_imgs)
        
        if max_samples and n_seen >= max_samples:
            break

    reps = np.concatenate(reps_list)
    labels = np.concatenate(labels_list)
    
    if max_samples:
        reps = reps[:max_samples]
        labels = labels[:max_samples]
    
    return reps, labels


def evaluate_linear_probe(
    encoder,
    embed_dim,
    train_loader,
    val_loader,
    num_classes,
    key,
    n_epochs=50,
    lr=0.01,
    batch_size=512,
    max_train_samples=None,
    max_val_samples=None,
    verbose=True,
):
    """Train linear probe and evaluate. Returns (top1, top5)."""
    key, k1, k2, k3 = jax.random.split(key, 4)
    
    if verbose:
        print("  Extracting train features...")
    train_reps, train_labels = extract_features(encoder, train_loader, k1, max_train_samples)
    
    if verbose:
        print("  Extracting val features...")
    val_reps, val_labels = extract_features(encoder, val_loader, k2, max_val_samples)
    
    if verbose:
        print(f"  Train: {train_reps.shape}, Val: {val_reps.shape}")

    # Train probe
    probe = LinearProbe(embed_dim, num_classes, key=k3)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(probe, eqx.is_array))

    n_train = len(train_reps)
    
    epoch_iter = tqdm(range(n_epochs), desc="  Probe", leave=False) if verbose else range(n_epochs)
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
    
    preds = jnp.argmax(logits, axis=-1)
    top1_acc = float((preds == val_labels).sum()) / len(val_labels)
    
    top5_preds = jnp.argsort(logits, axis=-1)[:, -5:]
    in_top5 = jnp.any(top5_preds == val_labels[:, None], axis=-1)
    top5_acc = float(in_top5.sum()) / len(val_labels)

    return top1_acc, top5_acc