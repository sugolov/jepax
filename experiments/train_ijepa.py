import argparse
import json
import os
from pathlib import Path
import time
import equinox as eqx
import jax
from jax.sharding import PartitionSpec as P, NamedSharding
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from jepax.config import load_config, to_dict
from jepax.data.datasets import build_dataset
from jepax.model.ijepa import IJEPA, ema_update
from jepax.masks import MaskCollator

"""
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
"""


def get_patch_size(dataset: str, img_size: int) -> int:
    # If resized to 224, use ImageNet-style patch size
    if img_size == 224:
        return 14  # 224x224 -> 16x16 = 256 patches
    return {
        "cifar10": 4,  # 32x32 -> 8x8 = 64 patches
        "cifar100": 4,  # 32x32 -> 8x8 = 64 patches
        "celeba": 8,  # 64x64 -> 8x8 = 64 patches
        "imagenet": 14,  # 224x224 -> 16x16 = 256 patches
    }[dataset.lower()]


def parse_args():
    p = argparse.ArgumentParser(
        description="I-JEPA training. Use --config for YAML config, CLI args override."
    )
    p.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    # Allow CLI overrides for common params
    p.add_argument("--epochs", type=int, help="Override train.epochs")
    p.add_argument("--batch_size", type=int, help="Override data.batch_size")
    p.add_argument("--lr", type=float, help="Override train.lr")
    p.add_argument("--shard", action="store_true", help="Enable sharding")
    p.add_argument("--no_shard", action="store_true", help="Disable sharding")
    return p.parse_args()


def apply_cli_overrides(cfg, args):
    """Apply CLI overrides to config."""
    if args.resume:
        cfg.resume = args.resume
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.data.batch_size = args.batch_size
    if args.lr:
        cfg.train.lr = args.lr
    if args.shard:
        cfg.shard = True
    if args.no_shard:
        cfg.shard = False
    return cfg


def smooth_l1_loss(pred, target):
    diff = pred - target
    abs_diff = jnp.abs(diff)
    return jnp.where(abs_diff < 1.0, 0.5 * diff**2, abs_diff - 0.5).mean()


def layer_norm(x, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def forward_single_block(model, ctx_emb, ctx_idx, target_out, tgt_block_idx):
    # Get target embeddings for this block
    target_emb = target_out[tgt_block_idx]  # [L_pred, D]
    target_emb = layer_norm(target_emb)
    target_emb = jax.lax.stop_gradient(target_emb)

    # Predictor: predict this block's embeddings
    pred_emb = model.predictor(ctx_emb, ctx_idx, tgt_block_idx)  # [L_pred, D]

    return smooth_l1_loss(pred_emb, target_emb)


def forward_single(model, target_encoder, img, ctx_idx, tgt_blocks):
    # Target encoder: all patches (no mask)
    target_out = target_encoder(img, mask_indices=None)  # [N, D]

    # Context encoder: only context patches
    ctx_emb = model.encoder(img, mask_indices=ctx_idx)  # [L_enc, D]

    # Predict each target block separately and average loss
    def block_loss(tgt_block_idx):
        return forward_single_block(model, ctx_emb, ctx_idx, target_out, tgt_block_idx)

    # vmap over the npred target blocks
    losses = jax.vmap(block_loss)(tgt_blocks)  # [npred]
    return jnp.mean(losses)


@eqx.filter_jit(donate="warn-except-first")
def train_step(models, optimizer, opt_state, batch, enc_masks, pred_masks):
    """
    Train step with per-block target prediction.

    Args:
        models: tuple of (model, target_encoder)
        batch: images [B, H, W, C]
        enc_masks: context masks [B, n_enc, L_enc] (usually n_enc=1)
        pred_masks: target masks [B, n_pred, L_pred] (usually n_pred=4)
    """
    model, target_encoder = models

    def loss_fn(model):
        # just hardcode to use first encoder mask (n_enc=1 is standard)
        ctx_masks = enc_masks[:, 0, :]  # [B, L_enc]
        # pred_masks is [B, n_pred, L_pred]
        losses = eqx.filter_vmap(forward_single, in_axes=(None, None, 0, 0, 0))(
            model, target_encoder, batch, ctx_masks, pred_masks
        )
        return jnp.mean(losses)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)

    return (model, target_encoder), opt_state, loss


class LinearProbe(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, in_dim: int, out_dim: int, *, key):
        self.linear = eqx.nn.Linear(in_dim, out_dim, key=key)

    def __call__(self, x):
        return self.linear(x)

    @eqx.filter_jit
    def veval(self, x):
        return jax.vmap(self.linear)(x)


@eqx.filter_jit
def get_representations(encoder, images):
    """Get mean-pooled representations for a batch of images."""

    def encode_single(img):
        out = encoder(img, mask_indices=None)  # [N, D]
        return out.mean(axis=0)  # [D] mean pool

    return jax.vmap(encode_single)(images)


@eqx.filter_jit
def linear_probe_step(probe, optimizer, opt_state, reps, labels, is_multilabel=False):
    def loss_fn(probe):
        logits = jax.vmap(probe)(reps)  # [B, num_classes]
        if is_multilabel:
            # Binary cross-entropy for multi-label (CelebA attributes)
            loss = optax.sigmoid_binary_cross_entropy(
                logits, labels.astype(jnp.float32)
            )
        else:
            # Softmax cross-entropy for single-label (CIFAR, ImageNet)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return loss.mean()

    loss, grads = eqx.filter_value_and_grad(loss_fn)(probe)
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(probe, eqx.is_array)
    )
    probe = eqx.apply_updates(probe, updates)
    return probe, opt_state, loss


def evaluate_linear_probe(
    encoder,
    train_loader,
    val_loader,
    key,
    num_classes,
    is_multilabel=False,
    n_epochs=50,
    max_train_samples=None,
    max_val_samples=None,
    verbose=True,
    data_sharding=None,
    replicate_sharding=None,
):
    """Train linear probe and evaluate. Returns (top1, top5)."""
    embed_dim = encoder.embed_dim
    use_sharding = data_sharding is not None

    if verbose:
        print("  Probe: extracting features...", end=" ", flush=True)
    t0 = time.time()

    def extract_features(loader, max_samples):
        reps_list, labels_list = [], []
        n_seen = 0
        for batch_imgs, batch_labels in loader:
            if use_sharding:
                batch_imgs = jax.device_put(batch_imgs, data_sharding)
            else:
                batch_imgs = jnp.array(batch_imgs)
            reps = get_representations(encoder, batch_imgs)
            reps_np = np.array(reps)
            labels_np = np.array(batch_labels)

            if max_samples is not None and (n_seen + len(reps_np)) > max_samples:
                take = max_samples - n_seen
                reps_np = reps_np[:take]
                labels_np = labels_np[:take]

            reps_list.append(reps_np)
            labels_list.append(labels_np)
            n_seen += len(reps_np)

            if max_samples is not None and n_seen >= max_samples:
                break

        return np.concatenate(reps_list), np.concatenate(labels_list)

    train_reps, train_labels = extract_features(train_loader, max_train_samples)
    val_reps, val_labels = extract_features(val_loader, max_val_samples)

    if verbose:
        print(f"({time.time() - t0:.1f}s)")

    probe = LinearProbe(embed_dim, num_classes, key=key)
    optimizer = optax.lars(0.05)
    opt_state = optimizer.init(eqx.filter(probe, eqx.is_array))

    if use_sharding:
        n_devices = len(jax.devices())
        batch_size = 512 * n_devices
        probe = eqx.filter_shard(probe, replicate_sharding)
        opt_state = eqx.filter_shard(opt_state, replicate_sharding)
    else:
        batch_size = 512

    n_train = len(train_reps)

    t0 = time.time()
    epoch_iter = range(n_epochs)
    if verbose:
        epoch_iter = tqdm(epoch_iter, desc="  Probe train", leave=False)

    for epoch in epoch_iter:
        perm = np.random.permutation(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            batch_reps = train_reps[idx]
            batch_labels_np = train_labels[idx]

            if use_sharding:
                batch_reps = jax.device_put(batch_reps, data_sharding)
                batch_labels_arr = jax.device_put(batch_labels_np, data_sharding)
            else:
                batch_reps = jnp.array(batch_reps)
                batch_labels_arr = jnp.array(batch_labels_np)

            probe, opt_state, _ = linear_probe_step(
                probe, optimizer, opt_state, batch_reps, batch_labels_arr, is_multilabel
            )

    if verbose:
        print(f"  Probe: trained {n_epochs} epochs ({time.time() - t0:.1f}s)")

    if verbose:
        print("  Probe: evaluating...", end=" ", flush=True)
    t0 = time.time()

    if use_sharding:
        val_reps_arr = jax.device_put(val_reps, data_sharding)
    else:
        val_reps_arr = jnp.array(val_reps)

    logits = probe.veval(val_reps_arr)

    if is_multilabel:
        preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32)
        correct = (preds == val_labels).sum()
        total_top1 = correct
        total_top5 = correct
        total_samples = val_labels.size
    else:
        preds = jnp.argmax(logits, axis=-1)
        total_top1 = (preds == val_labels).sum()
        top5_preds = jnp.argsort(logits, axis=-1)[:, -5:]
        in_top5 = jnp.any(top5_preds == val_labels[:, None], axis=-1)
        total_top5 = in_top5.sum()
        total_samples = val_labels.shape[0]

    if verbose:
        print(f"({time.time() - t0:.1f}s)")

    top1_acc = float(total_top1) / total_samples
    top5_acc = float(total_top5) / total_samples
    return top1_acc, top5_acc


def save_checkpoint(model, target_encoder, opt_state, epoch, cfg, path):
    eqx.tree_serialise_leaves(path + "_model.eqx", model)
    eqx.tree_serialise_leaves(path + "_target.eqx", target_encoder)
    eqx.tree_serialise_leaves(path + "_opt.eqx", opt_state)
    with open(path + "_meta.json", "w") as f:
        json.dump({"epoch": epoch, "config": to_dict(cfg)}, f)


def load_checkpoint(path, model, target_encoder, opt_state):
    model = eqx.tree_deserialise_leaves(path + "_model.eqx", model)
    target_encoder = eqx.tree_deserialise_leaves(path + "_target.eqx", target_encoder)
    opt_state = eqx.tree_deserialise_leaves(path + "_opt.eqx", opt_state)
    with open(path + "_meta.json", "r") as f:
        meta = json.load(f)
    return model, target_encoder, opt_state, meta["epoch"]


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    if not hasattr(cfg, "resume"):
        cfg.resume = None

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    print(f"JAX devices: {jax.devices()}")
    print(f"Config: {args.config}")
    key = jax.random.key(cfg.train.seed)

    resize_to = 224 if cfg.data.resize else None
    crop_scale = tuple(cfg.augment.crop_scale)

    print(f"Loading {cfg.data.dataset}...")
    train_loader, num_classes, n_train, img_size = build_dataset(
        cfg.data.dataset,
        cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        is_train=True,
        num_workers=cfg.data.num_workers,
        img_size=resize_to,
        crop_scale=crop_scale,
        horizontal_flip=cfg.augment.horizontal_flip,
        normalize=cfg.augment.normalize,
        preload=getattr(cfg.data, "preload", False),
    )
    val_loader, _, _, _ = build_dataset(
        cfg.data.dataset,
        cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        is_train=False,
        num_workers=cfg.data.num_workers,
        img_size=resize_to,
        normalize=cfg.augment.normalize,
        preload=getattr(cfg.data, "preload", False),
    )
    n_batches = n_train // cfg.data.batch_size
    patch_size = get_patch_size(cfg.data.dataset, img_size)
    print(
        f"Dataset: {cfg.data.dataset}, img_size: {img_size}, patch_size: {patch_size}"
    )
    print(
        f"num_classes: {num_classes}, training samples: {n_train}, batches: {n_batches}"
    )
    print(
        f"Augmentations: crop_scale={crop_scale}, horizontal_flip={cfg.augment.horizontal_flip}, normalize={cfg.augment.normalize}"
    )

    key, model_key = jax.random.split(key)
    model = IJEPA(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=cfg.model.embed_dim,
        encoder_depth=cfg.model.encoder_depth,
        predictor_dim=cfg.model.predictor_dim,
        predictor_depth=cfg.model.predictor_depth,
        num_heads=cfg.model.num_heads,
        key=model_key,
    )
    target_encoder = jax.tree.map(lambda x: x, model.encoder)

    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"Model parameters: {n_params:,}")

    total_steps = cfg.train.epochs * n_batches
    warmup_steps = int(cfg.train.warmup_frac * total_steps)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=cfg.train.start_lr,
        peak_value=cfg.train.lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=cfg.train.final_lr,
    )

    def map_weight_decay(param):
        if not eqx.is_array(param):
            return False
        if param.ndim > 1:
            return True
        return False

    mask = jax.tree.map(map_weight_decay, model)
    mask = eqx.tree_at(
        lambda m: [
            m.encoder.pos_embed,
            m.predictor.pos_embed,
        ],
        mask,
        [False, False],
    )

    wd_schedule = optax.linear_schedule(cfg.train.wd, cfg.train.wd * 10, total_steps)
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=wd_schedule, mask=mask)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    start_epoch = 0
    if cfg.resume:
        model, target_encoder, opt_state, start_epoch = load_checkpoint(
            cfg.resume, model, target_encoder, opt_state
        )
        print(f"Resumed from {cfg.resume}, epoch {start_epoch}")

    print("Using variable-length masks")

    mask_collator = MaskCollator(
        img_size=img_size,
        patch_size=patch_size,
        n_pred_masks=cfg.mask.n_pred_masks,
    )
    mask_rng = np.random.default_rng(cfg.train.seed)

    ema_schedule = np.linspace(cfg.train.ema_start, cfg.train.ema_end, total_steps)
    step = start_epoch * n_batches

    log_path = Path(__file__).parent / f"{cfg.exp_name}_log.csv"
    if cfg.resume:
        logf = open(log_path, "a")
    else:
        logf = open(log_path, "w")
        logf.write("epoch,loss,top1,top5\n")

    is_multilabel = cfg.data.dataset.lower() == "celeba"

    if cfg.shard:
        n_devices = len(jax.devices())
        mesh = jax.make_mesh((n_devices,), ("batch",))
        data_sharding = NamedSharding(mesh, P("batch"))
        replicate_sharding = NamedSharding(mesh, P())

        model = eqx.filter_shard(model, replicate_sharding)
        opt_state = eqx.filter_shard(opt_state, replicate_sharding)
        target_encoder = eqx.filter_shard(target_encoder, replicate_sharding)
        if cfg.data.batch_size % n_devices != 0:
            raise ValueError(
                f"batch size {cfg.data.batch_size} not divisible by {n_devices} devices"
            )
        print(
            f"Sharding enabled: {n_devices} devices, batch splits to {cfg.data.batch_size // n_devices} per device"
        )
    else:
        data_sharding = None
        replicate_sharding = None

    if not cfg.resume:
        if cfg.eval.probe_interval > 0:
            key, probe_key = jax.random.split(key)
            top1, top5 = evaluate_linear_probe(
                target_encoder,
                train_loader,
                val_loader,
                probe_key,
                num_classes=num_classes,
                is_multilabel=is_multilabel,
                n_epochs=cfg.eval.probe_epochs,
                max_train_samples=cfg.eval.probe_train_max_samples,
                max_val_samples=cfg.eval.probe_val_max_samples,
                data_sharding=data_sharding,
                replicate_sharding=replicate_sharding,
            )
            print(f"Epoch 0 (untrained): top1={top1:.4f}, top5={top5:.4f}")
        else:
            top1, top5 = float("nan"), float("nan")
            print("Epoch 0 (untrained): probe skipped (probe_interval=0)")

        logf.write(f"0,0.0,{top1:.4f},{top5:.4f}\n")
        logf.flush()

    seen_shapes = set()
    slow_steps = []

    for epoch in range(start_epoch, cfg.train.epochs):
        epoch_loss = 0.0
        t_iter_end = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}")
        for batch_idx, (batch_imgs, _) in enumerate(pbar):
            t_loader = time.time()
            t_wait = t_loader - t_iter_end

            t0 = time.time()
            enc_masks, pred_masks = mask_collator(mask_rng, batch_size=len(batch_imgs))
            t_mask = time.time()

            shape_key = (enc_masks.shape, pred_masks.shape)
            is_new_shape = shape_key not in seen_shapes
            if is_new_shape:
                seen_shapes.add(shape_key)

            if cfg.shard:
                enc_masks = jax.device_put(enc_masks, data_sharding)
                pred_masks = jax.device_put(pred_masks, data_sharding)
                batch_imgs = jax.device_put(batch_imgs, data_sharding)
            else:
                enc_masks = jnp.array(enc_masks)
                pred_masks = jnp.array(pred_masks)
                batch_imgs = jnp.array(batch_imgs)
            t1 = time.time()

            (model, target_encoder), opt_state, loss = train_step(
                (model, target_encoder),
                optimizer,
                opt_state,
                batch_imgs,
                enc_masks,
                pred_masks,
            )
            _ = jax.block_until_ready(loss)
            t2 = time.time()

            momentum = ema_schedule[min(step, total_steps - 1)]
            target_encoder = ema_update(model.encoder, target_encoder, momentum)
            t3 = time.time()

            t_total = t3 - t_iter_end
            t_iter_end = t3

            if t_total > 5.0:
                slow_steps.append(
                    {
                        "step": step,
                        "total": t_total,
                        "wait": t_wait,
                        "mask": t_mask - t0,
                        "xfer": t1 - t_mask,
                        "train": t2 - t1,
                        "ema": t3 - t2,
                        "shape": shape_key,
                        "new_shape": is_new_shape,
                    }
                )
                print(
                    f"\n[SLOW STEP {step}] total={t_total:.1f}s wait={t_wait:.1f}s "
                    f"mask={t_mask - t0:.2f}s xfer={t1 - t_mask:.2f}s train={t2 - t1:.1f}s "
                    f"shape={shape_key} new={is_new_shape}"
                )

            step += 1
            epoch_loss += float(loss)
            pbar.set_postfix(
                {
                    "loss": f"{loss:.7f}",
                    "wait": f"{t_wait:.2f}s",
                    "xfer": f"{t1 - t_mask:.2f}s",
                    "train": f"{t2 - t1:.2f}s",
                }
            )

        print(
            f"\nEpoch {epoch + 1} shapes seen: {len(seen_shapes)}, slow steps: {len(slow_steps)}"
        )

        avg_loss = epoch_loss / n_batches

        if cfg.eval.probe_interval > 0 and ((epoch + 1) % cfg.eval.probe_interval == 0):
            key, probe_key = jax.random.split(key)
            top1, top5 = evaluate_linear_probe(
                target_encoder,
                train_loader,
                val_loader,
                probe_key,
                num_classes=num_classes,
                is_multilabel=is_multilabel,
                n_epochs=cfg.eval.probe_epochs,
                max_train_samples=cfg.eval.probe_train_max_samples,
                max_val_samples=cfg.eval.probe_val_max_samples,
                data_sharding=data_sharding,
                replicate_sharding=replicate_sharding,
            )
            print(
                f"Epoch {epoch + 1}: loss={avg_loss:.4f}, top1={top1:.4f}, top5={top5:.4f}"
            )
        else:
            top1, top5 = float("nan"), float("nan")
            print(f"Epoch {epoch + 1}: loss={avg_loss:.4f} (probe skipped)")

        logf.write(f"{epoch + 1},{avg_loss:.6f},{top1:.4f},{top5:.4f}\n")
        logf.flush()

        if (epoch + 1) % cfg.save_interval == 0:
            path = os.path.join(cfg.save_dir, f"{cfg.exp_name}_epoch_{epoch + 1}")
            save_checkpoint(model, target_encoder, opt_state, epoch + 1, cfg, path)
            print(f"Saved checkpoint: {path}")

    path = os.path.join(cfg.save_dir, f"{cfg.exp_name}_final")
    save_checkpoint(model, target_encoder, opt_state, cfg.train.epochs, cfg, path)
    logf.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
