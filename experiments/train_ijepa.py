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


from jepax.data.datasets import build_dataset
from jepax.model.ijepa import IJEPA, ema_update
from jepax.masks import MaskCollator


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
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", type=str, default="ijepa")
    p.add_argument(
        "--dataset",
        type=str,
        default="celeba",
        choices=["cifar10", "cifar100", "celeba", "imagenet"],
        help="Dataset to train on",
    )
    p.add_argument(
        "--resize",
        action="store_true",
        help="Resize images to 224x224 (to match ImageNet patch count)",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=512)  # todo: FSDP
    p.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    p.add_argument("--start_lr", type=float, default=2e-4, help="Initial LR for warmup")
    p.add_argument("--final_lr", type=float, default=1e-6, help="Final LR after decay")
    p.add_argument(
        "--warmup_frac",
        type=float,
        default=0.01,
        help="Warmup as fraction of total epochs",
    )
    p.add_argument("--wd", type=float, default=0.04, help="Initial weight decay")
    p.add_argument("--ema_start", type=float, default=0.996)
    p.add_argument("--ema_end", type=float, default=1.0)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--data_dir", type=str, default="~/data")
    # Model
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--encoder_depth", type=int, default=8)
    p.add_argument("--predictor_dim", type=int, default=192)
    p.add_argument("--predictor_depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=6)
    # Masking
    p.add_argument("--n_pred_masks", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--fixed_masks",
        action="store_true",
        help="Use fixed-length masks (avoids JAX retracing but subsamples)",
    )

    p.add_argument(
        "--resume", type=str, default=None, help="Checkpoint path to resume from"
    )

    p.add_argument("--crop_scale_min", type=float, default=0.3, help="Min crop scale")
    p.add_argument("--crop_scale_max", type=float, default=1.0, help="Max crop scale")
    p.add_argument(
        "--horizontal_flip", action="store_true", help="Enable horizontal flip"
    )
    p.add_argument(
        "--no_normalize", action="store_true", help="Disable ImageNet normalization"
    )

    p.add_argument(
        "--shard", action="store_true", help="Enable data parallelism across devices"
    )
    return p.parse_args()


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

    train_reps, train_labels = [], []
    for batch_imgs, batch_labels in train_loader:
        reps = get_representations(encoder, jnp.array(batch_imgs))
        train_reps.append(np.array(reps))
        train_labels.append(np.array(batch_labels))
    train_reps = np.concatenate(train_reps)
    train_labels = np.concatenate(train_labels)

    val_reps, val_labels = [], []
    for batch_imgs, batch_labels in val_loader:
        reps = get_representations(encoder, jnp.array(batch_imgs))
        val_reps.append(np.array(reps))
        val_labels.append(np.array(batch_labels))
    val_reps = np.concatenate(val_reps)
    val_labels = np.concatenate(val_labels)

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


def save_checkpoint(model, target_encoder, opt_state, epoch, args, path):
    eqx.tree_serialise_leaves(path + "_model.eqx", model)
    eqx.tree_serialise_leaves(path + "_target.eqx", target_encoder)
    eqx.tree_serialise_leaves(path + "_opt.eqx", opt_state)
    with open(path + "_meta.json", "w") as f:
        json.dump({"epoch": epoch, "args": vars(args)}, f)


def load_checkpoint(path, model, target_encoder, opt_state):
    model = eqx.tree_deserialise_leaves(path + "_model.eqx", model)
    target_encoder = eqx.tree_deserialise_leaves(path + "_target.eqx", target_encoder)
    opt_state = eqx.tree_deserialise_leaves(path + "_opt.eqx", opt_state)
    with open(path + "_meta.json", "r") as f:
        meta = json.load(f)
    return model, target_encoder, opt_state, meta["epoch"]


def main():
    args = parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print(f"JAX devices: {jax.devices()}")
    key = jax.random.key(args.seed)

    print(f"Loading {args.dataset}...")
    resize_to = 224 if args.resize else None
    crop_scale = (args.crop_scale_min, args.crop_scale_max)
    train_loader, num_classes, n_train, img_size = build_dataset(
        args.dataset,
        args.data_dir,
        batch_size=args.batch_size,
        is_train=True,
        num_workers=0,
        img_size=resize_to,
        crop_scale=crop_scale,
        horizontal_flip=args.horizontal_flip,
        normalize=not args.no_normalize,
    )
    val_loader, _, _, _ = build_dataset(
        args.dataset,
        args.data_dir,
        batch_size=args.batch_size,
        is_train=False,
        num_workers=0,
        img_size=resize_to,
        normalize=not args.no_normalize,
    )
    n_batches = n_train // args.batch_size
    patch_size = get_patch_size(args.dataset, img_size)
    print(f"Dataset: {args.dataset}, img_size: {img_size}, patch_size: {patch_size}")
    print(
        f"num_classes: {num_classes}, training samples: {n_train}, batches: {n_batches}"
    )
    print(
        f"Augmentations: crop_scale={crop_scale}, horizontal_flip={args.horizontal_flip}, "
        f"normalize={not args.no_normalize}"
    )

    key, model_key = jax.random.split(key)
    model = IJEPA(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        predictor_dim=args.predictor_dim,
        predictor_depth=args.predictor_depth,
        num_heads=args.num_heads,
        key=model_key,
    )
    target_encoder = jax.tree.map(lambda x: x, model.encoder)

    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"Model parameters: {n_params:,}")

    total_steps = args.epochs * n_batches
    warmup_steps = int(args.warmup_frac * total_steps)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=args.start_lr,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=args.final_lr,
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

    wd_schedule = optax.linear_schedule(args.wd, args.wd * 10, total_steps)
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=wd_schedule, mask=mask)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    start_epoch = 0
    if args.resume:
        model, target_encoder, opt_state, start_epoch = load_checkpoint(
            args.resume, model, target_encoder, opt_state
        )
        print(f"Resumed from {args.resume}, epoch {start_epoch}")

    if args.fixed_masks:
        raise ValueError("Fixed lengths not currently supported")
        print(f"Using fixed-length masks: enc_keep={enc_keep}, pred_keep={pred_keep}")
    else:
        pred_keep = None
        enc_keep = None
        print("Using variable-length masks")

    mask_collator = MaskCollator(
        img_size=img_size,
        patch_size=patch_size,
        n_pred_masks=args.n_pred_masks,
    )
    mask_rng = np.random.default_rng(args.seed)

    ema_schedule = np.linspace(args.ema_start, args.ema_end, total_steps)
    step = start_epoch * n_batches

    log_path = Path(__file__).parent / f"{args.exp_name}_log.csv"
    if args.resume:
        logf = open(log_path, "a")
    else:
        logf = open(log_path, "w")
        logf.write("epoch,loss,top1,top5\n")

    is_multilabel = args.dataset.lower() == "celeba"

    if args.shard:
        n_devices = len(jax.devices())
        mesh = jax.make_mesh((n_devices,), ("batch",))
        data_sharding = NamedSharding(mesh, P("batch"))
        replicate_sharding = NamedSharding(mesh, P())

        model = eqx.filter_shard(model, replicate_sharding)
        opt_state = eqx.filter_shard(opt_state, replicate_sharding)
        target_encoder = eqx.filter_shard(target_encoder, replicate_sharding)
        if args.batch_size % n_devices != 0:
            raise ValueError(
                f"batch size {args.batch_size} not divisible by {n_devices} devices"
            )
        print(
            f"Sharding enabled: {n_devices} devices, batch splits to {args.batch_size // n_devices} per device"
        )
    else:
        data_sharding = None
        replicate_sharding = None

    if not args.resume:
        key, probe_key = jax.random.split(key)
        top1, top5 = evaluate_linear_probe(
            target_encoder,
            train_loader,
            val_loader,
            probe_key,
            num_classes=num_classes,
            is_multilabel=is_multilabel,
            data_sharding=data_sharding,
            replicate_sharding=replicate_sharding,
        )
        print(f"Epoch 0 (untrained): top1={top1:.4f}, top5={top5:.4f}")
        logf.write(f"0,0.0,{top1:.4f},{top5:.4f}\n")
        logf.flush()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_imgs, _ in pbar:
            t0 = time.time()

            # enc_masks [B, n_enc, L_enc], pred_masks [B, n_pred, L_pred]
            enc_masks, pred_masks = mask_collator(mask_rng, batch_size=len(batch_imgs))
            if args.shard:
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

            step += 1
            epoch_loss += float(loss)
            pbar.set_postfix(
                {
                    "loss": f"{loss:.7f}",
                    "data": f"{t1 - t0:.3f}s",
                    "train": f"{t2 - t1:.3f}s",
                    "ema": f"{t3 - t2:.3f}s",
                }
            )

        avg_loss = epoch_loss / n_batches

        key, probe_key = jax.random.split(key)
        top1, top5 = evaluate_linear_probe(
            target_encoder,
            train_loader,
            val_loader,
            probe_key,
            num_classes=num_classes,
            is_multilabel=is_multilabel,
            data_sharding=data_sharding,
            replicate_sharding=replicate_sharding,
        )

        print(
            f"Epoch {epoch + 1}: loss={avg_loss:.4f}, top1={top1:.4f}, top5={top5:.4f}"
        )
        logf.write(f"{epoch + 1},{avg_loss:.6f},{top1:.4f},{top5:.4f}\n")
        logf.flush()

        if (epoch + 1) % args.save_interval == 0:
            path = os.path.join(args.save_dir, f"{args.exp_name}_epoch_{epoch + 1}")
            save_checkpoint(model, target_encoder, opt_state, epoch + 1, args, path)
            print(f"Saved checkpoint: {path}")

    path = os.path.join(args.save_dir, f"{args.exp_name}_final")
    save_checkpoint(model, target_encoder, opt_state, args.epochs, args, path)
    logf.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
