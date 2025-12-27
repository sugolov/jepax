import argparse
import json
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from jepax.data.datasets import build_dataset
from jepax.model.ijepa import IJEPA, IJEPAEncoder, ema_update
from jepax.masks import MaskCollator


def get_patch_size(dataset: str) -> int:
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
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema_start", type=float, default=0.996)
    p.add_argument("--ema_end", type=float, default=1.0)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--data_dir", type=str, default="~/data")
    # Model
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--encoder_depth", type=int, default=8)
    p.add_argument("--predictor_dim", type=int, default=128)
    p.add_argument("--predictor_depth", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=6)
    # Masking
    p.add_argument("--n_pred_masks", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--resume", type=str, default=None, help="Checkpoint path to resume from"
    )
    return p.parse_args()


def smooth_l1_loss(pred, target):
    diff = pred - target
    abs_diff = jnp.abs(diff)
    return jnp.where(abs_diff < 1.0, 0.5 * diff**2, abs_diff - 0.5).mean()


def forward_single(model, target_encoder, img, ctx_idx, tgt_idx):
    # Target encoder: all patches (no mask)
    target_out = target_encoder(img, mask_indices=None)
    target_emb = target_out[tgt_idx]  # [N_tgt, D]
    target_emb = jax.lax.stop_gradient(target_emb)

    # Context encoder: only context patches
    ctx_emb = model.encoder(img, mask_indices=ctx_idx)  # [N_ctx, D]

    # Predictor: predict target embeddings
    pred_emb = model.predictor(ctx_emb, ctx_idx, tgt_idx)  # [N_tgt, D]

    # Normalize
    pred_norm = pred_emb / (jnp.linalg.norm(pred_emb, axis=-1, keepdims=True) + 1e-6)
    target_norm = target_emb / (
        jnp.linalg.norm(target_emb, axis=-1, keepdims=True) + 1e-6
    )

    return smooth_l1_loss(pred_norm, target_norm)


@eqx.filter_jit
def train_step(
    model, target_encoder, optimizer, opt_state, batch, ctx_masks, tgt_masks
):
    def loss_fn(model):
        losses = []
        for i in range(len(batch)):
            loss = forward_single(
                model, target_encoder, batch[i], ctx_masks[i], tgt_masks[i]
            )
            losses.append(loss)
        return jnp.mean(jnp.stack(losses))

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


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


# todo: can't we just invert matrix and exactly solve?
def evaluate_linear_probe(
    encoder,
    train_loader,
    val_loader,
    key,
    num_classes,
    is_multilabel=False,
    n_epochs=10,
):
    """Train linear probe and evaluate. Returns (top1, top5)."""
    embed_dim = encoder.embed_dim

    probe = LinearProbe(embed_dim, num_classes, key=key)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(probe, eqx.is_array))

    for _ in range(n_epochs):
        for batch_imgs, batch_labels in train_loader:
            batch_imgs = jnp.array(batch_imgs)
            batch_labels = jnp.array(batch_labels)
            reps = get_representations(encoder, batch_imgs)
            probe, opt_state, _ = linear_probe_step(
                probe, optimizer, opt_state, reps, batch_labels, is_multilabel
            )

    total_top1 = 0
    total_top5 = 0
    total_samples = 0

    # todo: can jit?
    for batch_imgs, batch_labels in val_loader:
        batch_imgs = jnp.array(batch_imgs)
        batch_labels = jnp.array(batch_labels)
        reps = get_representations(encoder, batch_imgs)
        logits = probe.veval(reps)

        if is_multilabel:
            preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32)
            correct = (preds == batch_labels).sum()
            total_top1 += correct
            total_top5 += correct
            total_samples += batch_labels.size
        else:
            preds = jnp.argmax(logits, axis=-1)
            total_top1 += (preds == batch_labels).sum()
            top5_preds = jnp.argsort(logits, axis=-1)[:, -5:]
            in_top5 = jnp.any(top5_preds == batch_labels[:, None], axis=-1)
            total_top5 += in_top5.sum()
            total_samples += batch_labels.shape[0]

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
    train_loader, num_classes, n_train, img_size = build_dataset(
        args.dataset,
        args.data_dir,
        batch_size=args.batch_size,
        is_train=True,
        num_workers=4,
    )
    val_loader, _, _, _ = build_dataset(
        args.dataset,
        args.data_dir,
        batch_size=args.batch_size,
        is_train=False,
        num_workers=4,
    )
    n_batches = n_train // args.batch_size
    patch_size = get_patch_size(args.dataset)
    print(f"Dataset: {args.dataset}, img_size: {img_size}, patch_size: {patch_size}")
    print(
        f"num_classes: {num_classes}, training samples: {n_train}, batches: {n_batches}"
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
    # this is hard coded to match the IJEPA init, which isn't ideal, let's just copy weights
    k1, k2 = jax.random.split(model_key)
    target_encoder = IJEPAEncoder(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=args.embed_dim,
        depth=args.encoder_depth,
        num_heads=args.num_heads,
        key=k1,
    )

    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"Model parameters: {n_params:,}")

    schedule = optax.cosine_decay_schedule(args.lr, args.epochs * n_batches)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    start_epoch = 0
    if args.resume:
        model, target_encoder, opt_state, start_epoch = load_checkpoint(
            args.resume, model, target_encoder, opt_state
        )
        print(f"Resumed from {args.resume}, epoch {start_epoch}")

    mask_collator = MaskCollator(
        img_size=img_size,
        patch_size=patch_size,
        n_pred_masks=args.n_pred_masks,
    )
    mask_rng = np.random.default_rng(args.seed)

    total_steps = args.epochs * n_batches
    ema_schedule = np.linspace(args.ema_start, args.ema_end, total_steps)
    step = start_epoch * n_batches

    log_path = Path(__file__).parent / f"{args.exp_name}_log.csv"
    if args.resume:
        logf = open(log_path, "a")
    else:
        logf = open(log_path, "w")
        logf.write("epoch,loss,top1,top5\n")

    is_multilabel = args.dataset.lower() == "celeba"

    if not args.resume:
        key, probe_key = jax.random.split(key)
        top1, top5 = evaluate_linear_probe(
            target_encoder,
            train_loader,
            val_loader,
            probe_key,
            num_classes=num_classes,
            is_multilabel=is_multilabel,
            n_epochs=5,
        )
        print(f"Epoch 0 (untrained): top1={top1:.4f}, top5={top5:.4f}")
        logf.write(f"0,0.0,{top1:.4f},{top5:.4f}\n")
        logf.flush()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_imgs, _ in pbar:
            batch_imgs = jnp.array(batch_imgs)

            # todo: more efficient JAX loading? (eval speed)
            ctx_masks, tgt_masks, _ = mask_collator(
                mask_rng, batch_size=len(batch_imgs)
            )
            ctx_masks = [jnp.array(m) for m in ctx_masks]
            tgt_masks = [jnp.array(m) for m in tgt_masks]

            model, opt_state, loss = train_step(
                model,
                target_encoder,
                optimizer,
                opt_state,
                batch_imgs,
                ctx_masks,
                tgt_masks,
            )

            momentum = ema_schedule[min(step, total_steps - 1)]
            target_encoder = ema_update(model.encoder, target_encoder, momentum)
            step += 1

            epoch_loss += float(loss)
            pbar.set_postfix({"loss": f"{loss:.4f}"})

        avg_loss = epoch_loss / n_batches

        key, probe_key = jax.random.split(key)
        top1, top5 = evaluate_linear_probe(
            target_encoder,
            train_loader,
            val_loader,
            probe_key,
            num_classes=num_classes,
            is_multilabel=is_multilabel,
            n_epochs=5,
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
    save_checkpoint(model, target_encoder, opt_state, args.epochs, path, args)
    logf.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
