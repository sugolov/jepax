import os
import argparse
from pathlib import Path
from datetime import datetime

import jax
from jax import numpy as jnp
import equinox as eqx
import optax
import aim
from tqdm import tqdm

from jepax.data import build_dataset
from jepax.model import get_ijepa_model
from jepax.train import save_checkpoint


# -----------------------------------------------------------------------------
# EMA utilities
# -----------------------------------------------------------------------------

def init_ema(model):
    """Initialize EMA model as a copy of the encoder."""
    return model.encoder


def update_ema(ema_encoder, encoder, decay: float):
    """Update EMA parameters: ema = decay * ema + (1 - decay) * new."""
    ema_params, ema_static = eqx.partition(ema_encoder, eqx.is_array)
    enc_params, _ = eqx.partition(encoder, eqx.is_array)
    
    new_ema_params = jax.tree.map(
        lambda e, p: decay * e + (1 - decay) * p,
        ema_params, enc_params
    )
    return eqx.combine(new_ema_params, ema_static)


# -----------------------------------------------------------------------------
# Masking utilities
# -----------------------------------------------------------------------------

def sample_block_mask(key, grid_size: int, scale: tuple[float, float], aspect_ratio: tuple[float, float] = (0.75, 1.5)):
    """Sample a single block mask.
    
    Returns:
        mask: bool array of shape (grid_size, grid_size), True = masked
    """
    # TODO: implement block masking
    # 1. Sample target block size from scale range
    # 2. Sample aspect ratio
    # 3. Sample random position
    # 4. Create boolean mask
    raise NotImplementedError


def sample_context_mask(key, grid_size: int, scale: tuple[float, float]):
    """Sample context mask (what the encoder sees).
    
    Returns:
        mask: bool array of shape (grid_size, grid_size), True = visible
    """
    # TODO: implement context masking
    raise NotImplementedError


def sample_pred_masks(key, grid_size: int, num_masks: int, scale: tuple[float, float]):
    """Sample multiple prediction target masks.
    
    Returns:
        masks: bool array of shape (num_masks, grid_size, grid_size), True = predict
    """
    # TODO: sample num_masks non-overlapping target blocks
    raise NotImplementedError


def get_masks(key, grid_size: int, num_pred_masks: int, pred_scale: tuple, ctx_scale: tuple):
    """Generate context and prediction masks for a single sample.
    
    Returns:
        mask_ctx: (grid_size, grid_size) - True = visible to encoder
        mask_pred: (num_pred_masks, grid_size, grid_size) - True = prediction target
    """
    k1, k2 = jax.random.split(key)
    
    # TODO: 
    # 1. Sample prediction target masks
    # 2. Sample context mask (should not overlap with prediction targets)
    # 3. Return both
    raise NotImplementedError


# -----------------------------------------------------------------------------
# Loss and training step
# -----------------------------------------------------------------------------

def ijepa_loss(z_pred, z_target, mask_idx):
    """Compute I-JEPA loss: MSE between predicted and target representations.
    
    Args:
        z_pred: predicted representations from predictor
        z_target: target representations from EMA encoder (stop gradient)
        mask_idx: indices of predicted tokens
        
    Returns:
        scalar loss
    """
    # TODO:
    # 1. Extract predicted token representations (last num_pad tokens from z_pred)
    # 2. Gather corresponding target representations using mask_idx
    # 3. Compute MSE loss (ignore padded positions where mask_idx == -1)
    raise NotImplementedError


@eqx.filter_value_and_grad
def compute_grads(model, ema_encoder, x, mask_ctx, mask_pred, num_pad, key):
    """Compute I-JEPA gradients.
    
    Args:
        model: IJEPA model (encoder + predictor)
        ema_encoder: EMA encoder for computing targets
        x: input images
        mask_ctx: context masks
        mask_pred: prediction target masks  
        num_pad: padding size for predictor
        key: PRNG key
    """
    k1, k2 = jax.random.split(key)
    
    # Forward pass through online encoder + predictor
    z, z_full, z_pred, mask_idx = model(k1, x, mask_ctx, mask_pred, num_pad=num_pad, train=True)
    
    # Forward pass through EMA encoder (no gradient)
    z_target = jax.lax.stop_gradient(
        ema_encoder(k2, x, mask=None, train=False)
    )
    
    # Compute loss
    loss = ijepa_loss(z_pred, z_target, mask_idx)
    
    return loss


@eqx.filter_jit
def step_model(model, ema_encoder, optimizer, state, x, mask_ctx, mask_pred, num_pad, ema_decay, key):
    """Single training step."""
    loss, grads = compute_grads(model, ema_encoder, x, mask_ctx, mask_pred, num_pad, key)
    
    updates, new_state = optimizer.update(grads, state, model)
    model = eqx.apply_updates(model, updates)
    
    # Update EMA
    ema_encoder = update_ema(ema_encoder, model.encoder, ema_decay)
    
    return model, ema_encoder, new_state, loss


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------

def train_ijepa(args):
    # Setup
    print(f"JAX backend: {jax.devices()[0].platform}")
    print(f"JAX devices: {jax.devices()}")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    tag = args.tag if args.tag else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.exp_name}_{tag}"

    logf = open(f"{args.save_dir}/{run_name}_log.txt", "a" if args.resume else "w")
    if not args.resume:
        logf.write("Epoch,Avg_Loss\n")

    # Create dataset (no labels needed for self-supervised)
    dataloader, _, n_batch, image_size = build_dataset(
        args.data_name,
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=True
    )

    # Initialize model
    key = jax.random.PRNGKey(args.seed)
    key, key_model = jax.random.split(key)
    
    model = get_ijepa_model(
        args.model_name,
        key=key_model,
        num_channels=args.num_channels,
        patch_size=args.patch_size,
        img_size=args.img_size,
        p_drop=args.p_drop,
        seq_len=args.seq_len,
    )
    
    # Initialize EMA encoder
    ema_encoder = init_ema(model)
    
    # Optimizer with weight decay + warmup
    # TODO: add warmup schedule
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=args.warmup_epochs * n_batch,
        decay_steps=args.epochs * n_batch,
    )
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Derived values
    grid_size = args.img_size // args.patch_size
    
    # Aim logging
    run = aim.Run(repo=args.aim_repo, experiment=args.exp_name)
    run["hparams"] = vars(args)

    step = 0

    # Training loop
    for epoch in range(args.epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for x, _ in pbar:  # ignore labels
            batch_size = x.shape[0]
            key, step_key, mask_key = jax.random.split(key, 3)
            
            # Generate masks for each sample in batch
            mask_keys = jax.random.split(mask_key, batch_size)
            # TODO: vmap get_masks over batch
            # mask_ctx, mask_pred = jax.vmap(
            #     lambda k: get_masks(k, grid_size, args.num_pred_masks, 
            #                         tuple(args.mask_scale), tuple(args.ctx_scale))
            # )(mask_keys)
            
            # PLACEHOLDER: create dummy masks for now
            mask_ctx = jnp.ones((batch_size, grid_size, grid_size), dtype=bool)
            mask_pred = jnp.zeros((batch_size, args.num_pred_masks, grid_size, grid_size), dtype=bool)
            
            # Training step (need to vmap over batch)
            # TODO: properly batch the forward pass
            model, ema_encoder, state, loss = step_model(
                model, ema_encoder, optimizer, state,
                x[0], mask_ctx[0], mask_pred[0],  # single sample for now
                args.num_pad, args.ema_decay, step_key
            )
            
            epoch_losses.append(loss)
            step += 1
            
            run.track(loss.item(), name="loss", step=step, epoch=epoch)
            pbar.set_postfix(loss=f"{loss:.4f}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        print(f"Epoch: {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}")
        logf.write(f"{epoch+1},{avg_loss:.4f}\n")
        logf.flush()

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f"{run_name}_epoch_{epoch+1}")
            # TODO: save both model and ema_encoder
            save_checkpoint(model, state, epoch + 1, args, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    logf.close()
    return model, ema_encoder, state


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    
    # data
    p.add_argument("--data_name", type=str, default="cifar10")
    p.add_argument("--data_dir", type=str, default=".data")
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--num_channels", type=int, default=3)
    
    # model (config-based)
    p.add_argument("--model_name", type=str, default="ijepa-test",
                   choices=["ijepa-ti", "ijepa-s", "ijepa-b", "ijepa-l", "ijepa-h", "ijepa-test"])
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--p_drop", type=float, default=0.0)
    p.add_argument("--seq_len", type=int, default=256)
    
    # masking
    p.add_argument("--num_pred_masks", type=int, default=4,
                   help="Number of prediction target blocks")
    p.add_argument("--num_pad", type=int, default=64,
                   help="Max tokens per prediction mask (for static shapes)")
    p.add_argument("--mask_scale", type=float, nargs=2, default=[0.15, 0.2],
                   help="Scale range for prediction target blocks")
    p.add_argument("--ctx_scale", type=float, nargs=2, default=[0.85, 1.0],
                   help="Scale range for context (visible) blocks")
    
    # ema
    p.add_argument("--ema_decay", type=float, default=0.996,
                   help="EMA decay for target encoder")
    
    # training
    p.add_argument("--exp_name", type=str, default="ijepa")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=10)
    
    # logging/checkpointing
    p.add_argument("--save_dir", type=str, default=".checkpoints")
    p.add_argument("--aim_repo", type=str, default=".aim")
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--print_interval", type=int, default=1)
    
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None)
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_ijepa(args)