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
from jepax.model import get_ijepa_model, IJEPAMasker, IJEPA 
from jepax.train import save_checkpoint

# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    
    # data
    p.add_argument("--data_name", type=str, default="cifar10")
    p.add_argument("--data_dir", type=str, default="~/data")
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
    p.add_argument("--pred_scale", type=float, nargs=2, default=[0.15, 0.2],
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


# -----------------------------------------------------------------------------
# EMA utilities
# -----------------------------------------------------------------------------

def get_num_pad(mask_pred):
    counts = jnp.sum(mask_pred, axis=-1)  # (B, M)
    return int(jnp.max(counts))


def update_ema(ema_encoder, encoder, decay: float):
    ema_params, ema_static = eqx.partition(ema_encoder, eqx.is_array)
    enc_params, _ = eqx.partition(encoder, eqx.is_array)
    
    new_ema_params = jax.tree.map(
        lambda e, p: decay * e + (1 - decay) * p,
        ema_params, enc_params
    )
    return eqx.combine(new_ema_params, ema_static)


@eqx.filter_value_and_grad
def compute_grads(model, x_b, mask_ctx_b, mask_pred_b, num_pad, key):
    keys = jax.random.split(key, x_b.shape[0])

    # 
    _, z_full, z_pred, mask_idx = jax.vmap(
        lambda k, x, mc, mp: model(k, x, mc, mp, num_pad=num_pad, train=True)
    )(keys, x_b, mask_ctx_b, mask_pred_b)


    valid = mask_idx >= 0  # (B, num_pad)
    safe_idx = jnp.where(valid, mask_idx, 0)  # avoid OOB indexing
    target = jax.vmap(lambda z, idx: z[idx])(z_full, safe_idx)  # (B, num_pad, D)

    mse = jnp.sum((target - z_pred) ** 2, axis=-1)  # (B, num_pad)
    loss = jnp.sum(mse * valid) / jnp.sum(valid)  # only count valid positions
    
    return loss


@eqx.filter_jit
def step_model(model, ema_encoder, optimizer, state, x, mask_ctx, mask_pred, num_pad, ema_decay, key):
    """Single training step."""
    loss, grads = compute_grads(model, x, mask_ctx, mask_pred, num_pad, key)
    
    updates, new_state = optimizer.update(grads, state, model)
    model = eqx.apply_updates(model, updates)
    
    # Update EMA
    ema_encoder = update_ema(ema_encoder, model.encoder, ema_decay)

    # set ema_encoder to model
    model = eqx.tree_at(lambda m: m.encoder, model, ema_encoder)
    
    return model, ema_encoder, new_state, loss


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------
def train_ijepa(
    # data
    data_name: str = "cifar10",
    data_dir: str = ".data",
    num_channels: int = 3,
    # model
    model_name: str = "ijepa-test",
    patch_size: int = 4,
    p_drop: float = 0.0,
    seq_len: int = 256,
    # masking
    num_pred_masks: int = 4,
    num_pad: int = 64,
    pred_scale: tuple = (0.15, 0.2),
    pred_aspect: tuple = (0.75, 1.5),
    ctx_scale: tuple = (0.85, 1.0),
    ctx_aspect: float = 1.0,
    # ema
    ema_decay: float = 0.996,
    # training
    exp_name: str = "ijepa",
    tag: str = None,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 10,
    # logging/checkpointing
    save_dir: str = ".checkpoints",
    use_aim: bool = False,
    aim_repo: str = ".aim",
    save_interval: int = 10,
    print_interval: int = 1,
    # misc
    num_workers: int = 4,
    seed: int = 0,
    resume: str = None,
):
    # Setup
    key = jax.random.PRNGKey(seed)
    print(f"JAX backend: {jax.devices()[0].platform}")
    print(f"JAX devices: {jax.devices()}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    tag = tag if tag else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{exp_name}_{tag}"

    logf = open(f"{save_dir}/{run_name}_log.txt", "a" if resume else "w")
    if not resume:
        logf.write("Epoch,Avg_Loss\n")

    # Create dataset + masker
    dataloader, _, n_batch, img_size = build_dataset(
        data_name,
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        is_train=True
    )
    masker = IJEPAMasker(
        height=img_size,
        width=img_size,
        patch_size=patch_size,
        ctx_scale=tuple(ctx_scale),
        ctx_aspect=tuple(ctx_aspect) if isinstance(ctx_aspect, list) else ctx_aspect,
        pred_scale=tuple(pred_scale),
        pred_aspect=tuple(pred_aspect) if isinstance(pred_aspect, list) else pred_aspect,
    )

    # Create model
    key, key_model = jax.random.split(key)
    model = get_ijepa_model(
        model_name,
        key=key_model,
        num_channels=num_channels,
        patch_size=patch_size,
        img_size=img_size,
        p_drop=p_drop,
        seq_len=seq_len,
    )
    ema_encoder = model.encoder
    
    # Optimizer with weight decay + warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_epochs * n_batch,
        decay_steps=epochs * n_batch,
    )
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    
    # Collect hparams for logging
    hparams = dict(
        data_name=data_name, data_dir=data_dir, img_size=img_size, num_channels=num_channels,
        model_name=model_name, patch_size=patch_size, p_drop=p_drop, seq_len=seq_len,
        num_pred_masks=num_pred_masks, num_pad=num_pad, pred_scale=pred_scale, ctx_scale=ctx_scale,
        ema_decay=ema_decay, exp_name=exp_name, tag=tag, epochs=epochs, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay, warmup_epochs=warmup_epochs, save_dir=save_dir,
        aim_repo=aim_repo, save_interval=save_interval, print_interval=print_interval,
        num_workers=num_workers, seed=seed, resume=resume,
    )
    
    # Aim logging
    if use_aim: 
        run = aim.Run(repo=aim_repo, experiment=exp_name)
        run["hparams"] = hparams

    # Training loop
    step = 0
    for epoch in range(epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for x, _ in pbar:  # ignore labels
            key, step_key, mask_key = jax.random.split(key, 3)
            
            # generate masks via vmap
            mask_keys = jax.random.split(mask_key, batch_size)
            mask_ctx, mask_pred = jax.vmap(lambda k: masker(k, num_pred_masks, flatten=True))(mask_keys)
            
            # step
            num_pad = get_num_pad(mask_pred) # get num pred tokens
            model, ema_encoder, state, loss = step_model(
                model, ema_encoder, optimizer, state,
                x, mask_ctx, mask_pred,
                num_pad, ema_decay, step_key
            )
            assert not jnp.isnan(loss), f"NaN loss at step {step}"
            
            epoch_losses.append(loss)
            step += 1
            
            if use_aim:
                run.track(loss.item(), name="loss", step=step, epoch=epoch)
            pbar.set_postfix(loss=f"{loss:.4f}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        print(f"Epoch: {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        logf.write(f"{epoch+1},{avg_loss:.4f}\n")
        logf.flush()

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"{run_name}_epoch_{epoch+1}")
            # TODO: save both model and ema_encoder
            # Note: save_checkpoint expects an args namespace, you may need to adapt this
            save_checkpoint(model, state, epoch + 1, hparams, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    logf.close()
    return model, ema_encoder, state


if __name__ == "__main__":
    args = parse_args()
    train_ijepa(**vars(args))