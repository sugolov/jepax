import argparse
import os
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.sharding as jshard
import optax
from jax import numpy as jnp
from tqdm import tqdm

from jepax.config import load_config, to_dict
from jepax.data import build_dataloader
from jepax.model import get_ijepa_model, IJEPAMasker
from jepax.model.masker import mask_to_indices
from jepax.train.eval_ijepa import evaluate_linear_probe


def parse_args():
    p = argparse.ArgumentParser(
        description="I-JEPA training. Use --config for YAML config, CLI args override."
    )
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    # Common CLI overrides
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


def save_checkpoint(model, ema_encoder, opt_state, epoch, hparams, path):
    import json

    eqx.tree_serialise_leaves(path + "_model.eqx", model)
    eqx.tree_serialise_leaves(path + "_ema_enc.eqx", ema_encoder)
    eqx.tree_serialise_leaves(path + "_opt.eqx", opt_state)

    with open(path + "_meta.json", "w") as f:
        json.dump({"epoch": epoch, "config": hparams}, f, indent=2)


def load_checkpoint(path, cfg):
    import json

    with open(path + "_meta.json", "r") as f:
        checkpoint = json.load(f)

    hparams = checkpoint["config"]

    model, _ = get_ijepa_model(
        hparams["model"]["name"],
        key=jax.random.key(hparams["train"]["seed"]),
        num_channels=hparams["model"]["num_channels"],
        patch_size=hparams["model"]["patch_size"],
        img_size=hparams["img_size"],
        p_drop=hparams["model"]["p_drop"],
        seq_len=hparams["model"]["seq_len"],
    )
    model = eqx.tree_deserialise_leaves(path + "_model.eqx", model)
    ema_encoder = eqx.tree_deserialise_leaves(path + "_ema_enc.eqx", model.encoder)

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=hparams["train"]["start_lr"],
        peak_value=hparams["train"]["lr"],
        end_value=hparams["train"]["final_lr"],
        warmup_steps=hparams["train"]["warmup_epochs"] * hparams["steps_per_epoch"],
        decay_steps=hparams["train"]["epochs"] * hparams["steps_per_epoch"],
    )

    final_wd = hparams["train"].get("final_wd")
    if final_wd is None:
        wd_schedule = lambda _: hparams["train"]["wd"]
    else:
        wd_schedule = optax.linear_schedule(
            init_value=hparams["train"]["wd"],
            end_value=final_wd,
            transition_steps=hparams["train"]["epochs"] * hparams["steps_per_epoch"],
        )

    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=wd_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    opt_state = eqx.tree_deserialise_leaves(path + "_opt.eqx", opt_state)

    return (
        model,
        ema_encoder,
        optimizer,
        opt_state,
        checkpoint["epoch"],
        lr_schedule,
        wd_schedule,
        hparams,
    )


def to_bf16(x):
    if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating):
        return x.astype(jnp.bfloat16)
    return x


def eval_probe(
    encoder,
    embed_dim,
    train_loader,
    val_loader,
    num_classes,
    key,
    cfg_eval,
):
    """Run linear probe evaluation."""
    eval_result = evaluate_linear_probe(
        encoder=encoder,
        embed_dim=embed_dim,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        batch_size=cfg_eval.batch_size,
        optim=cfg_eval.optim,
        key=key,
        lr=cfg_eval.lr,
        n_concat=cfg_eval.n_concat,
        n_epochs=cfg_eval.epochs,
        max_train_samples=cfg_eval.train_samples,
        max_val_samples=cfg_eval.val_samples,
        weight_decay=cfg_eval.wd,
    )

    log_result = {}
    for k, (top1, top5) in eval_result.items():
        log_result[f"{k}_top1"] = top1
        log_result[f"{k}_top5"] = top5

    return eval_result, log_result


@eqx.filter_jit
def update_ema(ema_encoder, encoder, decay: float):
    ema_params, ema_static = eqx.partition(ema_encoder, eqx.is_array)
    enc_params, _ = eqx.partition(encoder, eqx.is_array)

    new_ema_params = jax.tree.map(
        lambda e, p: decay * e + (1 - decay) * p, ema_params, enc_params
    )
    return eqx.combine(new_ema_params, ema_static)


@eqx.filter_jit
def compute_target_reps(ema_encoder, x_b, keys):
    """Compute target representations using EMA encoder (all patches)."""
    z_ema = jax.vmap(lambda k, x: ema_encoder(k, x, indices=None, train=False))(keys, x_b)
    return z_ema


@eqx.filter_jit
def normalize_targets(z_ema):
    """Layer norm on targets before loss computation (prevents collapse)."""
    mean = jnp.mean(z_ema, axis=-1, keepdims=True)
    var = jnp.var(z_ema, axis=-1, keepdims=True)
    return (z_ema - mean) / jnp.sqrt(var + 1e-6)


def masks_to_indices_batch(mask_ctx_b, mask_pred_b):
    """Convert batch of boolean masks to indices, truncated to batch minimum.

    Args:
        mask_ctx_b: [B, N_patches] context block masks
        mask_pred_b: [B, M, N_patches] target block masks

    Returns:
        ctx_indices: [B, min_n_ctx] context indices (truncated to batch min)
        tgt_indices: [B, min_n_tgt] target indices (truncated to batch min)
        min_n_ctx: minimum context count (Python int for static shape)
        min_n_tgt: minimum target count (Python int for static shape)
    """
    n_patches = mask_ctx_b.shape[1]

    # Combine target masks and compute encoder mask (context minus targets)
    mask_tgt = jnp.any(mask_pred_b, axis=1)  # [B, N_patches]
    mask_enc = mask_ctx_b & ~mask_tgt  # [B, N_patches]

    # Convert to indices (padded to n_patches)
    ctx_indices, n_ctx = jax.vmap(lambda m: mask_to_indices(m, n_patches))(mask_enc)
    tgt_indices, n_tgt = jax.vmap(lambda m: mask_to_indices(m, n_patches))(mask_tgt)

    # Compute min and truncate (Python ints for static shapes in JIT)
    min_n_ctx = int(jnp.min(n_ctx))
    min_n_tgt = int(jnp.min(n_tgt))

    ctx_indices = ctx_indices[:, :min_n_ctx]
    tgt_indices = tgt_indices[:, :min_n_tgt]

    return ctx_indices, tgt_indices, min_n_ctx, min_n_tgt


@partial(eqx.filter_value_and_grad)
def compute_grads(model, x_b, z_ema, ctx_indices, tgt_indices, keys):
    """Compute loss and gradients.

    Args:
        model: IJEPA model
        x_b: images [B, C, H, W]
        z_ema: target representations [B, N_patches, D]
        ctx_indices: context indices [B, N_ctx] - pre-truncated to uniform size
        tgt_indices: target indices [B, N_tgt] - pre-truncated to uniform size
        keys: random keys [B, 2] - pre-split and sharded
    """
    # Forward pass - model returns predictions [B, N_tgt, D]
    z_pred = jax.vmap(
        lambda k, x, ci, ti: model(k, x, ci, ti, train=True)
    )(keys, x_b, ctx_indices, tgt_indices)

    z_pred = z_pred.astype(jnp.float32)
    z_ema = z_ema.astype(jnp.float32)

    # Gather target representations from z_ema at target positions
    z_tgt = jax.vmap(lambda z, idx: z[idx])(z_ema, tgt_indices)  # [B, N_tgt, D]

    # Layer norm on targets (helps prevent collapse)
    z_tgt = z_tgt - jnp.mean(z_tgt, axis=-1, keepdims=True)
    z_tgt = z_tgt / jnp.sqrt(jnp.var(z_tgt, axis=-1, keepdims=True) + 1e-6)

    # Compute smooth L1 loss
    diff = z_pred - z_tgt
    abs_diff = jnp.abs(diff)
    smooth_l1 = jnp.where(abs_diff < 1.0, 0.5 * diff**2, abs_diff - 0.5)

    # Mean over all
    loss = jnp.mean(smooth_l1)

    return loss


def train_ijepa(cfg):
    """Main training function."""
    # Unpack config sections
    data_cfg = cfg.data
    model_cfg = cfg.model
    train_cfg = cfg.train
    mask_cfg = cfg.mask
    eval_cfg = cfg.eval
    log_cfg = cfg.logging
    prof_cfg = cfg.profile

    # Setup
    key = jax.random.key(train_cfg.seed)
    print(f"JAX backend: {jax.devices()[0].platform}")
    print(f"JAX devices: {jax.devices()}")
    num_devices = len(jax.devices())

    # Sharding setup
    if cfg.shard and num_devices > 1:
        mesh = jax.make_mesh((num_devices,), ("batch",))
        data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("batch"))
        model_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec())
    else:
        data_sharding = None
        model_sharding = None

    # Directory and logging
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    run_name = f"{model_cfg.name}-{data_cfg.dataset.lower()}"
    if cfg.bfloat16:
        run_name += "-bf16"
    if log_cfg.tag:
        run_name = f"{run_name}-{log_cfg.tag}"

    logf = open(f"{cfg.save_dir}/{run_name}_log.txt", "w")
    logf.write("epoch,itr,loss,mask-A,mask-B,time (ms)\n")

    # Create dataset
    dataloader, num_classes, steps_per_epoch, img_size = build_dataloader(
        data_cfg.dataset,
        data_cfg.data_dir,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        prefetch_factor=data_cfg.prefetch_factor,
        shuffle=False,
        is_train=True,
        sharding=(num_devices > 1 and cfg.shard),
        seed=train_cfg.seed,
    )

    val_loader = None
    if eval_cfg.interval > 0:
        val_loader, _, _, _ = build_dataloader(
            data_cfg.dataset,
            data_cfg.data_dir,
            batch_size=data_cfg.batch_size,
            num_workers=data_cfg.num_workers,
            prefetch_factor=data_cfg.prefetch_factor,
            shuffle=False,
            is_train=False,
            seed=train_cfg.seed,
        )

    # Masker
    masker = IJEPAMasker(
        height=img_size,
        width=img_size,
        patch_size=model_cfg.patch_size,
        ctx_scale=tuple(mask_cfg.ctx_scale),
        ctx_aspect=mask_cfg.ctx_aspect,
        pred_scale=tuple(mask_cfg.pred_scale),
        pred_aspect=tuple(mask_cfg.pred_aspect),
    )

    # Initialize model
    key, key_model = jax.random.split(key)
    model, embed_dim = get_ijepa_model(
        model_cfg.name,
        key=key_model,
        num_channels=model_cfg.num_channels,
        patch_size=model_cfg.patch_size,
        img_size=img_size,
        p_drop=model_cfg.p_drop,
        seq_len=model_cfg.seq_len,
    )
    ema_encoder = jax.tree.map(lambda x: x, model.encoder)

    # Optimizer
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=train_cfg.start_lr,
        peak_value=train_cfg.lr,
        end_value=train_cfg.final_lr,
        warmup_steps=train_cfg.warmup_epochs * steps_per_epoch,
        decay_steps=train_cfg.epochs * steps_per_epoch,
    )

    final_wd = getattr(train_cfg, "final_wd", None)
    if final_wd is None:
        wd_schedule = lambda _: train_cfg.wd
    else:
        wd_schedule = optax.linear_schedule(
            init_value=train_cfg.wd,
            end_value=final_wd,
            transition_steps=train_cfg.epochs * steps_per_epoch,
        )

    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=wd_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Store config with runtime info
    hparams = to_dict(cfg)
    hparams["img_size"] = img_size
    hparams["embed_dim"] = embed_dim
    hparams["steps_per_epoch"] = steps_per_epoch

    normalize_tgt = getattr(train_cfg, "normalize_targets", False)
    print(f"Target normalization: {normalize_tgt}")

    # EMA schedule (linear from ema_start to ema_end)
    ema_start = getattr(train_cfg, "ema_start", 0.996)
    ema_end = getattr(train_cfg, "ema_end", 1.0)
    total_steps = train_cfg.epochs * steps_per_epoch
    print(f"EMA schedule: {ema_start} -> {ema_end}")

    start_epoch = 0

    if cfg.bfloat16:
        model = jax.tree.map(to_bf16, model)
        ema_encoder = jax.tree.map(to_bf16, ema_encoder)

    # Init logging
    if log_cfg.use_wandb:
        import wandb
        wandb.init(
            entity=getattr(log_cfg, "wandb_entity", None),
            project=log_cfg.wandb_project,
            name=run_name,
            config=hparams,
        )

    # Shard model
    if model_sharding is not None:
        model, ema_encoder, opt_state = eqx.filter_shard(
            (model, ema_encoder, opt_state), model_sharding
        )

    # JIT compiled functions
    @eqx.filter_jit
    def step_model(model, opt_state, x, z_ema, ctx_indices, tgt_indices, keys):
        loss, grads = compute_grads(model, x, z_ema, ctx_indices, tgt_indices, keys)

        # Track gradient statistics per block
        grad_stats = {}
        enc_blocks = grads.encoder.transformer.blocks
        for i, block in enumerate(enc_blocks):
            block_grads = jax.tree.leaves(eqx.filter(block, eqx.is_array))
            if block_grads:
                grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in block_grads))
                grad_stats[f"enc_{i}"] = grad_norm

        pred_blocks = grads.predictor.transformer.blocks
        for i, block in enumerate(pred_blocks):
            block_grads = jax.tree.leaves(eqx.filter(block, eqx.is_array))
            if block_grads:
                grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in block_grads))
                grad_stats[f"pred_{i}"] = grad_norm

        grad_stats["total"] = jnp.sqrt(
            sum(jnp.sum(v**2) for v in grad_stats.values())
        )

        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        if model_sharding is not None:
            return *eqx.filter_shard((model, opt_state, loss), model_sharding), grad_stats
        return model, opt_state, loss, grad_stats

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def generate_masks(key, masker, num_pred_masks, batch_size):
        mask_keys = jax.random.split(key, batch_size)
        return jax.vmap(lambda k: masker(k, num_pred_masks, flatten=True))(mask_keys)

    def process_masks(mask_ctx, mask_pred):
        """Convert boolean masks to indices (not JIT'd - needs Python ints for static shapes)."""
        return masks_to_indices_batch(mask_ctx, mask_pred)

    # Training loop
    step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, train_cfg.epochs):
        time_ep_start = time.time()
        epoch_losses = []
        pbar = tqdm(
            dataloader,
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{train_cfg.epochs}",
        )

        load_time = time.time()
        for _, batch in enumerate(pbar):
            load_time = time.time() - load_time

            # Profiling
            if prof_cfg.enabled and step == prof_cfg.start_step:
                print("profiling started")
                Path(prof_cfg.log_dir).mkdir(parents=True, exist_ok=True)
                jax.profiler.start_trace(prof_cfg.log_dir)

            # Generate masks
            mask_time = time.time()
            key, mask_key, ema_key, step_key = jax.random.split(key, 4)
            mask_ctx, mask_pred = generate_masks(
                mask_key, masker, mask_cfg.n_pred_masks, data_cfg.batch_size
            )
            # Convert to indices (truncated to batch min for static shapes)
            ctx_indices, tgt_indices, min_n_ctx, min_n_tgt = process_masks(mask_ctx, mask_pred)
            mask_time = time.time() - mask_time

            x = batch["image"]
            if cfg.bfloat16:
                x = x.astype(jnp.bfloat16)

            # Split keys for batch (before sharding)
            batch_size = x.shape[0]
            ema_keys = jax.random.split(ema_key, batch_size)
            step_keys = jax.random.split(step_key, batch_size)

            if data_sharding is not None:
                x = jax.device_put(x, data_sharding)
                ctx_indices = jax.device_put(ctx_indices, data_sharding)
                tgt_indices = jax.device_put(tgt_indices, data_sharding)
                ema_keys = jax.device_put(ema_keys, data_sharding)
                step_keys = jax.device_put(step_keys, data_sharding)

            # Target representations (EMA encoder on full images)
            target_time = time.time()
            z_ema = compute_target_reps(ema_encoder, x, ema_keys)
            target_time = time.time() - target_time

            # Debug info on first step
            if step == start_epoch * steps_per_epoch:
                model_dtype = jax.tree.leaves(eqx.filter(model, eqx.is_array))[0].dtype
                print(f"model dtype: {model_dtype}")
                print(f"x: {x.shape}, dtype: {x.dtype}")
                print(f"z_ema: {z_ema.shape}, dtype: {z_ema.dtype}")
                print(f"ctx_indices: {ctx_indices.shape}, min_n_ctx: {min_n_ctx}")
                print(f"tgt_indices: {tgt_indices.shape}, min_n_tgt: {min_n_tgt}")

            # Train step
            step_time = time.time()
            model, opt_state, loss, grad_stats = step_model(
                model, opt_state, x, z_ema, ctx_indices, tgt_indices, step_keys
            )
            # EMA with linear schedule
            ema_decay = ema_start + (ema_end - ema_start) * (step / total_steps)
            ema_encoder = update_ema(ema_encoder, model.encoder, ema_decay)
            assert not jnp.isnan(loss), f"NaN loss at step {step}"
            step_time = time.time() - step_time

            # Mask counts for logging
            mask_a = min_n_ctx  # context patches (visible to encoder)
            mask_b = min_n_tgt  # target patches

            # Profile end
            if prof_cfg.enabled and step == prof_cfg.end_step:
                jax.block_until_ready(loss)
                jax.profiler.stop_trace()
                print(f"profiling finished, saved to {prof_cfg.log_dir}")
                return

            step += 1
            epoch_losses.append(loss)

            # Logging
            step_ms = int(step_time * 1000)
            logf.write(f"{epoch + 1},{step},{loss:.5f},{mask_a},{mask_b},{step_ms}\n")

            if step % 100 == 0:
                logf.flush()
                if log_cfg.use_wandb:
                    import wandb
                    wandb.log(
                        {
                            "loss": loss.item(),
                            "epoch": epoch,
                            "lr": float(lr_schedule(step)),
                            "wd": float(wd_schedule(step)),
                            "ema_decay": ema_decay,
                            "mask_a": mask_a,
                            "mask_b": mask_b,
                            **{f"grad/{k}": float(v) for k, v in grad_stats.items()},
                        },
                        step=step,
                    )

            pbar.set_postfix(
                OrderedDict([
                    ("loss", f"{loss:.4f}"),
                    ("A", mask_a),
                    ("B", mask_b),
                    ("ms", step_ms),
                ])
            )
            load_time = time.time()

        # End of epoch - linear probe eval (always run on epoch 1 for baseline)
        run_probe = eval_cfg.interval > 0 and (
            epoch == 0 or (epoch + 1) % eval_cfg.interval == 0
        )
        if run_probe:
            probe_time = time.time()
            key, eval_key = jax.random.split(key)
            print("Running linear probe evaluation...")
            eval_result, log_result = eval_probe(
                encoder=ema_encoder,
                embed_dim=embed_dim,
                train_loader=dataloader,
                val_loader=val_loader,
                num_classes=num_classes,
                key=eval_key,
                cfg_eval=eval_cfg,
            )
            probe_time = time.time() - probe_time
            top1, top5 = eval_result["best"]
            print(
                f"Epoch {epoch + 1}: top1 {top1 * 100:.2f}%, "
                f"top5 {top5 * 100:.2f}% ({probe_time:.1f}s)"
            )
            if log_cfg.use_wandb:
                import wandb
                wandb.log(
                    {
                        "probe/top1": top1 * 100,
                        "probe/top5": top5 * 100,
                        "probe/time_s": probe_time,
                        **{f"probe/{k}": v for k, v in log_result.items()},
                    },
                    step=step,
                )

        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_time = time.time() - time_ep_start
        print(
            f"Epoch {epoch + 1}/{train_cfg.epochs}: "
            f"avg loss {avg_loss:.4f} ({epoch_time:.1f}s)"
        )
        if log_cfg.use_wandb:
            import wandb
            wandb.log(
                {"epoch/avg_loss": avg_loss, "epoch/time_s": epoch_time},
                step=step,
            )

        # Save checkpoint
        if (epoch + 1) % cfg.save_interval == 0:
            ckpt_path = os.path.join(cfg.save_dir, f"{run_name}_epoch_{epoch + 1}")
            save_checkpoint(
                model, ema_encoder, opt_state, epoch + 1, hparams, ckpt_path
            )
            print(f"Saved checkpoint to {ckpt_path}")

    logf.close()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    train_ijepa(cfg)
