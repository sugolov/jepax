import argparse
import json
import os
import time
from collections import OrderedDict
from dataclasses import asdict
import dacite
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.sharding as jshard
import optax
from jax import numpy as jnp
from tqdm import tqdm

from jepax.config import Config, load_config
from jepax.data import build_dataloader
from jepax.model import get_ijepa_model, IJEPAMasker
from jepax.train.eval_ijepa import evaluate_linear_probe


def parse_args():
    p = argparse.ArgumentParser()
    # config
    p.add_argument("--config", type=str, required=True)
    # dirs
    p.add_argument("--save_dir", type=str, default="~/.checkpoints")
    p.add_argument("--data_dir", type=str, default="~/.data")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save_interval", type=int, default=50)
    # runtime
    p.add_argument("--bfloat16", action="store_true")
    p.add_argument("--shard", action="store_true")
    p.add_argument("--exp_name", type=str, default="jepa")
    # logging
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="ijepa")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--tag", type=str, default=None)
    # profiling
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile_start_step", type=int, default=10)
    p.add_argument("--profile_end_step", type=int, default=60)
    p.add_argument("--profile_log_dir", type=str, default=".logs")
    return p.parse_args()


def save_checkpoint(model, ema_encoder, opt_state, epoch, hparams, path):
    eqx.tree_serialise_leaves(path + "_model.eqx", model)
    eqx.tree_serialise_leaves(path + "_ema_enc.eqx", ema_encoder)
    eqx.tree_serialise_leaves(path + "_opt.eqx", opt_state)

    with open(path + "_meta.json", "w") as f:
        json.dump({"epoch": epoch, "config": hparams}, f, indent=2)

def load_checkpoint(path, img_size, steps_per_epoch):
    import json
    with open(path + "_meta.json", "r") as f:
        checkpoint = json.load(f)

    cfg = dacite.from_dict(Config, checkpoint["config"], config=dacite.Config(cast=[tuple]))

    model, embed_dim = get_ijepa_model(
        cfg.model.name,
        key=jax.random.key(cfg.train.seed),
        num_channels=cfg.model.num_channels,
        patch_size=cfg.model.patch_size,
        img_size=img_size,
        p_drop=cfg.model.p_drop,
        seq_len=cfg.model.seq_len,
    )
    model = eqx.tree_deserialise_leaves(path + "_model.eqx", model)
    ema_encoder = eqx.tree_deserialise_leaves(path + "_ema_enc.eqx", model.encoder)

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=cfg.train.start_lr,
        peak_value=cfg.train.lr,
        end_value=cfg.train.final_lr,
        warmup_steps=cfg.train.warmup_epochs * steps_per_epoch,
        decay_steps=cfg.train.epochs * steps_per_epoch,
    )

    if cfg.train.final_wd is None:
        wd_schedule = lambda _: cfg.train.wd
    else:
        wd_schedule = optax.linear_schedule(
            init_value=cfg.train.wd,
            end_value=cfg.train.final_wd,
            transition_steps=cfg.train.epochs * steps_per_epoch,
        )

    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=wd_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    opt_state = eqx.tree_deserialise_leaves(path + "_opt.eqx", opt_state)

    return model, ema_encoder, optimizer, opt_state, checkpoint["epoch"], lr_schedule, wd_schedule, cfg


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
        bn_mode=getattr(cfg_eval, "bn_mode", "ema"),
    )

    log_result = {k: v for k, v in eval_result.items() if k not in ("top1", "top5")}
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
def compute_target_reps(ema_encoder, x_b, key):
    keys = jax.random.split(key, x_b.shape[0])
    z_ema = jax.vmap(lambda k, x: ema_encoder(k, x, mask=None, train=False)[0])(keys, x_b)
    return z_ema


@eqx.filter_jit
def normalize_targets(z_ema):
    mean = jnp.mean(z_ema, axis=-1, keepdims=True)
    var = jnp.var(z_ema, axis=-1, keepdims=True)
    return (z_ema - mean) / jnp.sqrt(var + 1e-6)


@eqx.filter_value_and_grad
def compute_grads(model, x_b, z_ema, mask_ctx_b, mask_pred_b, key):
    keys = jax.random.split(key, x_b.shape[0])
    seq_len = z_ema.shape[1]

    z_pred, tgt_indices, n_tgt, pred_start, pred_end = jax.vmap(
        lambda k, x, mc, mp: model(k, x, mc, mp, train=True)
    )(keys, x_b, mask_ctx_b, mask_pred_b)

    z_ema = z_ema.astype(jnp.float32)
    z_pred = z_pred.astype(jnp.float32)

    z_tgt = jax.vmap(lambda z, idx: z[idx])(z_ema, tgt_indices)

    def shift_pred(z_p, ps):
        return jnp.roll(z_p, -ps, axis=0)

    z_pred_shifted = jax.vmap(shift_pred)(z_pred, pred_start)

    pos_idx = jnp.arange(seq_len)[None, :]
    valid_mask = pos_idx < n_tgt[:, None]

    diff = z_pred_shifted - z_tgt
    abs_diff = jnp.abs(diff)
    smooth_l1 = jnp.where(abs_diff < 1.0, 0.5 * diff**2, abs_diff - 0.5)

    loss_per_token = jnp.mean(smooth_l1, axis=-1)
    loss = jnp.sum(loss_per_token * valid_mask) / jnp.sum(valid_mask)

    return loss


def train_ijepa(
    cfg: Config,
    # dirs
    save_dir: str = "./checkpoints",
    data_dir: str | None = None,
    resume: str | None = None,
    save_interval: int = 50,
    # runtime
    bfloat16: bool = False,
    shard: bool = False,
    exp_name: str = "jepa",
    # logging
    use_wandb: bool = False,
    wandb_project: str = "ijepa",
    wandb_entity: str | None = None,
    tag: str | None = None,
    # profiling
    profile: bool = False,
    profile_start_step: int = 10,
    profile_end_step: int = 60,
    profile_log_dir: str = ".logs",
    **kwargs,
):
    """Main training function."""
    data_cfg = cfg.data
    model_cfg = cfg.model
    train_cfg = cfg.train
    mask_cfg = cfg.mask
    eval_cfg = cfg.eval

    key = jax.random.key(train_cfg.seed)
    print(f"JAX backend: {jax.devices()[0].platform}")
    print(f"JAX devices: {jax.devices()}")
    num_devices = len(jax.devices())

    # Sharding setup
    if shard and num_devices > 1:
        mesh = jax.make_mesh((num_devices,), ("batch",))
        data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("batch"))
        model_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec())
    else:
        data_sharding = None
        model_sharding = None

    # Directory and logging
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    run_name = f"{exp_name}-{model_cfg.name}-{data_cfg.dataset.lower()}"
    if bfloat16:
        run_name += "-bf16"
    if tag:
        run_name = f"{run_name}-{tag}"

    logf = open(f"{save_dir}/{run_name}_log.txt", "w")
    logf.write("epoch,itr,loss,mask-A,mask-B,time (ms)\n")

    dataloader, num_classes, steps_per_epoch, img_size = build_dataloader(
        data_cfg.dataset,
        data_cfg.data_dir,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        prefetch_factor=data_cfg.prefetch_factor,
        shuffle=False,
        is_train=True,
        sharding=(num_devices > 1 and shard),
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

    masker = IJEPAMasker(
        height=img_size,
        width=img_size,
        patch_size=model_cfg.patch_size,
        ctx_scale=tuple(mask_cfg.ctx_scale),
        ctx_aspect=mask_cfg.ctx_aspect,
        pred_scale=tuple(mask_cfg.pred_scale),
        pred_aspect=tuple(mask_cfg.pred_aspect),
    )

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

    normalize_tgt = train_cfg.normalize_targets
    print(f"Target normalization: {normalize_tgt}")

    # EMA schedule
    ema_start = train_cfg.ema_start
    ema_end = train_cfg.ema_end
    total_steps = train_cfg.epochs * steps_per_epoch
    print(f"EMA schedule: {ema_start} -> {ema_end}")

    start_epoch = 0

    if bfloat16:
        model = jax.tree.map(to_bf16, model)
        ema_encoder = jax.tree.map(to_bf16, ema_encoder)

    # Init logging
    if use_wandb:
        import wandb

        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=run_name,
            config=hparams,
        )

    if model_sharding is not None:
        model, ema_encoder, opt_state = eqx.filter_shard(
            (model, ema_encoder, opt_state), model_sharding
        )

    @eqx.filter_jit
    def step_model(model, opt_state, x, z_ema, mask_ctx, mask_pred, key):
        loss, grads = compute_grads(model, x, z_ema, mask_ctx, mask_pred, key)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        if model_sharding is not None:
            return eqx.filter_shard((model, opt_state, loss), model_sharding)
        return model, opt_state, loss

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def generate_masks(key, masker, num_pred_masks, batch_size):
        mask_keys = jax.random.split(key, batch_size)
        return jax.vmap(lambda k: masker(k, num_pred_masks, flatten=True))(mask_keys)

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
            if profile and step == profile_start_step:
                print("profiling started")
                Path(profile_log_dir).mkdir(parents=True, exist_ok=True)
                jax.profiler.start_trace(profile_log_dir)

            mask_time = time.time()
            key, mask_key, ema_key, step_key = jax.random.split(key, 4)
            mask_ctx, mask_pred = generate_masks(
                mask_key, masker, mask_cfg.n_pred_masks, data_cfg.batch_size
            )
            mask_time = time.time() - mask_time

            x = batch["image"]
            if bfloat16:
                x = x.astype(jnp.bfloat16)
            if data_sharding is not None:
                x = jax.device_put(x, data_sharding)
                mask_ctx = jax.device_put(mask_ctx, data_sharding)
                mask_pred = jax.device_put(mask_pred, data_sharding)

            target_time = time.time()
            z_ema = compute_target_reps(ema_encoder, x, ema_key)
            if normalize_tgt:
                z_ema = normalize_targets(z_ema)
            target_time = time.time() - target_time

            if step == start_epoch * steps_per_epoch:
                model_dtype = jax.tree.leaves(eqx.filter(model, eqx.is_array))[0].dtype
                print(f"model dtype: {model_dtype}")
                print(f"x: {x.shape}, dtype: {x.dtype}")
                print(f"z_ema: {z_ema.shape}, dtype: {z_ema.dtype}")

            step_time = time.time()
            model, opt_state, loss = step_model(
                model, opt_state, x, z_ema, mask_ctx, mask_pred, step_key
            )
            ema_decay = ema_start + (ema_end - ema_start) * (step / total_steps)
            ema_encoder = update_ema(ema_encoder, model.encoder, ema_decay)
            assert not jnp.isnan(loss), f"NaN loss at step {step}"
            step_time = time.time() - step_time

            # Mask counts for logging
            mask_a = int(jnp.sum(mask_ctx[0]))
            mask_b = int(jnp.sum(mask_pred[0].any(axis=0)))

            # Profile end
            if profile and step == profile_end_step:
                jax.block_until_ready(loss)
                jax.profiler.stop_trace()
                print(f"profiling finished, saved to {profile_log_dir}")
                return

            step += 1
            epoch_losses.append(loss)

            step_ms = int(step_time * 1000)
            logf.write(f"{epoch + 1},{step},{loss:.5f},{mask_a},{mask_b},{step_ms}\n")

            if step % 100 == 0:
                logf.flush()
                if use_wandb:
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
                        },
                        step=step,
                    )

            pbar.set_postfix(
                OrderedDict(
                    [
                        ("loss", f"{loss:.4f}"),
                        ("A", mask_a),
                        ("B", mask_b),
                        ("ms", step_ms),
                    ]
                )
            )
            load_time = time.time()

        # End of epoch - linear probe eval
        run_probe = eval_cfg.interval > 0 and (
            epoch == 0 or (epoch + 1) % eval_cfg.interval == 0
        )
        if run_probe:
            probe_time = time.time()
            key, eval_key = jax.random.split(key)
            print("Running linear probe evaluation...")
            eval_result = eval_probe(
                encoder=ema_encoder,
                embed_dim=embed_dim,
                train_loader=dataloader,
                val_loader=val_loader,
                num_classes=num_classes,
                key=eval_key,
                cfg_eval=eval_cfg,
            )
            probe_time = time.time() - probe_time
            best_top1 = max(v for k, v in eval_result.items() if k.endswith("_top1"))
            best_top5 = max(v for k, v in eval_result.items() if k.endswith("_top5"))
            print(
                f"Epoch {epoch + 1}: best top1 {best_top1 * 100:.2f}%, "
                f"top5 {best_top5 * 100:.2f}% ({probe_time:.1f}s)"
            )
            if use_wandb:
                import wandb

                wandb.log(
                    {
                        "probe/time_s": probe_time,
                        **{f"probe/{k}": v * 100 for k, v in eval_result.items()},
                    },
                    step=step,
                )

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_time = time.time() - time_ep_start
        print(
            f"Epoch {epoch + 1}/{train_cfg.epochs}: "
            f"avg loss {avg_loss:.4f} ({epoch_time:.1f}s)"
        )
        if use_wandb:
            import wandb

            wandb.log(
                {"epoch/avg_loss": avg_loss, "epoch/time_s": epoch_time},
                step=step,
            )

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"{run_name}_epoch_{epoch + 1}")
            save_checkpoint(
                model, ema_encoder, opt_state, epoch + 1, hparams, ckpt_path
            )
            print(f"Saved checkpoint to {ckpt_path}")

    logf.close()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_dir:
        cfg.data.data_dir = args.data_dir
    train_ijepa(cfg=cfg, **vars(args))