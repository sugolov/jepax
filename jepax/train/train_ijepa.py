import argparse
import os
import time
from collections import OrderedDict
from dataclasses import asdict
from functools import partial
from pathlib import Path

import dacite
import equinox as eqx
import jax
import jax.sharding as jshard
import optax
import yaml
from jax import numpy as jnp
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from jepax.config import IJEPAConfig, load_ijepa_config
from jepax.data import build_dataloader
from jepax.losses import smooth_l1_loss
from jepax.model import get_ijepa_model, IJEPAMasker
from jepax.train.eval_ijepa import evaluate_linear_probe
from jepax.utils import filter_shard_map


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


def init_from_config(
    cfg: IJEPAConfig, img_size: int, steps_per_epoch: int, key, *, shard: bool = False
):
    model_cfg, train_cfg, mask_cfg = cfg.model, cfg.train, cfg.mask
    total_steps = train_cfg.epochs * steps_per_epoch

    attn_impl = model_cfg.attn_implementation
    if attn_impl is not None and attn_impl not in {"cudnn", "xla"}:
        raise ValueError(
            f"Unsupported model.attn_implementation={attn_impl!r}. "
            "Choose 'cudnn', 'xla', or null."
        )
    if shard and attn_impl == "cudnn":
        print(
            "Warning: --shard with cudnn attention can fail during jit+grad with "
            "XLA stride errors. Falling back to model.attn_implementation=null."
        )
        attn_impl = None

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
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        attn_implementation=attn_impl,
    )
    ema_encoder = jax.tree.map(lambda x: x, model.encoder)

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=train_cfg.start_lr,
        peak_value=train_cfg.lr,
        end_value=train_cfg.final_lr,
        warmup_steps=train_cfg.warmup_epochs * steps_per_epoch,
        decay_steps=total_steps,
    )

    final_wd = getattr(train_cfg, "final_wd", None)
    if final_wd is None:
        wd_schedule = lambda _: train_cfg.wd
    else:
        wd_schedule = optax.linear_schedule(
            init_value=train_cfg.wd,
            end_value=final_wd,
            transition_steps=total_steps,
        )

    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=wd_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    normalize_tgt = train_cfg.normalize_targets
    print(f"Target normalization: {normalize_tgt}")

    ema_scheduler = optax.linear_schedule(
        init_value=train_cfg.ema_start,
        end_value=train_cfg.ema_end,
        transition_steps=total_steps,
    )
    print(f"EMA schedule: {train_cfg.ema_start} -> {train_cfg.ema_end}")

    return (
        model,
        ema_encoder,
        optimizer,
        opt_state,
        masker,
        lr_schedule,
        wd_schedule,
        ema_scheduler,
        embed_dim,
        normalize_tgt,
        key,
    )


def save_checkpoint(model, ema_encoder, opt_state, epoch, cfg: IJEPAConfig, path):
    eqx.tree_serialise_leaves(path + "_model.eqx", model)
    eqx.tree_serialise_leaves(path + "_ema_enc.eqx", ema_encoder)
    eqx.tree_serialise_leaves(path + "_opt.eqx", opt_state)

    with open(path + "_meta.yaml", "w") as f:
        yaml.dump({"epoch": epoch, "config": asdict(cfg)}, f, default_flow_style=False)


def load_checkpoint(path, img_size, steps_per_epoch, key, *, shard: bool = False):
    with open(path + "_meta.yaml", "r") as f:
        meta = yaml.safe_load(f)

    cfg = dacite.from_dict(
        IJEPAConfig, meta["config"], config=dacite.Config(cast=[tuple])
    )

    (
        model,
        ema_encoder,
        optimizer,
        opt_state,
        masker,
        lr_schedule,
        wd_schedule,
        ema_scheduler,
        embed_dim,
        normalize_tgt,
        key,
    ) = init_from_config(cfg, img_size, steps_per_epoch, key, shard=shard)

    model = eqx.tree_deserialise_leaves(path + "_model.eqx", model)
    ema_encoder = eqx.tree_deserialise_leaves(path + "_ema_enc.eqx", ema_encoder)
    opt_state = eqx.tree_deserialise_leaves(path + "_opt.eqx", opt_state)

    return (
        meta["epoch"],
        cfg,
        (
            model,
            ema_encoder,
            optimizer,
            opt_state,
            masker,
            lr_schedule,
            wd_schedule,
            ema_scheduler,
            embed_dim,
            normalize_tgt,
            key,
        ),
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
        modes=getattr(cfg_eval, "modes", None),
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
def compute_target_reps(ema_encoder, x_b):
    def encode(x):
        return ema_encoder(None, x, mask=None, train=False)[0]

    z_ema = jax.vmap(encode)(x_b)
    return z_ema


@eqx.filter_jit
def normalize_targets(z_ema):
    mean = jnp.mean(z_ema, axis=-1, keepdims=True)
    var = jnp.var(z_ema, axis=-1, keepdims=True)
    return (z_ema - mean) / jnp.sqrt(var + 1e-6)


def loss_fn(model, x, z_ema, mask_ctx, mask_pred):
    """
    Loss function for IJEPA

    model: IJEPAModel
    x: input image
    z_ema: ema predicted latents
    mask_ctx: context mask
    mask_pred: predictor mask

    Returns:
        float: scalar loss value
    """
    seq_len = z_ema.shape[1]
    z_pred, tgt_indices, n_tgt = jax.vmap(
        lambda xi, mc, mp: model(None, xi, mc, mp, train=True)
    )(x, mask_ctx, mask_pred)
    z_ema = z_ema.astype(jnp.float32)
    z_pred = z_pred.astype(jnp.float32)
    z_tgt = jax.vmap(lambda z, idx: z[idx])(z_ema, tgt_indices)  # subsetted z_ema

    pos_idx = jnp.arange(seq_len)[None, :]
    valid_mask = pos_idx < n_tgt[:, None]

    return smooth_l1_loss(z_pred, z_tgt, valid_mask)


def make_step_model(optimizer, shard=False, mesh=None):
    grad_fn = eqx.filter_value_and_grad(loss_fn)

    def apply_grads(model, opt_state, loss, grads):
        grad_norms = get_grad_norms(grads)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, grad_norms

    if shard:
        P = jshard.PartitionSpec
        assert mesh is not None, "sharding must specify mesh"

        @eqx.filter_jit
        def step_model(model, opt_state, x, z_ema, mask_ctx, mask_pred):

            # partition spec `P` assumes DDP: naming datch dimensions
            @filter_shard_map(
                mesh=mesh,
                in_specs=(P(), P("batch"), P("batch"), P("batch"), P("batch")),
                out_specs=(P(), P()),
            )
            def sharded_loss_and_grad(model, x, z_ema, mask_ctx, mask_pred):
                loss, grads = grad_fn(model, x, z_ema, mask_ctx, mask_pred)
                loss = jax.lax.pmean(loss, "batch")
                grads = jax.lax.pmean(grads, "batch")
                return loss, grads

            loss, grads = sharded_loss_and_grad(model, x, z_ema, mask_ctx, mask_pred)

            return apply_grads(model, opt_state, loss, grads)

    else:

        @eqx.filter_jit
        def step_model(model, opt_state, x, z_ema, mask_ctx, mask_pred):
            loss, grads = grad_fn(model, x, z_ema, mask_ctx, mask_pred)
            return apply_grads(model, opt_state, loss, grads)

    return step_model


def get_grad_norms(grads):
    # track gradient statistics
    grad_stats = {}
    enc_blocks = grads.encoder.transformer.blocks

    for i, block in enumerate(enc_blocks):
        block_grads = jax.tree.leaves(eqx.filter(block, eqx.is_array))
        if block_grads:
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in block_grads))
            grad_stats[f"enc_block_{i}"] = grad_norm

    pred_blocks = grads.predictor.transformer.blocks
    for i, block in enumerate(pred_blocks):
        block_grads = jax.tree.leaves(eqx.filter(block, eqx.is_array))
        if block_grads:
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in block_grads))
            grad_stats[f"pred_block_{i}"] = grad_norm

    grad_stats["all_grads"] = jnp.sqrt(
        sum((jnp.sum(v**2) for v in list(grad_stats.values())))
    )

    return grad_stats


def train_ijepa(
    cfg: IJEPAConfig,
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
    data_cfg = cfg.data
    model_cfg = cfg.model
    train_cfg = cfg.train
    mask_cfg = cfg.mask
    eval_cfg = cfg.eval

    key = jax.random.key(train_cfg.seed)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.devices()[0].platform}")
    print(f"JAX devices: {jax.devices()}")
    num_devices = len(jax.devices())

    # Sharding setup
    if shard and num_devices > 1:
        mesh = jax.make_mesh((num_devices,), ("batch",))
        data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("batch"))
        assert data_cfg.batch_size % num_devices == 0, (
            f"Batch size {data_cfg.batch_size} must be "
            f"divisible by {num_devices} devices"
        )
    else:
        data_sharding = None
        shard = False
        mesh = None

    # Directory and logging
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    run_name = f"{exp_name}-{model_cfg.name}-{data_cfg.dataset.lower()}"
    if bfloat16:
        run_name += "-bf16"
    if tag:
        run_name = f"{run_name}-{tag}"

    logf = open(f"{save_dir}/{run_name}_log.txt", "w")
    logf.write("epoch,step,loss,mask-A,mask-B,step_ms\n")

    # data loader creation
    dataloader, num_classes, steps_per_epoch, img_size = build_dataloader(
        data_cfg.dataset,
        data_dir,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        prefetch_factor=data_cfg.prefetch_factor,
        shuffle=False,
        is_train=True,
        seed=train_cfg.seed,
    )
    val_loader = None
    if eval_cfg.interval > 0:
        val_loader, _, _, _ = build_dataloader(
            data_cfg.dataset,
            data_dir,
            batch_size=data_cfg.batch_size,
            num_workers=data_cfg.num_workers,
            prefetch_factor=data_cfg.prefetch_factor,
            shuffle=False,
            is_train=False,
            seed=train_cfg.seed,
        )

    if resume is None:
        start_epoch = 0
        (
            model,
            ema_encoder,
            optimizer,
            opt_state,
            masker,
            lr_schedule,
            wd_schedule,
            ema_scheduler,
            embed_dim,
            normalize_tgt,
            key,
        ) = init_from_config(cfg, img_size, steps_per_epoch, key, shard=shard)
    else:
        (
            start_epoch,
            cfg,
            (
                model,
                ema_encoder,
                optimizer,
                opt_state,
                masker,
                lr_schedule,
                wd_schedule,
                ema_scheduler,
                embed_dim,
                normalize_tgt,
                key,
            ),
        ) = load_checkpoint(resume, img_size, steps_per_epoch, key, shard=shard)

        # refresh
        data_cfg = cfg.data
        model_cfg = cfg.model
        train_cfg = cfg.train
        mask_cfg = cfg.mask
        eval_cfg = cfg.eval

    # training optimization
    if bfloat16:
        model = jax.tree.map(to_bf16, model)
        ema_encoder = jax.tree.map(to_bf16, ema_encoder)

    # init logging
    if use_wandb and wandb is None:
        raise ImportError("wandb is required for --use_wandb. pip install wandb")
    if use_wandb:
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=run_name,
            config=asdict(cfg),
        )

    # step model
    step_model = make_step_model(optimizer=optimizer, shard=shard, mesh=mesh)

    # mask helper
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

            # generate masks
            mask_time = time.time()
            key, mask_key = jax.random.split(key)
            mask_ctx, mask_pred = generate_masks(
                mask_key, masker, mask_cfg.n_pred_masks, data_cfg.batch_size
            )
            mask_time = time.time() - mask_time

            mask_a = int(jnp.sum(mask_ctx[0]))
            mask_b = int(jnp.sum(mask_pred[0].any(axis=0)))

            # draw inputs
            x = batch["image"]
            if bfloat16:
                x = x.astype(jnp.bfloat16)
            if data_sharding is not None:
                x = jax.device_put(x, data_sharding)
                mask_ctx = jax.device_put(mask_ctx, data_sharding)
                mask_pred = jax.device_put(mask_pred, data_sharding)

            # z_ema target representation
            target_time = time.time()
            z_ema = compute_target_reps(ema_encoder, x)
            if normalize_tgt:
                z_ema = normalize_targets(z_ema)
            if data_sharding is not None:
                z_ema = jax.device_put(z_ema, data_sharding)
            target_time = time.time() - target_time

            # diagnostics
            if step == start_epoch * steps_per_epoch:
                model_dtype = jax.tree.leaves(eqx.filter(model, eqx.is_array))[0].dtype
                print(f"model dtype: {model_dtype}")
                print(f"x: {x.shape}, dtype: {x.dtype}")
                print(f"z_ema: {z_ema.shape}, dtype: {z_ema.dtype}")
                if data_sharding is not None:
                    print(f"mesh: {data_sharding.mesh}")
                    print(f"x.sharding: {x.sharding}")
                    print(f"z_ema.sharding: {z_ema.sharding}")
                    print(f"mask_ctx.sharding: {mask_ctx.sharding}")

            step_time = time.time()

            # gradient step
            model, opt_state, loss, grad_norms = step_model(
                model, opt_state, x, z_ema, mask_ctx, mask_pred
            )

            # update ema encoder
            ema_encoder = update_ema(ema_encoder, model.encoder, ema_scheduler(step))
            # assert not jnp.isnan(loss), f"NaN loss at step {step}"
            step_time = time.time() - step_time

            # profiler end
            if profile and step == profile_end_step:
                jax.block_until_ready(loss)
                jax.profiler.stop_trace()
                print(f"profiling finished, saved to {profile_log_dir}")
                return

            # logging
            step += 1
            epoch_losses.append(loss)
            step_ms = int(step_time * 1000)
            logf.write(f"{epoch + 1},{step},{loss:.5f},{mask_a},{mask_b},{step_ms}\n")

            if step % 100 == 0:
                logf.flush()
                if use_wandb:
                    wandb.log(
                        {
                            "loss": loss.item(),
                            "schedule/epoch": epoch,
                            "schedule/lr": float(lr_schedule(step)),
                            "schedule/wd": float(wd_schedule(step)),
                            "schedule/ema_enc": float(ema_scheduler(step)),
                            "mask_a": mask_a,
                            "mask_b": mask_b,
                            **{f"grad/{k}": float(v) for k, v in grad_norms.items()},
                        },
                        step=step,
                    )

            pbar.set_postfix(
                OrderedDict(
                    [
                        ("loss", f"{loss:.3f}"),
                        ("mask_A", mask_a),
                        ("mask_B", mask_b),
                        ("load_ms", int(load_time * 1000)),
                        ("step_ms", int(step_time * 1000)),
                        ("tgt_ms", int(target_time * 1000)),
                        ("mask_ms", int(mask_time * 1000)),
                    ]
                )
            )
            load_time = time.time()

        # probe eval
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
            best_top1 = max(v for k, v in eval_result.items() if k.endswith("_top1"))
            best_top5 = max(v for k, v in eval_result.items() if k.endswith("_top5"))
            print(
                f"Epoch {epoch + 1}: best top1 {best_top1 * 100:.2f}%, "
                f"top5 {best_top5 * 100:.2f}% ({probe_time:.1f}s)"
            )
            if use_wandb:
                wandb.log(
                    {
                        "probe/time_s": probe_time,
                        **{f"probe/{k}": v * 100 for k, v in log_result.items()},
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
            wandb.log(
                {"epoch/avg_loss": avg_loss, "epoch/time_s": epoch_time},
                step=step,
            )

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"{run_name}_epoch_{epoch + 1}")
            save_checkpoint(model, ema_encoder, opt_state, epoch + 1, cfg, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    logf.close()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_ijepa_config(args.config)
    train_ijepa(cfg=cfg, **vars(args))
