import argparse
import os
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

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

from jepax.config import EBJEPAConfig, load_ebjepa_config
from jepax.data import build_dataloader
from jepax.data.augmentations import augment_batch
from jepax.data.dataset import build_two_view_dataloader, get_normalize_stats
from jepax.losses import bcs_loss, vicreg_loss
from jepax.model.ebjepa import get_ebjepa_model
from jepax.model.resnet import InferenceResNet
from jepax.train.eval_ijepa import evaluate_linear_probe
from jepax.utils import filter_shard_map


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="~/.checkpoints")
    p.add_argument("--data_dir", type=str, default="~/.data")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save_interval", type=int, default=50)
    p.add_argument("--bfloat16", action="store_true")
    p.add_argument("--shard", action="store_true")
    p.add_argument("--exp_name", type=str, default="ebjepa")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="ebjepa")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--tag", type=str, default=None)
    return p.parse_args()


def init_from_config(cfg: EBJEPAConfig, img_size: int, steps_per_epoch: int, key):
    model_cfg = cfg.model
    train_cfg = cfg.train
    total_steps = train_cfg.epochs * steps_per_epoch

    key, key_model = jax.random.split(key)
    model, embed_dim = get_ebjepa_model(
        model_cfg,
        key=key_model,
        img_size=img_size,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
    )

    n_params = sum(p.size for p in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    has_stateful = model_cfg.proj_norm == "bn" or model_cfg.type == "resnet"
    bn_state = eqx.nn.State(model) if has_stateful else None

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=train_cfg.start_lr,
        peak_value=train_cfg.lr,
        end_value=train_cfg.final_lr,
        warmup_steps=train_cfg.warmup_epochs * steps_per_epoch,
        decay_steps=total_steps,
    )

    def exclude_bias_and_norm(params):
        return jax.tree.map(lambda p: p.ndim >= 2, params)

    if train_cfg.optimizer == "lars":
        optimizer = optax.lars(
            learning_rate=lr_schedule,
            weight_decay=train_cfg.wd,
            trust_coefficient=0.02,
            trust_ratio_mask=exclude_bias_and_norm,
            weight_decay_mask=exclude_bias_and_norm,
            momentum=0.9,
        )
    elif train_cfg.optimizer in ("adam", "adamw"):
        optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=train_cfg.wd)
    else:
        raise ValueError(f"Unknown optimizer: {train_cfg.optimizer}")

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    return model, optimizer, opt_state, lr_schedule, embed_dim, bn_state, key


def save_checkpoint(model, opt_state, bn_state, epoch, cfg: EBJEPAConfig, path):
    eqx.tree_serialise_leaves(path + "_model.eqx", model)
    eqx.tree_serialise_leaves(path + "_opt.eqx", opt_state)
    if bn_state is not None:
        eqx.tree_serialise_leaves(path + "_bn_state.eqx", bn_state)
    with open(path + "_meta.yaml", "w") as f:
        yaml.dump({"epoch": epoch, "config": asdict(cfg)}, f, default_flow_style=False)


def load_checkpoint(path, img_size, steps_per_epoch, key):
    import dacite

    with open(path + "_meta.yaml") as f:
        meta = yaml.safe_load(f)
    cfg = dacite.from_dict(
        EBJEPAConfig, meta["config"], config=dacite.Config(cast=[tuple])
    )
    ret = init_from_config(cfg, img_size, steps_per_epoch, key)
    model, optimizer, opt_state, lr_schedule, embed_dim, bn_state, key = ret
    model = eqx.tree_deserialise_leaves(path + "_model.eqx", model)
    opt_state = eqx.tree_deserialise_leaves(path + "_opt.eqx", opt_state)
    if bn_state is not None and os.path.exists(path + "_bn_state.eqx"):
        bn_state = eqx.tree_deserialise_leaves(path + "_bn_state.eqx", bn_state)
    return (
        meta["epoch"],
        cfg,
        model,
        optimizer,
        opt_state,
        lr_schedule,
        embed_dim,
        bn_state,
        key,
    )


def to_bf16(x):
    if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating):
        return x.astype(jnp.bfloat16)
    return x


def eval_probe(
    encoder, embed_dim, train_loader, val_loader, num_classes, key, cfg_eval
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
        n_concat=1,
        n_epochs=cfg_eval.epochs,
        max_train_samples=cfg_eval.train_samples,
        max_val_samples=cfg_eval.val_samples,
        weight_decay=cfg_eval.wd,
        bn_mode=getattr(cfg_eval, "bn_mode", "ema"),
        modes=getattr(cfg_eval, "modes", None) or ["last", "last_bn"],
        lars_trust_coefficient=0.02,
        warmup_epochs=10,
    )
    log_result = {k: v for k, v in eval_result.items() if k not in ("top1", "top5")}
    return eval_result, log_result


def encode_two_views(model, bn_state, x1, x2, model_key):
    view1_key, view2_key = jax.random.split(model_key)
    keys1 = jax.random.split(view1_key, x1.shape[0])
    keys2 = jax.random.split(view2_key, x2.shape[0])

    def fwd(k, xi, state):
        _, proj, state = model(k, xi, state, train=True)
        return proj, state

    vfwd = jax.vmap(
        fwd,
        axis_name="batch",
        in_axes=(0, 0, None),
        out_axes=(0, None),
    )
    z1, bn_state = vfwd(keys1, x1, bn_state)
    z2, bn_state = vfwd(keys2, x2, bn_state)
    return z1.astype(jnp.float32), z2.astype(jnp.float32), bn_state


def compute_ebjepa_loss(z1, z2, bcs_key, loss_cfg):
    if loss_cfg.type == "vicreg":
        return vicreg_loss(z1, z2, loss_cfg.std_coeff, loss_cfg.cov_coeff)
    if loss_cfg.type == "bcs":
        return bcs_loss(
            z1,
            z2,
            bcs_key,
            loss_cfg.bcs_num_slices,
            loss_cfg.bcs_lmbd,
        )
    raise ValueError(f"Unknown EB-JEPA loss: {loss_cfg.type}")


def ebjepa_loss_and_state(model, bn_state, x1, x2, model_key, bcs_key, loss_cfg):
    z1, z2, bn_state = encode_two_views(model, bn_state, x1, x2, model_key)
    loss_dict = compute_ebjepa_loss(z1, z2, bcs_key, loss_cfg)
    return loss_dict["loss"], (loss_dict, bn_state)


def train_ebjepa(
    cfg: EBJEPAConfig,
    save_dir: str = "./checkpoints",
    data_dir: str | None = None,
    resume: str | None = None,
    save_interval: int = 50,
    bfloat16: bool = False,
    shard: bool = False,
    exp_name: str = "ebjepa",
    use_wandb: bool = False,
    wandb_project: str = "ebjepa",
    wandb_entity: str | None = None,
    tag: str | None = None,
    **kwargs,
):
    data_cfg = cfg.data
    model_cfg = cfg.model
    train_cfg = cfg.train
    loss_cfg = cfg.loss
    aug_cfg = cfg.aug
    eval_cfg = cfg.eval

    key = jax.random.key(train_cfg.seed)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.devices()[0].platform}")
    print(f"JAX devices: {jax.devices()}")
    num_devices = len(jax.devices())

    if shard and num_devices > 1:
        mesh = jax.make_mesh((num_devices,), ("batch",))
        data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("batch"))
        assert data_cfg.batch_size % num_devices == 0
    else:
        data_sharding = None
        shard = False

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if model_cfg.type == "resnet":
        backbone_name = model_cfg.resnet.variant
    else:
        backbone_name = model_cfg.vit.name
    run_name = f"{exp_name}-{backbone_name}-{loss_cfg.type}-{data_cfg.dataset.lower()}"
    if bfloat16:
        run_name += "-bf16"
    if tag:
        run_name = f"{run_name}-{tag}"

    norm_stats = get_normalize_stats(data_cfg.dataset)
    if norm_stats is not None:
        norm_mean = jnp.array(norm_stats[0])
        norm_std = jnp.array(norm_stats[1])
        print(f"Using dataset normalization: mean={norm_stats[0]}, std={norm_stats[1]}")
    else:
        norm_mean = None
        norm_std = None

    logf = open(f"{save_dir}/{run_name}_log.txt", "w")
    logf.write("epoch,step,loss,invariance,var,cov,step_ms\n")

    dataloader, num_classes, steps_per_epoch, img_size = build_two_view_dataloader(
        data_cfg.dataset,
        data_dir,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        prefetch_factor=data_cfg.prefetch_factor,
        shuffle=True,
        is_train=True,
        crop_scale=tuple(aug_cfg.random_crop_scale),
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
            normalize=True,
        )
        train_eval_loader, _, _, _ = build_dataloader(
            data_cfg.dataset,
            data_dir,
            batch_size=data_cfg.batch_size,
            num_workers=data_cfg.num_workers,
            prefetch_factor=data_cfg.prefetch_factor,
            shuffle=False,
            is_train=True,
            seed=train_cfg.seed,
            normalize=True,
        )

    if resume is None:
        start_epoch = 0
        ret = init_from_config(cfg, img_size, steps_per_epoch, key)
        model, optimizer, opt_state, lr_schedule, embed_dim, bn_state, key = ret
    else:
        (
            start_epoch,
            cfg,
            model,
            optimizer,
            opt_state,
            lr_schedule,
            embed_dim,
            bn_state,
            key,
        ) = load_checkpoint(resume, img_size, steps_per_epoch, key)
        data_cfg = cfg.data
        model_cfg = cfg.model
        train_cfg = cfg.train
        loss_cfg = cfg.loss
        aug_cfg = cfg.aug
        eval_cfg = cfg.eval

    if bfloat16:
        model = jax.tree.map(to_bf16, model)

    if use_wandb and wandb is None:
        raise ImportError("wandb is required for --use_wandb. pip install wandb")
    if use_wandb:
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=run_name,
            config=asdict(cfg),
        )

    P = jshard.PartitionSpec

    def apply_optimizer_step(model, opt_state, grads):
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        return eqx.apply_updates(model, updates), opt_state

    if shard:
        # todo: test sharding
        @eqx.filter_jit
        def step_model(model, opt_state, bn_state, x1, x2, model_key, bcs_key):
            @filter_shard_map(
                mesh=mesh,
                in_specs=(P(), P("batch"), P("batch"), P(), P()),
                out_specs=(P(), P(), P()),
            )
            def sharded_loss_and_grad(model, x1, x2, model_key, bcs_key):
                @eqx.filter_value_and_grad(has_aux=True)
                def local_loss(model, x1, x2, model_key, bcs_key):
                    return ebjepa_loss_and_state(
                        model, bn_state, x1, x2, model_key, bcs_key, loss_cfg
                    )

                (loss, (loss_dict, new_st)), grads = local_loss(
                    model, x1, x2, model_key, bcs_key
                )
                loss = jax.lax.pmean(loss, "batch")
                grads = jax.lax.pmean(grads, "batch")
                return loss, grads, (loss_dict, new_st)

            loss, grads, (loss_dict, new_bn_state) = sharded_loss_and_grad(
                model, x1, x2, model_key, bcs_key
            )
            model, opt_state = apply_optimizer_step(model, opt_state, grads)
            return model, opt_state, new_bn_state, loss, loss_dict

    else:

        @eqx.filter_jit
        def step_model(model, opt_state, bn_state, x1, x2, model_key, bcs_key):
            @eqx.filter_value_and_grad(has_aux=True)
            def loss_fn(model, bn_state, x1, x2, model_key, bcs_key):
                return ebjepa_loss_and_state(
                    model, bn_state, x1, x2, model_key, bcs_key, loss_cfg
                )

            (loss, (loss_dict, new_bn_state)), grads = loss_fn(
                model, bn_state, x1, x2, model_key, bcs_key
            )
            model, opt_state = apply_optimizer_step(model, opt_state, grads)
            return model, opt_state, new_bn_state, loss, loss_dict

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
        for batch in pbar:
            load_time = time.time() - load_time

            v1 = batch["view1"]  # [B, H, W, C]
            v2 = batch["view2"]

            key, k1, k2, model_key, bcs_key = jax.random.split(key, 5)
            cjp = aug_cfg.color_jitter_prob
            gsp = aug_cfg.grayscale_prob
            hfp = aug_cfg.hflip_prob
            sp = aug_cfg.solarize_prob
            v1 = augment_batch(k1, v1, cjp, gsp, hfp, sp)
            v2 = augment_batch(k2, v2, cjp, gsp, hfp, sp)

            if norm_mean is not None:
                v1 = (v1 - norm_mean) / norm_std
                v2 = (v2 - norm_mean) / norm_std

            # Transpose HWC -> CHW for model
            x1 = jnp.transpose(v1, (0, 3, 1, 2))
            x2 = jnp.transpose(v2, (0, 3, 1, 2))

            if bfloat16:
                x1 = x1.astype(jnp.bfloat16)
                x2 = x2.astype(jnp.bfloat16)
            if data_sharding is not None:
                x1 = jax.device_put(x1, data_sharding)
                x2 = jax.device_put(x2, data_sharding)

            if step == start_epoch * steps_per_epoch:
                model_dtype = jax.tree.leaves(eqx.filter(model, eqx.is_array))[0].dtype
                print(f"model dtype: {model_dtype}")
                print(f"x1: {x1.shape}, dtype: {x1.dtype}")

            step_time = time.time()
            model, opt_state, bn_state, loss, loss_dict = step_model(
                model, opt_state, bn_state, x1, x2, model_key, bcs_key
            )
            assert not jnp.isnan(loss), f"NaN loss at step {step}"
            step_time = time.time() - step_time

            step += 1
            epoch_losses.append(loss)

            inv = float(loss_dict.get("invariance_loss", 0.0))
            var = float(loss_dict.get("var_loss", 0.0))
            cov = float(loss_dict.get("cov_loss", 0.0))
            step_ms = int(step_time * 1000)

            logf.write(
                f"{epoch + 1},{step},{float(loss):.5f},"
                f"{inv:.5f},{var:.5f},{cov:.5f},{step_ms}\n"
            )

            if step % 100 == 0:
                logf.flush()
                if use_wandb:
                    log_data = {
                        "loss": float(loss),
                        "schedule/lr": float(lr_schedule(step)),
                        "schedule/epoch": epoch,
                    }
                    for k, v in loss_dict.items():
                        if k != "loss":
                            log_data[f"loss/{k}"] = float(v)
                    wandb.log(log_data, step=step)

            pbar.set_postfix(
                OrderedDict(
                    [
                        ("loss", f"{loss:.3f}"),
                        ("inv", f"{inv:.3f}"),
                        ("var", f"{var:.3f}"),
                        ("cov", f"{cov:.3f}"),
                        ("load_ms", int(load_time * 1000)),
                        ("step_ms", step_ms),
                    ]
                )
            )
            load_time = time.time()

        run_probe = (
            eval_cfg.interval > 0
            and val_loader is not None
            and (epoch == 0 or (epoch + 1) % eval_cfg.interval == 0)
        )
        if run_probe:
            probe_time = time.time()
            key, eval_key = jax.random.split(key)
            print("Running linear probe evaluation...")
            if model.uses_resnet:
                eval_encoder = InferenceResNet(
                    backbone=eqx.nn.inference_mode(model.encoder),
                    state=bn_state,
                )
            else:
                eval_encoder = model.encoder
            eval_result, log_result = eval_probe(
                encoder=eval_encoder,
                embed_dim=embed_dim,
                train_loader=train_eval_loader,
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

        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"{run_name}_epoch_{epoch + 1}")
            save_checkpoint(model, opt_state, bn_state, epoch + 1, cfg, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    logf.close()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_ebjepa_config(args.config)
    train_ebjepa(cfg=cfg, **vars(args))
