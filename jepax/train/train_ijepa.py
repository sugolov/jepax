import os
import argparse
from collections import OrderedDict
from pathlib import Path
from datetime import datetime
import time

from functools import partial

import jax
from jax import numpy as jnp
import jax.sharding as jshard
import equinox as eqx
import optax
import aim
from tqdm import tqdm

from jepax.data import build_dataloader #build_torch_dataloader
from jepax.model import get_ijepa_model, IJEPAMasker, IJEPA 
from jepax.train.eval_ijepa import evaluate_linear_probe

import wandb

def parse_args():
    p = argparse.ArgumentParser()
    
    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--xla_buckets", type=int, nargs="+", default=[64, 128, 192, 256])
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--bfloat16", action="store_true")
    # profiling / testing
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile_start_step", type=int, default=10)
    p.add_argument("--profile_end_step", type=int, default=60)
    p.add_argument("--profile_log_dir", type=str, default=".logs")
    p.add_argument("--skip_epoch", type=int, default=None)
    # data
    p.add_argument("--data_name", type=str, default="cifar10")
    p.add_argument("--data_dir", type=str, default=".data")
    # model
    p.add_argument("--model_name", type=str, default="ijepa-test",
                   choices=["ijepa-ti", "ijepa-s", "ijepa-b", "ijepa-l", "ijepa-h", "ijepa-test"])
    # logging/checkpointing
    p.add_argument("--save_dir", type=str, default=".checkpoints")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="ijepa")
    p.add_argument("--use_aim", action="store_true")
    p.add_argument("--aim_repo", type=str, default=".aim")
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--print_interval", type=int, default=1)
    p.add_argument("--tag", type=str, default=None)
    # model architecture
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--num_channels", type=int, default=3)
    p.add_argument("--p_drop", type=float, default=0.0)
    # training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--start_lr", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--end_lr", type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int, default=15)
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--final_weight_decay", type=float, default=0.4) 
    # ema
    p.add_argument("--ema_decay", type=float, default=0.996)
    # masking
    p.add_argument("--num_pred_masks", type=int, default=4)
    p.add_argument("--num_pad", type=int, default=64)
    p.add_argument("--pred_scale", type=float, nargs=2, default=[0.15, 0.2])
    p.add_argument("--pred_aspect", type=float, nargs=2, default=[0.75, 1.5])
    p.add_argument("--ctx_scale", type=float, nargs=2, default=[0.85, 1.0])
    p.add_argument("--ctx_aspect", type=float, default=1.0)
    # eval
    p.add_argument("--n_concat", type=int, default=4)
    p.add_argument("--eval_interval", type=int, default=10)
    p.add_argument("--eval_train_samples", type=int, default=None)
    p.add_argument("--eval_val_samples", type=int, default=None)
    p.add_argument("--eval_epochs", type=int, default=50)
    p.add_argument("--eval_batch_size", type=int, default=4096)
    p.add_argument("--eval_optim", type=str, default="adam", choices=["adam", "lars", "sgd"])
    p.add_argument("--eval_weight_decay", type=float, default=5e-4)
    p.add_argument("--eval_lr", type=float, default=1e-2)
    return p.parse_args()

def save_checkpoint(model, ema_encoder, opt_state, epoch, hparams, path):
    import json
    if hasattr(hparams, '__dict__'):
        hparams = vars(hparams)
    
    eqx.tree_serialise_leaves(path + "_model.eqx", model)
    eqx.tree_serialise_leaves(path + "_ema_enc.eqx", ema_encoder)
    eqx.tree_serialise_leaves(path + "_opt.eqx", opt_state)
    
    with open(path + "_meta.json", "w") as f:
        json.dump({'epoch': epoch, 'args': hparams}, f, indent=2)


def load_checkpoint(path):
    import json
    with open(path + "_meta.json", "r") as f:
        checkpoint = json.load(f)
    
    hparams = checkpoint['args']
    
    model, _ = get_ijepa_model(
        hparams['model_name'],
        key=jax.random.PRNGKey(hparams['seed']),
        num_channels=hparams['num_channels'],
        patch_size=hparams['patch_size'],
        img_size=hparams['img_size'],
        p_drop=hparams['p_drop'],
        seq_len=hparams['seq_len'],
    )
    model = eqx.tree_deserialise_leaves(path + "_model.eqx", model)
    ema_encoder = eqx.tree_deserialise_leaves(path + "_ema_enc.eqx", model.encoder)
    
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=hparams['start_lr'],
        peak_value=hparams['lr'],
        end_value=hparams['end_lr'],
        warmup_steps=hparams['warmup_epochs'] * hparams['steps_per_epoch'],
        decay_steps=hparams['epochs'] * hparams['steps_per_epoch'],
    )

    if hparams['final_weight_decay'] is None:
        wd_schedule = lambda _: hparams['weight_decay']
    else:
        wd_schedule = optax.linear_schedule(
            init_value=hparams['weight_decay'],
            end_value=hparams['final_weight_decay'],
            transition_steps=hparams['epochs'] * hparams['steps_per_epoch']
        )

    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=wd_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    opt_state = eqx.tree_deserialise_leaves(path + "_opt.eqx", opt_state)
    
    return model, ema_encoder, optimizer, opt_state, checkpoint['epoch'], \
         lr_schedule, wd_schedule, hparams

def to_bf16(x):
    if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating):
        return x.astype(jnp.bfloat16)
    return x

def eval_probe(encoder, embed_dim, train_loader, val_loader, num_classes, key,
               batch_size, n_concat, optim="adam", weight_decay=5e-4, lr=5e-2,
               max_train_samples=None, max_val_samples=None, n_epochs=20):
    """Run linear probe evaluation."""
    eval_result = evaluate_linear_probe(
        encoder=encoder,
        embed_dim=embed_dim,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        batch_size=batch_size,
        optim=optim,
        key=key,
        lr=lr,
        n_concat=n_concat,
        n_epochs=n_epochs,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        weight_decay=weight_decay
    )

    log_result = {}
    for k, (top1, top5) in eval_result.items():
        log_result[f"{k}_top1"] = top1
        log_result[f"{k}_top5"] = top5
    
    return eval_result, log_result

def get_num_pad(mask_pred, buckets=None):
    if buckets is None:
        return mask_pred.shape[-1]
    counts = int(jnp.max(jnp.sum(mask_pred, axis=-1)))
    for b in sorted(buckets):
        if b >= counts:
            return b
    return buckets[-1]

@eqx.filter_jit
def update_ema(ema_encoder, encoder, decay: float):
    ema_params, ema_static = eqx.partition(ema_encoder, eqx.is_array)
    enc_params, _ = eqx.partition(encoder, eqx.is_array)
    
    new_ema_params = jax.tree.map(
        lambda e, p: decay * e + (1 - decay) * p,
        ema_params, enc_params
    )
    return eqx.combine(new_ema_params, ema_static)


@eqx.filter_jit
def compute_target_reps(ema_encoder, x_b, key):
    keys = jax.random.split(key, x_b.shape[0])
    z_ema = jax.vmap(lambda k, x: ema_encoder(k, x, train=False))(keys, x_b)
    return z_ema


@eqx.filter_value_and_grad
def compute_grads(model, x_b, z_ema, mask_ctx_b, mask_pred_b, num_pad, key):
    keys = jax.random.split(key, x_b.shape[0])

    # ijepa forward
    _, z_pred, mask_idx = jax.vmap(
        lambda k, x, mc, mp: model(k, x, mc, mp, num_pad=num_pad, train=True)
    )(keys, x_b, mask_ctx_b, mask_pred_b)

    # padded loss calculation
    valid = mask_idx >= 0  # (B, num_pad)
    safe_idx = jnp.where(valid, mask_idx, 0)  # avoid OOB indexing
    target = jax.vmap(lambda z, idx: z[idx])(z_ema, safe_idx)  # (B, num_pad, D)

    target = target.astype(jnp.float32)
    z_pred = z_pred.astype(jnp.float32)

    mse = jnp.sum((target - z_pred) ** 2, axis=-1)  # (B, num_pad)
    loss = jnp.sum(mse * valid) / jnp.sum(valid)  # only count valid positions
    
    return loss

def train_ijepa(
    args: dict,
    # misc
    num_workers: int = 4,
    prefetch_factor: int = 4,
    seed: int = 0,
    resume: str = None,
    xla_buckets: tuple = (64, 128, 192, 256),
    bfloat16: bool = False,
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
    num_pad: int = 128,
    pred_scale: tuple = (0.15, 0.2),
    pred_aspect: tuple = (0.75, 1.5),
    ctx_scale: tuple = (0.85, 1.0),
    ctx_aspect: float = 1.0,
    # encoder ema
    ema_decay: float = 0.996,
    # training
    #exp_name: str = "ijepa",
    tag: str = None,
    epochs: int = 300,
    batch_size: int = 64,
    start_lr: float = 1e-4,
    lr: float = 1e-3,
    end_lr: float = 1e-6,
    weight_decay: float = 0.04,
    final_weight_decay: float = 0.4,
    warmup_epochs: int = 10,
    # logging/checkpointing
    save_dir: str = ".checkpoints",
    use_aim: bool = False,
    aim_repo: str = ".aim",
    use_wandb: bool = False,
    wandb_project: str = "ijepa",
    save_interval: int = 10,
    print_interval: int = 1,
    # eval
    n_concat: int = 4,
    eval_interval: int = 10,
    eval_train_samples: int = None, # on entire dataset
    eval_val_samples: int = None,
    eval_epochs: int = 50,
    eval_batch_size: int = 4096,
    eval_optim: str = "adam",
    eval_weight_decay: float = 5e-4,
    eval_lr: float = 1e-2,
    # profiling
    profile: bool = False,
    profile_start_step: int = 10,
    profile_end_step: int = 60,
    profile_log_dir: str = ".logs/profile",
    skip_epoch: int = None,
):
    # setup
    key = jax.random.PRNGKey(seed)
    print(f"JAX backend: {jax.devices()[0].platform}")
    print(f"JAX devices: {jax.devices()}")
    num_devices = len(jax.devices())

    # sharding setup
    mesh = jax.make_mesh((num_devices,), ("batch",))
    data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("batch"))
    model_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec())

    # directory and logging
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    run_name = f"{model_name}-{data_name.lower()}"
    if bfloat16:
        run_name += "-bf16"
    run_name = f"{run_name}-{tag}" if tag else run_name

    logf = open(f"{save_dir}/{run_name}_log.txt", "a" if resume else "w")
    if not resume: logf.write("Epoch,Avg_Loss\n")

    # create dataset and mask collator
    dataloader, num_classes, steps_per_epoch, img_size = build_dataloader(
        data_name, 
        data_dir, 
        batch_size=batch_size,
        num_workers=num_workers, 
        prefetch_factor=prefetch_factor,
        shuffle=False,
        is_train=True,
        sharding=(num_devices > 1),
        seed=seed
    )

    if eval_interval > 0:
        val_loader, _, _, _ = build_dataloader(
            data_name, 
            data_dir, 
            batch_size=batch_size,
            num_workers=num_workers, 
            prefetch_factor=prefetch_factor,
            shuffle=False,
            is_train=False,
            seed=seed
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

    # checkpointing
    if resume is None:
        # initialize model
        key, key_model = jax.random.split(key)
        model, embed_dim = get_ijepa_model(
            model_name,
            key=key_model,
            num_channels=num_channels,
            patch_size=patch_size,
            img_size=img_size,
            p_drop=p_drop,
            seq_len=seq_len,
        )
        ema_encoder = jax.tree.map(lambda x: x, model.encoder)
        
        # initialize optimizer
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=start_lr,
            peak_value=lr,
            end_value=end_lr,
            warmup_steps=warmup_epochs * steps_per_epoch,
            decay_steps=epochs * steps_per_epoch,
        )

        if final_weight_decay is None:
            wd_schedule = lambda _: weight_decay
        else:
            wd_schedule = optax.linear_schedule(
                init_value=weight_decay,
                end_value=final_weight_decay,
                transition_steps=epochs * steps_per_epoch,
            )
    
        optimizer = optax.adamw(
            learning_rate=lr_schedule, 
            weight_decay=wd_schedule
        )
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        hparams = args.copy()
        hparams.update({
            'img_size': img_size,
            'embed_dim': embed_dim,
            'steps_per_epoch': steps_per_epoch,
        })

        start_epoch = 0
    else:
        model, ema_encoder, optimizer, opt_state, start_epoch, lr_schedule, \
            wd_schedule, hparams = load_checkpoint(resume)
        embed_dim = hparams.get('embed_dim')
        bfloat16 = hparams.get('bfloat16', False)

    if bfloat16:
        model = jax.tree.map(to_bf16, model)
        ema_encoder = jax.tree.map(to_bf16, ema_encoder)

    # init logging
    if use_aim: 
        run = aim.Run(repo=aim_repo, experiment=run_name)
    if use_wandb:
        wandb.init(project=wandb_project, name=run_name, config=hparams)

    # record random guess at epoch 0
    if eval_interval > 0 and start_epoch == 0:
        random_top1 = 1 / num_classes
        random_top5 = min(5 / num_classes, 1.0)
        if use_aim:
            run.track(random_top1, name="best_top1", epoch=start_epoch)
            run.track(random_top5, name="best_top5", epoch=start_epoch)
        if use_wandb:
            wandb.log({
                "best_top1": random_top1,
                "best_top5": random_top5,
                "epoch": start_epoch,
            })
    
    # shard model
    model, ema_encoder, opt_state = eqx.filter_shard(
            (model, ema_encoder, opt_state), model_sharding
        )
    
    # step model prep
    @eqx.filter_jit
    def step_model(model, opt_state, x, z_ema, mask_ctx, mask_pred, num_pad, key):

        loss, grads = compute_grads(model, x, z_ema, mask_ctx, mask_pred, num_pad, key)
        updates, opt_state = optimizer.update(grads, opt_state, model)

        model = eqx.apply_updates(model, updates)
        
        return eqx.filter_shard((model, opt_state, loss), model_sharding)
    
    @partial(jax.jit, static_argnums=(1, 2, 3))
    def generate_masks(key, masker, num_pred_masks, batch_size):
        with jax.named_scope("generate_masks"):
            mask_keys = jax.random.split(key, batch_size)
            return jax.vmap(lambda k: masker(k, num_pred_masks, flatten=True))(mask_keys)

    # training loop
    step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, epochs):
        time_ep_start = time.time()
        epoch_losses = []
        pbar = tqdm(dataloader, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs}")

        load_time = time.time()
        for _, batch in enumerate(pbar):  # ignore labels
            load_time = time.time() - load_time

            # profile to check bottlenecks
            if profile and step == profile_start_step:
                print(f"profiling started")
                Path(profile_log_dir).mkdir(parents=True, exist_ok=True)
                jax.profiler.start_trace(profile_log_dir)
            if profile and step == profile_end_step:
                jax.block_until_ready(loss)
                jax.profiler.stop_trace()
                print(f"profiling finished, saved to {profile_log_dir}")
                return
            
            # data for step
            mask_time = time.time()
            key, mask_key, ema_key, step_key = jax.random.split(key, 4)
            mask_ctx, mask_pred = generate_masks(mask_key, masker, num_pred_masks, batch_size)
            mask_time = time.time() - mask_time
            
            x = batch["image"]
            x = x.astype(jnp.bfloat16) if bfloat16 else x
            x = jax.device_put(x, data_sharding)

            mask_ctx = jax.device_put(mask_ctx, data_sharding)
            mask_pred = jax.device_put(mask_pred, data_sharding)

            target_time = time.time()
            z_ema = compute_target_reps(ema_encoder, x, ema_key)
            target_time = time.time() - target_time

            z_ema = z_ema.astype(jnp.bfloat16) if bfloat16 else z_ema

            if step == start_epoch * steps_per_epoch:
                print(f"model dtype: {jax.tree.leaves(eqx.filter(model, eqx.is_array))[0].dtype}")
                print(f"x: {x.shape}, dtype: {x.dtype}")
                print(f"z_ema: {z_ema.shape}, dtype: {z_ema.dtype}")
                print(f"mask_ctx: {mask_ctx.shape}")
                print(f"mask_pred: {mask_pred.shape}")
                print(f"num_pad: {num_pad}")

            # train step
            step_time = time.time()
            model, opt_state, loss = step_model(
                model, opt_state,
                x, z_ema, mask_ctx, mask_pred,
                num_pad, step_key
            )
            ema_encoder = update_ema(ema_encoder, model.encoder, ema_decay)
            assert not jnp.isnan(loss), f"Epoch {epoch+1}/{epochs}:NaN \
                loss at step {step}"
            step_time = time.time() - step_time
            
            # track and log
            step += 1
            epoch_losses.append(loss)
            
            if step % 100 == 0:
                if use_aim:
                    run.track(loss.item(), name="loss", step=step, epoch=epoch)
                if use_wandb:
                    wandb.log({
                        "loss": loss.item(), 
                        "epoch": epoch,
                        "lr": float(lr_schedule(step)),
                        "wd": float(wd_schedule(step))
                    }, 
                        step=step
                    )
            pbar.set_postfix(OrderedDict([
                ("loss", f"{loss:.3f}"),
                ("load_time", f"{load_time:.3f}s"),
                ("step_time", f"{step_time:.3f}s"),
                ("target_time", f"{target_time:.3f}s"),
                ("mask_time", f"{mask_time:.3f}s"),
            ]))
            load_time = time.time()

            if skip_epoch is not None and step == skip_epoch:
                break

        # end of epoch
        # linear probe eval

        if eval_interval > 0 and (epoch + 1) % eval_interval == 0:
            probe_time = time.time()
            key, eval_key = jax.random.split(key)
            print(f"Epoch {epoch+1}/{epochs}: linear probe eval, " \
                    f"{eval_train_samples or 'all'} train, {eval_val_samples or 'all'} val"
                  )
            eval_result, log_result = eval_probe(
                encoder=ema_encoder,
                embed_dim=embed_dim,
                train_loader=dataloader,
                val_loader=val_loader,
                num_classes=num_classes,
                key=eval_key,
                max_train_samples=eval_train_samples,
                max_val_samples=eval_val_samples,
                n_concat=n_concat,
                n_epochs=eval_epochs,
                batch_size=eval_batch_size,
                optim=eval_optim,
                weight_decay=eval_weight_decay,
                lr=eval_lr
            )
            probe_time = time.time() - probe_time
            top1, top5 = eval_result["best"]
            print(f"Epoch {epoch+1}/{epochs}: max top1 {top1*100:.2f}%, max top5 {top5*100:.2f}%" \
                    f"probe time {probe_time:.3f}s"
                )
            
            if use_aim:
                run.track(top1, name="probe_top1", epoch=epoch)
                run.track(top5, name="probe_top5", epoch=epoch)
            if use_wandb:
                wandb.log({**log_result, "epoch": epoch}, step=step)

        # tracking
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_time = time.time() - time_ep_start
        
        print(f"Epoch {epoch+1}/{epochs}: avg. loss {avg_loss:.4f}")
        logf.write(f"{epoch+1},{avg_loss:.4f}\n")
        logf.flush()

        if use_aim:
            run.track(avg_loss, name="avg_loss", step=step, epoch=epoch)
        if use_wandb:
            wandb.log({
                "avg_loss": avg_loss, 
                "epoch_time": epoch_time,
                "epoch": epoch,
            }, 
                step=step
            )

        # save checkpoints
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"{run_name}_epoch_{epoch+1}")
            save_checkpoint(model, ema_encoder, opt_state, epoch + 1, hparams, checkpoint_path)
            print(f"Epoch {epoch+1}/{epochs}: Saved checkpoint to {checkpoint_path}")

    logf.close()


if __name__ == "__main__":
    args = parse_args()
    train_ijepa(args=vars(args), **vars(args))