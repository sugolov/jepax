import os, functools


import aim
import argparse

import jax
from jax import numpy as jnp
import numpy as np
import einops

import equinox as eqx

import optax

from jepax.data import build_dataset
from jepax.model import get_vit_clf_model
from jepax.train import save_checkpoint

def load_checkpoint(path, model_name, num_classes, seed):
    import json
    with open(path + "_meta.json", "r") as f:
        checkpoint = json.load(f)

    args = argparse.Namespace(**checkpoint['args'])

    key = jax.random.key(seed)
    model = get_vit_clf_model(name=model_name, num_classes=num_classes, key=key)

    model = eqx.tree_deserialise_leaves(path + "_model.eqx", model)

    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    opt_state = eqx.tree_deserialise_leaves(path + "_opt.eqx", opt_state)

    return model, opt_state, checkpoint['epoch'], args

@eqx.filter_value_and_grad
def compute_grads(model, x, y, key):
    print(key)
    logits = jax.vmap(model, in_axes=(0, 0, None))(x, key, True)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return jnp.mean(loss)

@eqx.filter_jit
def step_model(model, optimizer, state, x, y, key):
    loss, grads = compute_grads(model, x, y, key)
    updates, new_state = optimizer.update(grads, state, model)

    model = eqx.apply_updates(model, updates)

    return model, new_state, loss


def train_vit_cifar10(
    num_steps: int = 50,
    print_every: int = 1,
    seed=0,
    lr=5e-3,
    batch_size=32
):
    key, key_model = jax.random.split(jax.random.PRNGKey(seed))
    dataloader, num_classes, n_batch, image_size = build_dataset(
        "cifar10",
        "~/data",
        batch_size=batch_size,
        num_workers=0
    )

    model = get_vit_clf_model("vit-ti", num_classes=num_classes, key=key_model)

    optimizer = optax.adamw(learning_rate=lr)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    losses = []

    def infinite_trainloader():
        while True:
            yield from dataloader

    for step, (x, y) in zip(range(num_steps), infinite_trainloader()):
        key, *subkeys = jax.random.split(key, num=batch_size + 1)
        subkeys = jnp.array(subkeys)

        print(x.shape)

        (model, state, loss) = step_model(
            model, optimizer, state, x, y, subkeys
        )
        losses.append(loss)

        if (step % print_every) == 0 or step == num_steps - 1:
            print(f"Step: {step}/{num_steps}, Loss: {loss}.")

    return model, state, losses

if __name__ == "__main__":
    train_vit_cifar10()

"""
def train(args):

    dataloader, num_classes, n_train, image_size = build_dataset(
        args.data_name,
        args.data_dir,
        batch_size=args.batch_size,
        is_train=True,
        num_workers=4,
    )
    
    key = jax.random.PRNGKey(args.key)
    model = get_vit_clf_model(args.model_name, num_classes=num_classes, key=key)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", type=str, default="vqvae_cifar10")
    p.add_argument("--tag", type=str, default=None)

    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--vis_interval", type=int, default=10)

    p.add_argument("--n_fid_samples", type=int, default=10_000)
    p.add_argument("--no_fid", action="store_true", help="Skip FID computation for testing")

    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--data_name", type=str, default="CIFAR10")
    p.add_argument("--data_dir", type=str, default="./data/")
    p.add_argument("--aim_repo", type=str, default=None, help="Path to aim repository (defaults to .aim in current dir)")
    
    p.add_argument("--resume", type=str, default=None)

    p.add_argument("--ch", type=int, default=128)
    p.add_argument("--ch_mult", type=str, default="1,2,4")
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--num_embeddings", type=int, default=1024)
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--beta_commit", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

"""