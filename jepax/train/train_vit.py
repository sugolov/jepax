import os, functools
import argparse
from pathlib import Path
from datetime import datetime

import jax
from jax import numpy as jnp
import numpy as np
import einops
import equinox as eqx
import optax

import aim
from tqdm import tqdm

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
    logits = jax.vmap(model, in_axes=(0, 0, None))(x, key, True)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return jnp.mean(loss)

@eqx.filter_jit
def step_model(model, optimizer, state, x, y, key):
    loss, grads = compute_grads(model, x, y, key)
    updates, new_state = optimizer.update(grads, state, model)

    model = eqx.apply_updates(model, updates)

    return model, new_state, loss

@eqx.filter_jit
def eval_step(model, x, y, key):
    """Compute loss and accuracy for a single batch."""
    logits = jax.vmap(model, in_axes=(0, 0, None))(x, key, False)  # is_training=False
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == y)
    return jnp.mean(loss), acc


def evaluate(model, dataloader, key):
    """Evaluate model on entire dataset."""
    losses = []
    accs = []
    
    for x, y in dataloader:
        key, subkey = jax.random.split(key)
        # Create keys for each sample in batch
        subkeys = jax.random.split(subkey, num=x.shape[0])
        loss, acc = eval_step(model, x, y, subkeys)
        losses.append(loss)
        accs.append(acc)
    
    return sum(losses) / len(losses), sum(accs) / len(accs)

def train_vit_cifar10(args):
    # setup

    print(f"JAX backend: {jax.devices()[0].platform}")
    print(f"JAX devices: {jax.devices()}")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if args.tag is None:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        tag = args.tag
    run_name = "_".join([args.exp_name, tag])

    logf = open(f"{args.save_dir}/{run_name}_log.txt", "a" if args.resume else "w")
    if not args.resume:
        logf.write("Epoch,Train_Loss,Test_Loss,Test_Acc\n")

    # create dataset
    dataloader, num_classes, n_batch, image_size = build_dataset(
        args.data_name,
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=True
    )

    test_dataloader, _, _, _ = build_dataset(
        args.data_name,
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=False
    )


    # initialize model state
    key, key_model = jax.random.split(jax.random.PRNGKey(args.seed))
    model = get_vit_clf_model(args.model_name, num_classes=num_classes, key=key_model)
    optimizer = optax.adamw(learning_rate=args.lr)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    
    losses = []
    step = 0

    # aim
    run = aim.Run(repo=args.aim_repo, experiment=args.exp_name)
    run["hparams"] = vars(args)

    # train loop

    for epoch in range(args.epochs):
        epoch_losses = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        
        for x,y in pbar:
            key, *subkeys = jax.random.split(key, num=x.shape[0] + 1)
            subkeys = jnp.array(subkeys)
            
            model, state, loss = step_model(model, optimizer, state, x, y, subkeys)
            epoch_losses.append(loss)
            losses.append(loss)
            step += 1

            run.track(loss.item(), name="loss", step=step, epoch=epoch)

        avg_loss = sum(epoch_losses) / len(epoch_losses)

        if (epoch % args.eval_interval) == 0 or epoch == args.epochs - 1:
            key, eval_key = jax.random.split(key)
            test_loss, test_acc = evaluate(model, test_dataloader, eval_key)
            
            run.track(test_loss, name="test_loss", step=step, epoch=epoch)
            run.track(test_acc, name="test_acc", step=step, epoch=epoch)
            
            print(f"Epoch: {epoch}/{args.epochs}, Avg Loss: {avg_loss:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
            
            logf.write(f"{epoch},{avg_loss:.4f},{test_loss:.4f},{test_acc:.4f}\n")
            logf.flush()
        elif (epoch % args.print_interval) == 0 or epoch == args.epochs - 1:
            print(f"Epoch: {epoch}/{args.epochs}, Step: {step}, Avg Loss: {avg_loss:.4f}")
            logf.write(f"{epoch},{avg_loss:.4f},N/A,N/A\n")
            logf.flush()

        
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f"{run_name}_epoch_{epoch+1}")
            save_checkpoint(model, state, epoch + 1, args, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    logf.close()
    return model, state, losses


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_name", type=str, default="cifar10")
    p.add_argument("--model_name", type=str, default="vit-ti")
    p.add_argument("--exp_name", type=str, default="vit_cifar10")
    p.add_argument("--tag", type=str, default=None)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-3)

    p.add_argument("--save_dir", type=str, default="~/.checkpoints")
    p.add_argument("--data_dir", type=str, default="~/.data")
    p.add_argument("--aim_repo", type=str, default="~/.aim")

    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=10)
    p.add_argument("--print_interval", type=int, default=1)

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_vit_cifar10(args)