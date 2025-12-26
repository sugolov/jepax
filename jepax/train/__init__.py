import equinox as eqx


def save_checkpoint(model, opt_state, epoch, args, path):
    checkpoint = {"epoch": epoch, "args": vars(args)}
    eqx.tree_serialise_leaves(path + "_model.eqx", model)
    eqx.tree_serialise_leaves(path + "_opt.eqx", opt_state)

    import json

    with open(path + "_meta.json", "w") as f:
        json.dump(checkpoint, f)
