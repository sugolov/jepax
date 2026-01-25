import jax
import jax.numpy as jnp
import equinox as eqx

import math

from jaxtyping import Key

is_linear = lambda x: isinstance(x, eqx.nn.Linear)


# https://docs.kidger.site/equinox/tricks/
def trunc_init(weight: jax.Array, key: Key, stddev: float = 0.02) -> jax.Array:
    return stddev * jax.random.truncated_normal(key, shape=weight.shape, lower=-2, upper=2)

def init_linear_weight(model, init_fn, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [x.weight
                            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                            if is_linear(x)]
    weights = get_weights(model)
    new_weights = [init_fn(weight, subkey) 
                    for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model

def rescale_linear_weight(model, scale):
    get_weights = lambda m: [x.weight
                            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                            if is_linear(x)]
    weights = get_weights(model)
    new_weights = [weight * scale for weight in weights]
    return eqx.tree_at(get_weights, model, new_weights)
    
def init_depth_weights(layers):
    scales = 1 / jnp.arange(1, len(layers)+1)
    return jax.vmap(rescale_linear_weight)(layers, scales)

def init_layernorm(model):
    is_layernorm = lambda x: isinstance(x, eqx.nn.LayerNorm)

    def init_fn(dim):
        return 1.0 * jnp.ones(dim), jnp.zeros(dim)
    
    print(jax.tree_util.tree_leaves(model, is_leaf=is_layernorm))

    get_params = lambda m: [
        attr
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_layernorm) 
        if is_layernorm(x)
        for attr in (x.weight, x.bias)
    ]

    params = get_params(model)

    init_params = [init_fn(x[0].shape) for x in params]

    new_model = eqx.tree_at(get_params, model, init_params)

    return new_model