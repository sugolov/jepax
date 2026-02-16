import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray


class FeedForward(eqx.Module):
    """A 2 layer feedforward network"""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    norm: eqx.nn.LayerNorm

    def __init__(self, dim: int, mlp_ratio: float, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        dmid = int(mlp_ratio * dim)
        self.linear1 = eqx.nn.Linear(dim, dmid, key=k1)
        self.linear2 = eqx.nn.Linear(dmid, dim, key=k2)
        self.norm = eqx.nn.LayerNorm(dmid)

    def __call__(self, x):
        """
        Note: the fp is batched

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # x: (B, D)
        x = jax.nn.gelu(jax.vmap(self.linear1)(x))
        x = jax.vmap(self.norm)(x)
        x = jax.nn.gelu(jax.vmap(self.linear2)(x))
        return x


if __name__ == "__main__":
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    model = FeedForward(8, 3.0, key=k1)
    x = jax.random.normal(k2, (4, 8))

    model(x)
    print("forward pass done")
