import numpy as np
import jax
from jax import numpy as jnp
import equinox as eqx
from typing import Optional
from jaxtyping import Array, Key, Float


# todo: sequence matrix typevar
class FeedForward(eqx.Module):
    """Standard ViT MLP: Linear -> GELU -> Linear"""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, dim: int, mlp_ratio: float, *, key: Key[Array, ""]):
        k1, k2 = jax.random.split(key)
        dmid = int(mlp_ratio * dim)
        self.linear1 = eqx.nn.Linear(dim, dmid, key=k1)
        self.linear2 = eqx.nn.Linear(dmid, dim, key=k2)

    def __call__(self, x: Float[Array, "S D"]) -> Float[Array, "S D"]:
        # x: (S, D)
        x = jax.nn.gelu(jax.vmap(self.linear1)(x))
        x = jax.vmap(self.linear2)(x)
        return x


class PositionalEncoding(eqx.Module):
    """Sinusoidal positional encoding"""

    pe: Float[Array, "S D"]

    def __init__(self, dim: int, seq_len: int = 5000):
        pe = np.zeros((seq_len, dim))
        position = np.arange(0, seq_len, dtype=np.float32)[:, None]
        div_term = np.exp(np.arange(0, dim, 2) * (-np.log(10000.0) / dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = jnp.array(pe)

    def __call__(self, x: Float[Array, "S D"]) -> Float[Array, "S D"]:
        # x: (S, D)
        seq_len = x.shape[0]
        return x + jax.lax.stop_gradient(self.pe[:seq_len])


class Attention(eqx.Module):
    """Multihead attention layer"""

    mha: eqx.nn.MultiheadAttention
    causal: bool

    def __init__(
        self, dim: int, num_head: int, causal: bool = True, *, key: Key[Array, ""]
    ):
        self.mha = eqx.nn.MultiheadAttention(
            num_head,
            dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            key=key,
        )
        self.causal = causal

    def __call__(self, x: Float[Array, "S D"]) -> Float[Array, "S dim"]:
        # x: (S, D)
        S, D = x.shape
        mask = jnp.tri(S, S).T if self.causal else None
        return self.mha(x, x, x, mask)


class TransformerBlock(eqx.Module):
    attn: Attention
    ff: FeedForward
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        dim: int,
        num_head: int,
        causal: bool = True,
        mlp_ratio: float = 3.0,
        p_drop: float = 0.1,
        *,
        key: Key[Array, ""],
    ):
        k1, k2 = jax.random.split(key)
        self.attn = Attention(dim=dim, num_head=num_head, causal=causal, key=k1)
        self.ff = FeedForward(dim=dim, mlp_ratio=mlp_ratio, key=k2)
        self.ln1 = eqx.nn.LayerNorm(dim)
        self.ln2 = eqx.nn.LayerNorm(dim)
        self.dropout = eqx.nn.Dropout(p_drop)

    def __call__(
        self,
        x: Float[Array, "S D"],
        *,
        key: Optional[Key[Array, ""]] = None,
        train: bool = True,
    ):
        # x: (S, D)
        if key is not None:
            k1, k2 = jax.random.split(key)
        else:
            k1 = k2 = None

        # x = x + attn(LN(x))
        attn_out = self.attn(jax.vmap(self.ln1)(x))
        x = x + self.dropout(attn_out, key=k1, inference=not train)

        # x = x + ff(LN(x))
        ff_out = self.ff(jax.vmap(self.ln2)(x))
        x = x + self.dropout(ff_out, key=k2, inference=not train)

        return x


class Transformer(eqx.Module):
    blocks: list
    pe: PositionalEncoding

    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_head: int,
        causal: bool = True,
        mlp_ratio: float = 3.0,
        p_drop: float = 0.1,
        seq_len: int = 2048,
        *,
        key: Key[Array, ""],
    ):
        keys = jax.random.split(key, num_layers)
        self.blocks = [
            TransformerBlock(
                dim=dim,
                num_head=num_head,
                causal=causal,
                mlp_ratio=mlp_ratio,
                p_drop=p_drop,
                key=k,
            )
            for k in keys
        ]
        self.pe = PositionalEncoding(dim=dim, seq_len=seq_len)

    def __call__(
        self,
        x: Float[Array, "S D"],
        *,
        key: Optional[Key[Array, ""]] = None,
        train: bool = True,
    ):
        # x: (S, D)
        x = self.pe(x)

        if key is not None:
            keys = jax.random.split(key, len(self.blocks))
        else:
            keys = [None] * len(self.blocks)

        for block, k in zip(self.blocks, keys):
            x = block(x, key=k, train=train)

        return x
