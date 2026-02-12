from typing import Optional

import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Float, Key


class PositionalEncoding(eqx.Module):
    """Sinusoidal positional encoding."""

    pe: Array
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int, seq_len: int = 5000):
        self.dim = dim
        pe = np.zeros((seq_len, dim), dtype=np.float32)
        position = np.arange(0, seq_len, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, dim, 2, dtype=np.float32) * (-np.log(10000.0) / dim)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = jnp.array(pe)

    def pe_from_idx(self, idx: Array) -> Float[Array, "S D"]:
        return jax.lax.stop_gradient(jnp.take(self.pe, idx, axis=0))

    def __call__(self, x: Float[Array, "S D"]) -> Float[Array, "S D"]:
        seq_len = x.shape[0]
        return x + jax.lax.stop_gradient(self.pe[:seq_len].astype(x.dtype))


class PositionalEncoding2D(PositionalEncoding):
    """2D sinusoidal positional encoding for vision transformers."""

    grid: Array
    dim: int = eqx.field(static=True)

    def __init__(self, grid_size: int, dim: int, seq_len: int = 5000):
        super().__init__(dim // 2, seq_len=seq_len)

        self.dim = dim
        self.grid = jnp.array(self._get_pe_grid(grid_size))

    def _get_pe_grid(self, grid_size: int) -> Array:
        grid_h = np.arange(grid_size, dtype=int)
        grid_w = np.arange(grid_size, dtype=int)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0).astype(int)
        return grid

    def _get_pe_from_grid(self, grid: Array) -> Float[Array, "S D"]:
        encx = self.pe_from_idx(grid[0].flatten())
        ency = self.pe_from_idx(grid[1].flatten())
        enc = jnp.concatenate([encx, ency], axis=-1)
        return enc

    def __call__(self, x: Float[Array, "S D"]) -> Float[Array, "S D"]:
        """Add 2D positional encoding. x is (grid_size * grid_size, D) tokens."""
        assert x.shape[0] == self.grid.shape[1] ** 2

        encx = self.pe_from_idx(self.grid[0].flatten())
        ency = self.pe_from_idx(self.grid[1].flatten())
        enc = jnp.concatenate([encx, ency], axis=-1, dtype=x.dtype)

        return x + enc


class FeedForward(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, dim: int, mlp_ratio: float, *, key: Key[Array, ""]):
        k1, k2 = jax.random.split(key)
        dmid = int(mlp_ratio * dim)
        self.linear1 = eqx.nn.Linear(dim, dmid, key=k1)
        self.linear2 = eqx.nn.Linear(dmid, dim, key=k2)

    def __call__(self, x: Float[Array, "S D"]) -> Float[Array, "S D"]:
        x = jax.nn.gelu(jax.vmap(self.linear1)(x))
        x = jax.vmap(self.linear2)(x)
        return x


class Attention(eqx.Module):
    """Multihead attention layer with fused QKV projection."""

    qkv_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    num_head: int
    dim: int
    causal: bool

    def __init__(
        self, dim: int, num_head: int, causal: bool = False, *, key: Key[Array, ""]
    ):
        k1, k2 = jax.random.split(key)
        self.dim = dim
        self.num_head = num_head
        self.causal = causal
        self.qkv_proj = eqx.nn.Linear(dim, 3 * dim, key=k1)
        self.out_proj = eqx.nn.Linear(dim, dim, key=k2)

    def __call__(
        self, x: Float[Array, "S D"], mask: Optional[Array] = None
    ) -> Float[Array, "S D"]:
        S, D = x.shape

        qkv = jax.vmap(self.qkv_proj)(x)  # (S, 3D)
        qkv = qkv.reshape(S, self.num_head, -1)  # (S, H, 3D/H)
        qkv = qkv.transpose(1, 0, 2)  # (H, S, 3D/H)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # (H, S, D/H)

        mask_causal = jnp.tri(S, S, dtype=bool).T if self.causal else None

        if mask is None:
            mask = mask_causal
        elif self.causal:
            mask = mask.astype(bool) | mask_causal

        vals = self._attention(q, k, v, mask=mask)  # (H, S, D/H)
        vals = vals.transpose(1, 0, 2)  # (S, H, D/H)
        vals = vals.reshape(S, -1)  # (S, D)

        out = jax.vmap(self.out_proj)(vals)  # (S, D)

        return out

    def _attention(
        self,
        q: Float[Array, "H S Dh"],
        k: Float[Array, "H S Dh"],
        v: Float[Array, "H S Dh"],
        mask: Optional[Array] = None,
    ) -> Float[Array, "H S Dh"]:
        """Scaled dot-product attention."""
        d = q.shape[-1]
        logits = q @ k.transpose(0, 2, 1) / jnp.sqrt(d)  # (H, S, S)

        if mask is not None:
            logits = jnp.where(mask == 0, -9e15, logits)

        attn = jax.nn.softmax(logits, axis=-1)  # (H, S, S)
        vals = attn @ v  # (H, S, D/H)

        return vals


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
        causal: bool = False,
        mlp_ratio: float = 4.0,
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
        attn_mask: Optional[Array] = None,
        *,
        key: Optional[Key[Array, ""]] = None,
        train: bool = True,
    ) -> Float[Array, "S D"]:
        if key is not None:
            k1, k2 = jax.random.split(key)
        else:
            k1 = k2 = None

        attn_out = self.attn(jax.vmap(self.ln1)(x), mask=attn_mask)
        x = x + self.dropout(attn_out, key=k1, inference=not train)

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
        causal: bool = False,
        mlp_ratio: float = 4.0,
        p_drop: float = 0.1,
        seq_len: int = 2048,
        pe_type: str = "1d",
        grid_size: Optional[int] = None,
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
        if pe_type == "1d":
            self.pe = PositionalEncoding(dim=dim, seq_len=seq_len)
        elif pe_type == "2d":
            assert grid_size is not None, (
                "Please specify static grid size for 2D encoding"
            )
            self.pe = PositionalEncoding2D(
                grid_size=grid_size, dim=dim, seq_len=seq_len
            )

    def __call__(
        self,
        x: Float[Array, "S D"],
        attn_mask: Optional[Array] = None,
        *,
        key: Optional[Key[Array, ""]] = None,
        use_pe: bool = True,
        train: bool = True,
        get_intermediates: bool = False,
    ) -> Float[Array, "S D"] | tuple[Float[Array, "S D"], list[Float[Array, "S D"]]]:
        x = self.pe(x) if use_pe else x

        if key is not None:
            keys = jax.random.split(key, len(self.blocks))
        else:
            keys = [None] * len(self.blocks)

        intermediates = [x] if get_intermediates else None

        for block, k in zip(self.blocks, keys):
            x = block(x, key=k, train=train, attn_mask=attn_mask)

            if intermediates is not None:
                intermediates += [x]

        if intermediates is not None:
            return x, intermediates

        return x
