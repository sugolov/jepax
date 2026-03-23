from functools import partial
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

    grid: Array = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    def __init__(self, grid_size: int, dim: int, seq_len: int = 5000):
        super().__init__(dim // 2, seq_len=seq_len)

        self.dim = dim
        self.grid = self._get_pe_grid(grid_size)

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
    implementation: Optional[str] = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_head: int,
        causal: bool = False,
        implementation: Optional[str] = None,
        *,
        key: Key[Array, ""],
    ):
        k1, k2 = jax.random.split(key)
        self.dim = dim
        self.num_head = num_head
        self.causal = causal
        self.implementation = implementation
        self.qkv_proj = eqx.nn.Linear(dim, 3 * dim, key=k1)
        self.out_proj = eqx.nn.Linear(dim, dim, key=k2)

    def __call__(
        self, x: Float[Array, "S D"], mask: Optional[Array] = None
    ) -> Float[Array, "S D"]:
        S, D = x.shape

        qkv = jax.vmap(self.qkv_proj)(x)  # (S, 3D)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each (S, D)
        q = q.reshape(S, self.num_head, -1)  # (S, N, Dh)
        k = k.reshape(S, self.num_head, -1)  # (S, N, Dh)
        v = v.reshape(S, self.num_head, -1)  # (S, N, Dh)

        if mask is not None and self.implementation == "cudnn":
            mask = jnp.broadcast_to(mask[None, :, :], (self.num_head, S, S))
        vals = jax.nn.dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            is_causal=self.causal,
            **({"implementation": self.implementation} if self.implementation else {}),
        )  # (S, N, Dh)
        vals = vals.reshape(S, -1)  # (S, D)

        out = jax.vmap(self.out_proj)(vals)  # (S, D)

        return out


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
        attn_implementation: Optional[str] = None,
        *,
        key: Key[Array, ""],
    ):
        k1, k2 = jax.random.split(key)
        self.attn = Attention(
            dim=dim,
            num_head=num_head,
            causal=causal,
            implementation=attn_implementation,
            key=k1,
        )
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
    gradient_checkpointing: bool = eqx.field(static=True)

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
        gradient_checkpointing: bool = False,
        attn_implementation: Optional[str] = None,
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
                attn_implementation=attn_implementation,
                key=k,
            )
            for k in keys
        ]
        if pe_type == "1d":
            self.pe = PositionalEncoding(dim=dim, seq_len=seq_len)
        elif pe_type == "2d":
            assert (
                grid_size is not None
            ), "Please specify static grid size for 2D encoding"
            self.pe = PositionalEncoding2D(
                grid_size=grid_size, dim=dim, seq_len=seq_len
            )
        self.gradient_checkpointing = gradient_checkpointing

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

        if self.gradient_checkpointing and not get_intermediates:
            return self._forward_scan(x, key, train, attn_mask)

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

    def _forward_scan(self, x, key, train, attn_mask):
        n = len(self.blocks)

        dynamics, static_template = [], None
        for b in self.blocks:
            dyn, static = eqx.partition(b, eqx.is_array)
            dynamics.append(dyn)
            static_template = static

        stacked_dyn = jax.tree.map(lambda *xs: jnp.stack(xs), *dynamics)

        if key is not None:
            scan_keys = jax.random.split(key, n)
        else:
            scan_keys = jnp.zeros((n, 2), dtype=jnp.uint32)

        @partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
        def body(carry, inputs):
            block_dyn, k = inputs
            block = eqx.combine(block_dyn, static_template)
            return block(carry, key=k, train=train, attn_mask=attn_mask), None

        x, _ = jax.lax.scan(body, x, (stacked_dyn, scan_keys))
        return x
