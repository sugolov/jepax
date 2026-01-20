import numpy as np
import jax
from jax import numpy as jnp
import equinox as eqx
from typing import Optional
from jaxtyping import Array, PRNGKeyArray

class PositionalEncoding(eqx.Module):
    """Sinusoidal positional encoding"""
    pe: Array
    dim: int  = eqx.field(static=True)
    
    def __init__(self, dim: int, seq_len: int = 5000):
        self.dim = dim
        pe = np.zeros((seq_len, dim))
        position = np.arange(0, seq_len, dtype=np.float32)[:, None]
        div_term = np.exp(np.arange(0, dim, 2) * (-np.log(10000.0) / dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = jax.lax.stop_gradient(np.array(pe))

    def pe_from_idx(self, idx):
        return jax.lax.stop_gradient(jnp.take(self.pe, idx, axis=0))
    
    def __call__(self, x):
        # x: (S, D)
        seq_len = x.shape[0]
        return x + self.pe[:seq_len]
    

    
class PositionalEncoding2D(PositionalEncoding):
    grid: Array
    dim: int = eqx.field(static=True)
    
    def __init__(self, grid_size: int, dim: int, seq_len: int = 5000):
        super().__init__(dim // 2, seq_len=seq_len)

        self.dim = dim
        self.grid = self._get_pe_grid(grid_size)
        
    def _get_pe_grid(self, grid_size):
        grid_h = np.arange(grid_size, dtype=float)
        grid_w = np.arange(grid_size, dtype=float)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0).astype(int)
        return jax.lax.stop_gradient(grid)
    
    def _get_pe_from_grid(self, grid):
        encx = self.pe_from_idx(grid[0].flatten())
        ency = self.pe_from_idx(grid[1].flatten())
        enc = jnp.concatenate([encx, ency], axis=-1) # concatenate halved dims
        return jax.lax.stop_gradient(enc)

    def __call__(self, x):
        """
        Assume x is (grid_size * grid_size, D) tokens
        """
        assert x.shape[0] == self.grid.shape[1]**2

        encx = self.pe_from_idx(self.grid[0].flatten())
        ency = self.pe_from_idx(self.grid[1].flatten())
        enc = jnp.concatenate([encx, ency], axis=-1) # concatenate halved dims

        return x + enc

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
        # x: (S, D)
        x = jax.nn.gelu(jax.vmap(self.linear1)(x))
        x = jax.vmap(self.norm)(x)
        x = jax.nn.gelu(jax.vmap(self.linear2)(x))
        return x


class Attention(eqx.Module):
    """Multihead attention layer"""
    qkv_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    num_head: int
    dim: int
    causal: bool
    
    def __init__(self, dim: int, num_head: int, causal: bool = True, *, 
                 key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.dim = dim
        self.num_head = num_head
        self.causal = causal
        self.qkv_proj = eqx.nn.Linear(dim, 3 * dim, key=k1)
        self.out_proj = eqx.nn.Linear(dim, dim, key=k2)
    
    def __call__(self, x, mask: Optional[Array]=None):
        # x: (S, D)
        S, D = x.shape
        
        qkv = jax.vmap(self.qkv_proj)(x)                # (S, 3D)
        qkv = qkv.reshape(S, self.num_head, -1)         # (S, H, 3D/H)
        qkv = qkv.transpose(1, 0, 2)                    # (H, S, 3D/H)
        q, k, v = jnp.split(qkv, 3, axis=-1)            # (H, S, D/H)
        

        mask_causal = jnp.tri(S, S).T if self.causal else None

        # NOTE: double check this
        if mask is None:
            # if mask is not passed, then mask is whatever mask_causal takes
            mask = mask_causal
        elif self.causal:
            # but if causal, and mask is passed, then we mask out both
            mask = mask.astype(bool) | mask_causal

        
        vals = self._attention(q, k, v, mask=mask)  # (H, S, D/H)
        vals = vals.transpose(1, 0, 2)                    # (S, H, D/H)
        vals = vals.reshape(S, -1)                        # (S, D)
        
        out = jax.vmap(self.out_proj)(vals)               # (S, D)
        
        return out
    
    def _attention(self, q, k, v, mask=None):
        # here, mask is where false
        """Attention mechanism. q, k, v: (H, S, D/H)"""
        d = q.shape[-1]
        logits = q @ k.transpose(0, 2, 1) / jnp.sqrt(d)   # (H, S, S)
        
        if mask is not None:
            logits = jnp.where(mask == 0, -9e15, logits) 

        attn = jax.nn.softmax(logits, axis=-1)            # (H, S, S)
        vals = attn @ v                                   # (H, S, D/H)
        
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
        causal: bool = True,
        mlp_ratio: float = 3.0,
        p_drop: float = 0.1,
        *,
        key: PRNGKeyArray
    ):
        k1, k2 = jax.random.split(key)
        self.attn = Attention(dim=dim, num_head=num_head, causal=causal, key=k1)
        self.ff = FeedForward(dim=dim, mlp_ratio=mlp_ratio, key=k2)
        self.ln1 = eqx.nn.LayerNorm(dim)
        self.ln2 = eqx.nn.LayerNorm(dim)
        self.dropout = eqx.nn.Dropout(p_drop)
    
    def __call__(self, x, attn_mask: Optional[Array] = None, *, 
                 key: Optional[PRNGKeyArray] = None, train: bool = True):
        # x: (S, D)
        if key is not None:
            k1, k2 = jax.random.split(key)
        else:
            k1 = k2 = None
        
        attn_out = self.attn(x, mask=attn_mask)
        x = x + self.dropout(attn_out, key=k1, inference=not train)
        x = jax.vmap(self.ln1)(x)
        
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out, key=k2, inference=not train)
        x = jax.vmap(self.ln2)(x)
        
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
        pe_type: str = "1d",
        grid_size: Optional[int] = None,
        *,
        key: PRNGKeyArray
    ):
        keys = jax.random.split(key, num_layers)
        self.blocks = [
            TransformerBlock(
                dim=dim,
                num_head=num_head,
                causal=causal,
                mlp_ratio=mlp_ratio,
                p_drop=p_drop,
                key=k
            )
            for k in keys
        ]
        if pe_type == "1d":
            self.pe = PositionalEncoding(dim=dim, seq_len=seq_len)
        elif pe_type == "2d":
            assert grid_size is not None, \
                "Please specify static grid size for 2D encoding"
            self.pe = PositionalEncoding2D(
                grid_size=grid_size, 
                dim=dim, 
                seq_len=seq_len
            )

    def __call__(self, x, attn_mask: Optional[Array] = None, *, 
                 key: Optional[PRNGKeyArray] = None, use_pe: bool = True, train: bool = True):
        """
        TODO: stochastic depth

        Args:
            x (_type_): _description_
            key (Optional[PRNGKeyArray], optional): _description_. Defaults to None.
            use_pe (bool, optional): _description_. Defaults to False.
            train (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        # x: (S, D)
        x = self.pe(x) if use_pe else x
        
        if key is not None:
            keys = jax.random.split(key, len(self.blocks))
        else:
            keys = [None] * len(self.blocks)
        
        for block, k in zip(self.blocks, keys):
            x = block(x, key=k, train=train, attn_mask=attn_mask)
        
        return x