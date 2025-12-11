import numpy as np
import jax
from jax import numpy as jnp
import equinox as eqx
from typing import Optional
from jaxtyping import Array, PRNGKeyArray


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


class PositionalEncoding(eqx.Module):
    """Sinusoidal positional encoding"""
    pe: Array = eqx.field(static=True)
    
    def __init__(self, dim: int, seq_len: int = 5000):
        pe = np.zeros((seq_len, dim))
        position = np.arange(0, seq_len, dtype=np.float32)[:, None]
        div_term = np.exp(np.arange(0, dim, 2) * (-np.log(10000.0) / dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = jnp.array(pe)
    
    def __call__(self, x):
        # x: (S, D)
        seq_len = x.shape[0]
        return x + self.pe[:seq_len]


class Attention(eqx.Module):
    """Multihead attention layer"""
    qkv_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    num_head: int
    dim: int
    causal: bool
    
    def __init__(self, dim: int, num_head: int, causal: bool = True, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.dim = dim
        self.num_head = num_head
        self.causal = causal
        self.qkv_proj = eqx.nn.Linear(dim, 3 * dim, key=k1)
        self.out_proj = eqx.nn.Linear(dim, dim, key=k2)
    
    def __call__(self, x):
        # x: (S, D)
        S, D = x.shape
        
        qkv = jax.vmap(self.qkv_proj)(x)                # (S, 3D)
        qkv = qkv.reshape(S, self.num_head, -1)         # (S, H, 3D/H)
        qkv = qkv.transpose(1, 0, 2)                    # (H, S, 3D/H)
        q, k, v = jnp.split(qkv, 3, axis=-1)            # (H, S, D/H)
        
        mask = jnp.tri(S, S).T if self.causal else None
        
        vals, attn = self._attention(q, k, v, mask=mask)  # (H, S, D/H)
        vals = vals.transpose(1, 0, 2)                    # (S, H, D/H)
        vals = vals.reshape(S, -1)                        # (S, D)
        
        out = jax.vmap(self.out_proj)(vals)               # (S, D)
        
        return out, attn
    
    def _attention(self, q, k, v, mask=None):
        """Attention mechanism. q, k, v: (H, S, D/H)"""
        d = q.shape[-1]
        logits = q @ k.transpose(0, 2, 1) / jnp.sqrt(d)   # (H, S, S)
        
        if mask is not None:
            logits = jnp.where(mask == 0, -9e15, logits)
        
        attn = jax.nn.softmax(logits, axis=-1)            # (H, S, S)
        vals = attn @ v                                   # (H, S, D/H)
        
        return vals, attn


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
    
    def __call__(self, x, *, key: Optional[PRNGKeyArray] = None, train: bool = True):
        # x: (S, D)
        if key is not None:
            k1, k2 = jax.random.split(key)
        else:
            k1 = k2 = None
        
        attn_out, _ = self.attn(x)
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
        self.pe = PositionalEncoding(dim=dim, seq_len=seq_len)
    
    def __call__(self, x, *, key: Optional[PRNGKeyArray] = None, train: bool = True):
        # x: (S, D)
        x = self.pe(x)
        
        if key is not None:
            keys = jax.random.split(key, len(self.blocks))
        else:
            keys = [None] * len(self.blocks)
        
        for block, k in zip(self.blocks, keys):
            x = block(x, key=k, train=train)
        
        return x