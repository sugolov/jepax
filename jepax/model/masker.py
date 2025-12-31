import jax
from jax import numpy as jnp

import einops

def patchify(x, patch_size):
    return einops.rearrange(
        x,
        "(h ph) (w pw) c -> (h w) (c ph pw)",
        ph=patch_size,
        pw=patch_size,
    )

def unpatchify(x, patch_size, n_patch):
    return einops.rearrange(
        x,
        "(h w) (c ph pw) -> (h ph) (w pw) c",
        h=n_patch,
        ph=patch_size,
        pw=patch_size,
    )

def set_mask(x, mask, vec): 
    return x * mask + (1 - mask) * vec

def set_token_mask(tokens, mask, vec): 
    return jax.vmap(set_mask, in_axes=(0, 0, None))(tokens, mask, vec)

class IJEPAMasker:
    def __init__(self, height, width, patch_size,
                 ctx_scale=(0.8, 1.0), ctx_aspect=1.0,
                 pred_scale=(0.15, 0.2), pred_aspect=(0.75, 1.5)):
        self.h = height
        self.w = width
        self.ps = patch_size


        self.h_patch = self.h // self.ps
        self.w_patch = self.w // self.ps

        self.ctx_scale = self._create_interval(ctx_scale)
        self.ctx_aspect = self._create_interval(ctx_aspect)
        self.pred_scale = self._create_interval(pred_scale)
        self.pred_aspect = self._create_interval(pred_aspect)

    
    def get_idx_pred_mask(self):
        return self._get_idx_mask(self.pred_aspect[-1], self.pred_scale[-1])
    
    def get_idx_ctx_mask(self):
        return self._get_idx_mask(self.ctx_aspect[-1], self.ctx_scale[-1])

    def _get_idx_mask(self, scale, aspect):
        # (i, j) indices for top left corner
        i_max = jnp.ceil((1 - scale) * self.w // 2) 
        j_max = jnp.ceil((1 - scale / aspect) * self.h // 2) 
        pw_mask = jnp.array(self.w * scale // self.ps, int)
        ph_mask = jnp.array(self.w * scale * aspect // self.ps, int)
        return i_max, j_max, pw_mask, ph_mask
    
    def _create_interval(self, x):
        return x if isinstance(x, tuple) else (x, x)
        
    def _sample(self, key, interval):
        return jax.random.uniform(key, minval=interval[0], maxval=interval[1])
    
    def _sample_aspects(self, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        ctx_scale = self._sample(k1, self.ctx_scale)
        ctx_aspect = self._sample(k2, self.ctx_aspect)
        pred_scale = self._sample(k3, self.pred_scale)
        pred_aspect = self._sample(k4, self.pred_aspect)
        return ctx_scale, ctx_aspect, pred_scale, pred_aspect
    
    def _sample_mask(self, key, i_max, j_max, pw_mask, ph_mask):
        k1, k2 = jax.random.split(key, 2)
        i = jax.random.randint(k1, (), minval=0, maxval=i_max)
        j = jax.random.randint(k2, (), minval=0, maxval=j_max)

        ii, jj = jnp.meshgrid(jnp.arange(self.h_patch), jnp.arange(self.w_patch), indexing='ij')
        mask = (ii >= i) & (ii < i + ph_mask) & (jj >= j) & (jj < j + pw_mask)

        return mask
        
    def __call__(self, key, M):
        """

        Args:
            key (Key): RNG key for mask sampling
            M (int): Number of prediction masks

        Returns:
            Masks of shape (self.h_patch, self.w_patch), (M, self.h_patch, self.w_patch)
        """
        keys = jax.random.split(key, M+2)
        k1, k2, pred_keys = keys[0], keys[1], keys[2:]  # pred_keys is already (M, 2)

        ctx_scale, ctx_aspect, pred_scale, pred_aspect = self._sample_aspects(k1)

        print(
            pred_scale
        )

        ctx_mask = self._sample_mask(k2, *self._get_idx_mask(ctx_scale, ctx_aspect))
        pred_mask = jax.vmap(self._sample_mask, in_axes=(0, None, None, None, None))(
            pred_keys, *self._get_idx_mask(pred_scale, pred_aspect)
        )

        return ctx_mask, pred_mask