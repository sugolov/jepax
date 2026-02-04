import einops
import jax
from jax import numpy as jnp


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


def visualize_mask(x, mask, patch_size, n_patch):
    x_masked = jax.vmap(lambda x, m: x * m)(patchify(x, patch_size), mask)
    x_masked_img = unpatchify(x_masked, patch_size, n_patch)
    return x_masked_img


def set_mask(x, mask, mask_vec):
    return x * mask + (1 - mask) * mask_vec


def set_token_mask(tokens, mask, mask_vec):
    return jnp.where(mask[..., None], tokens, mask_vec)


def mask_to_indices(mask: jnp.ndarray, max_len: int = None) -> tuple:
    """Convert boolean mask to indices array.

    Args:
        mask: boolean mask [N] where True = keep
        max_len: pad indices to this length (default: mask length)

    Returns:
        indices: indices of True positions, padded with 0s to max_len
        n_keep: number of True positions
    """
    if max_len is None:
        max_len = mask.shape[0]
    indices = jnp.where(mask, size=max_len, fill_value=0)[0]
    n_keep = jnp.sum(mask)
    return indices, n_keep


class IJEPAMasker:
    def __init__(
        self,
        height,
        width,
        patch_size,
        ctx_scale=(0.8, 1.0),
        ctx_aspect=1.0,
        pred_scale=(0.15, 0.2),
        pred_aspect=(0.75, 1.5),
        min_keep=10,
    ):
        self.h = height
        self.w = width
        self.ps = patch_size

        self.h_patch = self.h // self.ps
        self.w_patch = self.w // self.ps
        self.n_patches = self.h_patch * self.w_patch

        self.ctx_scale = self._create_interval(ctx_scale)
        self.ctx_aspect = self._create_interval(ctx_aspect)
        self.pred_scale = self._create_interval(pred_scale)
        self.pred_aspect = self._create_interval(pred_aspect)
        self.min_keep = min_keep

    def _get_block_size(self, scale, aspect):
        n_keep = self.n_patches * scale
        h_block = jnp.round(jnp.sqrt(n_keep * aspect)).astype(jnp.int32)
        w_block = jnp.round(jnp.sqrt(n_keep / aspect)).astype(jnp.int32)
        h_block = jnp.minimum(h_block, self.h_patch - 1)
        w_block = jnp.minimum(w_block, self.w_patch - 1)
        return h_block, w_block

    def _create_interval(self, x):
        return x if isinstance(x, tuple) else (x, x)

    def _sample(self, key, interval):
        return jax.random.uniform(key, minval=interval[0], maxval=interval[1])

    def _sample_block_mask(self, key, h_block, w_block, flatten=False):
        k1, k2 = jax.random.split(key, 2)
        max_top = (self.h_patch - h_block + 1).astype(jnp.float32)
        max_left = (self.w_patch - w_block + 1).astype(jnp.float32)
        top = jnp.floor(jax.random.uniform(k1) * max_top).astype(jnp.int32)
        left = jnp.floor(jax.random.uniform(k2) * max_left).astype(jnp.int32)

        ii, jj = jnp.meshgrid(
            jnp.arange(self.h_patch), jnp.arange(self.w_patch), indexing="ij"
        )
        mask = (ii >= top) & (ii < top + h_block) & (jj >= left) & (jj < left + w_block)
        return mask.flatten() if flatten else mask

    def __call__(self, key, M, flatten=False):
        keys = jax.random.split(key, M + 3)
        k_scales, k_ctx, pred_keys = keys[0], keys[1], keys[2:]

        k1, k2, k3, k4 = jax.random.split(k_scales, 4)
        ctx_scale = self._sample(k1, self.ctx_scale)
        ctx_aspect = self._sample(k2, self.ctx_aspect)
        pred_scale = self._sample(k3, self.pred_scale)
        pred_aspect = self._sample(k4, self.pred_aspect)

        pred_h, pred_w = self._get_block_size(pred_scale, pred_aspect)
        pred_mask = jax.vmap(
            lambda k: self._sample_block_mask(k, pred_h, pred_w, flatten)
        )(pred_keys)

        ctx_h, ctx_w = self._get_block_size(ctx_scale, ctx_aspect)
        ctx_mask = self._sample_block_mask(k_ctx, ctx_h, ctx_w, flatten)

        return ctx_mask, pred_mask
