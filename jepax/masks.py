# adapted from https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py

import math
import warnings

import numpy as np


class MaskCollator:
    """
    Generates context and target masks for I-JEPA training.

    Returns masks as numpy arrays of patch indices.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        enc_mask_scale: tuple = (0.85, 1.0),
        pred_mask_scale: tuple = (0.15, 0.2),
        pred_aspect_ratio: tuple = (0.75, 1.5),
        n_pred_masks: int = 4,
        min_keep: int = 4,
    ):
        self.h = img_size // patch_size
        self.w = img_size // patch_size
        self.n_patches = self.h * self.w

        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.pred_aspect_ratio = pred_aspect_ratio
        self.n_pred_masks = n_pred_masks
        self.min_keep = min_keep

    def _sample_block_size(self, rng, scale, aspect_ratio):
        """Sample block height and width in patches."""
        rand_val = rng.random()
        mask_scale = scale[0] + rand_val * (scale[1] - scale[0])
        max_patches = int(self.h * self.w * mask_scale)

        # Log-space sampling for aspect ratio
        log_ar = math.log(aspect_ratio[0]) + rand_val * (
            math.log(aspect_ratio[1]) - math.log(aspect_ratio[0])
        )
        ar = math.exp(log_ar)

        bh = int(round(math.sqrt(max_patches * ar)))
        bw = int(round(math.sqrt(max_patches / ar)))
        bh = max(1, min(bh, self.h))
        bw = max(1, min(bw, self.w))

        return bh, bw

    def _sample_block_indices(self, rng, bh, bw):
        """Sample a block and return patch indices."""
        top = rng.integers(0, self.h - bh + 1)
        left = rng.integers(0, self.w - bw + 1)

        indices = []
        for r in range(top, top + bh):
            for c in range(left, left + bw):
                indices.append(r * self.w + c)
        return np.array(indices, dtype=np.int32)

    def __call__(self, rng, batch_size: int = 1):
        """Generate context and target masks for a batch.

        Args:
            rng: numpy random Generator
            batch_size: number of images in batch

        Returns:
            ctx_masks: list of [N_ctx] arrays (one per image, all same length)
            tgt_masks: list of [N_tgt] arrays (one per image, all same length)
            tgt_blocks_list: list of list of arrays (target blocks per image)
        """
        # Sample block sizes once for the entire batch.
        pred_bh, pred_bw = self._sample_block_size(
            rng, self.pred_mask_scale, self.pred_aspect_ratio
        )
        enc_bh, enc_bw = self._sample_block_size(rng, self.enc_mask_scale, (1.0, 1.0))

        ctx_masks = []
        tgt_masks = []
        tgt_blocks_list = []

        # Track minimum sizes across batch for truncation
        min_ctx_len = self.n_patches
        min_tgt_len = self.n_patches

        # First pass: sample masks for each image
        for _ in range(batch_size):
            # Sample target blocks (same size, different locations)
            tgt_blocks = []
            all_tgt_indices = set()

            for _ in range(self.n_pred_masks):
                block_indices = self._sample_block_indices(rng, pred_bh, pred_bw)
                tgt_blocks.append(block_indices)
                all_tgt_indices.update(block_indices.tolist())

            # Sample context block location
            ctx_indices = self._sample_block_indices(rng, enc_bh, enc_bw)

            # Remove target patches from context
            ctx_indices = np.array(
                [i for i in ctx_indices if i not in all_tgt_indices], dtype=np.int32
            )

            if len(ctx_indices) < self.min_keep:
                warnings.warn(
                    f"Context too small ({len(ctx_indices)} < {self.min_keep}), "
                    "using all non-target patches"
                )
                ctx_indices = np.array(
                    [i for i in range(self.n_patches) if i not in all_tgt_indices],
                    dtype=np.int32,
                )

            tgt_indices = np.array(sorted(all_tgt_indices), dtype=np.int32)

            ctx_masks.append(ctx_indices)
            tgt_masks.append(tgt_indices)
            tgt_blocks_list.append(tgt_blocks)

            min_ctx_len = min(min_ctx_len, len(ctx_indices))
            min_tgt_len = min(min_tgt_len, len(tgt_indices))

        # Truncate all masks to minimum length for batching.
        # This allows stacking into tensors of shape [B, N_ctx] and [B, N_tgt].
        ctx_masks = [m[:min_ctx_len] for m in ctx_masks]
        tgt_masks = [m[:min_tgt_len] for m in tgt_masks]

        # For batch_size=1, return single masks
        if batch_size == 1:
            return ctx_masks[0], tgt_masks[0], tgt_blocks_list[0]

        return ctx_masks, tgt_masks, tgt_blocks_list
