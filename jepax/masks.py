# adapted from https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py

import math
import numpy as np


class MaskCollator:
    """
    Generates context and target masks for I-JEPA training.

    Context is a contiguous block with target regions carved out (matching original I-JEPA).
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        enc_mask_scale: tuple = None,
        pred_mask_scale: tuple = None,
        pred_aspect_ratio: tuple = (0.75, 1.5),
        n_enc_masks: int = 1,
        n_pred_masks: int = None,
        min_keep: int = 4,
        allow_overlap: bool = False,
    ):
        self.h = img_size // patch_size
        self.w = img_size // patch_size
        self.n_patches = self.h * self.w

        # Auto-adjust scales for small grids
        if self.n_patches <= 64:
            self.enc_mask_scale = enc_mask_scale or (0.5, 0.7)
            self.pred_mask_scale = pred_mask_scale or (0.15, 0.2)
            self.n_pred_masks = n_pred_masks if n_pred_masks is not None else 2
        else:
            self.enc_mask_scale = enc_mask_scale or (0.85, 1.0)
            self.pred_mask_scale = pred_mask_scale or (0.15, 0.2)
            self.n_pred_masks = n_pred_masks if n_pred_masks is not None else 4

        self.pred_aspect_ratio = pred_aspect_ratio
        self.n_enc_masks = n_enc_masks
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap

    def _sample_block_size(self, rng, scale, aspect_ratio):
        """Sample block height and width in patches."""
        rand_val = rng.random()
        mask_scale = scale[0] + rand_val * (scale[1] - scale[0])
        max_patches = int(self.h * self.w * mask_scale)

        min_ar, max_ar = aspect_ratio
        log_ar = math.log(min_ar) + rand_val * (math.log(max_ar) - math.log(min_ar))
        ar = math.exp(log_ar)

        bh = int(round(math.sqrt(max_patches * ar)))
        bw = int(round(math.sqrt(max_patches / ar)))

        # Clamp to grid size
        bh = max(1, min(bh, self.h - 1))
        bw = max(1, min(bw, self.w - 1))

        return bh, bw

    def _sample_block_mask(self, rng, bh, bw, acceptable_regions=None):
        """
        Sample a contiguous block, optionally constrained by acceptable_regions.

        Returns (sorted_indices, complement_2d).
        """
        max_tries = 200

        for attempt in range(max_tries):
            top = rng.integers(0, max(1, self.h - bh + 1))
            left = rng.integers(0, max(1, self.w - bw + 1))

            mask_2d = np.zeros((self.h, self.w), dtype=np.int32)
            mask_2d[top : top + bh, left : left + bw] = 1

            if acceptable_regions is not None:
                for region in acceptable_regions:
                    mask_2d = mask_2d * region

            idx = np.flatnonzero(mask_2d).astype(np.int32)

            if len(idx) >= self.min_keep:
                complement = np.ones((self.h, self.w), dtype=np.int32)
                complement[top : top + bh, left : left + bw] = 0
                return idx, complement

        raise RuntimeError(
            f"Could not find valid mask after {max_tries} tries. "
            f"Grid: {self.h}x{self.w}, block: {bh}x{bw}, min_keep: {self.min_keep}. "
            f"Consider using --resize to upscale images to 224x224."
        )

    def __call__(self, rng, batch_size: int = 1):
        """
        Generate context and target masks for a batch.

        Returns:
            enc_masks: [B, n_enc, L_enc] - context indices (sorted)
            pred_masks: [B, n_pred, L_pred] - target indices (sorted)
        """
        pred_bh, pred_bw = self._sample_block_size(
            rng, self.pred_mask_scale, self.pred_aspect_ratio
        )
        enc_bh, enc_bw = self._sample_block_size(rng, self.enc_mask_scale, (1.0, 1.0))

        all_enc_masks = []
        all_pred_masks = []
        min_enc_len = self.n_patches
        min_pred_len = self.n_patches

        for _ in range(batch_size):
            # 1. Sample target blocks and their complements
            pred_masks_b = []
            pred_complements = []

            for _ in range(self.n_pred_masks):
                idx, complement = self._sample_block_mask(rng, pred_bh, pred_bw)
                pred_masks_b.append(idx)
                pred_complements.append(complement)
                min_pred_len = min(min_pred_len, len(idx))

            all_pred_masks.append(pred_masks_b)

            # 2. Sample encoder block, constrained to avoid targets
            acceptable_regions = None if self.allow_overlap else pred_complements

            enc_masks_b = []
            for _ in range(self.n_enc_masks):
                idx, _ = self._sample_block_mask(
                    rng, enc_bh, enc_bw, acceptable_regions=acceptable_regions
                )
                enc_masks_b.append(idx)
                min_enc_len = min(min_enc_len, len(idx))

            all_enc_masks.append(enc_masks_b)

        # Truncate to batch minimum length
        enc_masks = np.array(
            [[m[:min_enc_len] for m in masks_b] for masks_b in all_enc_masks],
            dtype=np.int32,
        )

        pred_masks = np.array(
            [[m[:min_pred_len] for m in masks_b] for masks_b in all_pred_masks],
            dtype=np.int32,
        )

        return enc_masks, pred_masks
