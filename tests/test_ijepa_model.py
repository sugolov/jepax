"""Tests for IJEPA index operations.

Runnable with: pytest test_ijepa_model.py -v

Organized bottom-up: pure helpers → predictor index math → encoder → full IJEPA.
If something breaks, the first failing test in this order tells you which layer
to debug.

Note: PatchEmbedding expects channels-first input [C, H, W], despite the encoder
docstring saying [H, W, C].
"""

import jax
import jax.numpy as jnp

from jepax.model.ijepa import (
    _sincos_embed,
    compute_2d_pe,
    get_ijepa_model,
    IJEPAEncoder,
    mask_to_indices,
)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

# Image shape used by all encoder/model tests: 3 channels, 16x16 pixels,
# patch_size=4 → 4x4 patch grid → 16 patches total.
IMG_SHAPE = (3, 16, 16)


def _bool_mask(n, true_indices):
    """Build a [n] boolean mask with True at the given indices."""
    return jnp.zeros(n, dtype=bool).at[jnp.array(true_indices)].set(True)


def _make_encoder():
    """Small encoder for fast tests: 4x4 patch grid, 32-dim, 1 layer."""
    return IJEPAEncoder(
        num_channels=3,
        patch_size=4,
        img_size=16,
        dim=32,
        num_layers=1,
        num_head=2,
        mlp_ratio=2.0,
        p_drop=0.0,
        seq_len=16,
        key=jax.random.key(0),
    )


def _make_model():
    """Small full IJEPA model using the 'ijepa-test' config."""
    model, _ = get_ijepa_model(
        "ijepa-test",
        key=jax.random.key(0),
        num_channels=3,
        patch_size=4,
        img_size=16,
        p_drop=0.0,
        seq_len=16,
    )
    return model


def _assemble(ctx_proj, pred_tokens, n_ctx, n_tgt, seq_len):
    """Mirror of IJEPAPredictor's masked assembly. Lets us test the math directly."""
    n_total = n_ctx + n_tgt
    ctx_mask = jnp.arange(seq_len) < n_ctx
    tgt_mask = (jnp.arange(seq_len) >= n_ctx) & (jnp.arange(seq_len) < n_total)
    pred_idx = jnp.clip(jnp.arange(seq_len) - n_ctx, 0, seq_len - 1)
    return jnp.where(ctx_mask[:, None], ctx_proj, 0.0) + jnp.where(
        tgt_mask[:, None], pred_tokens[pred_idx], 0.0
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_sincos_embed():
    """_sincos_embed returns shape [N, dim] and gives"""
    """sin=0, cos=1 at position 0."""
    out = _sincos_embed(jnp.arange(10), dim=16)
    assert out.shape == (10, 16)
    pos0 = _sincos_embed(jnp.array([0]), dim=8)
    assert jnp.allclose(pos0[0, 0::2], 0.0)  # interleaved sin slots
    assert jnp.allclose(pos0[0, 1::2], 1.0)  # interleaved cos slots


def test_compute_2d_pe_decomposition():
    """2D PE for index k splits into [col PE | row PE]"""
    """where col=k%G, row=k//G."""
    out = compute_2d_pe(jnp.array([9]), grid_size=4, dim=32)  # 9 → row=2, col=1
    col_pe = _sincos_embed(jnp.array([1]), dim=16)
    row_pe = _sincos_embed(jnp.array([2]), dim=16)
    expected = jnp.concatenate([col_pe, row_pe], axis=-1)
    assert jnp.allclose(out, expected)


def test_compute_2d_pe_padding():
    """Padded indices (fill_value=0) get the same PE as patch 0 — must be"""
    """masked downstream."""
    padded = compute_2d_pe(jnp.array([5, 0, 0]), grid_size=4, dim=32)
    pe_zero = compute_2d_pe(jnp.array([0]), grid_size=4, dim=32)
    assert jnp.allclose(padded[1], pe_zero[0])
    assert jnp.allclose(padded[2], pe_zero[0])


def test_mask_to_indices_basic():
    """mask_to_indices packs True positions to the front,"""
    """pads remainder with 0."""
    m = _bool_mask(8, [1, 3, 4])
    idx, n = mask_to_indices(m, max_len=8)
    assert int(n) == 3
    assert [int(idx[i]) for i in range(3)] == [1, 3, 4]
    assert all(int(idx[i]) == 0 for i in range(3, 8))


def test_mask_to_indices_edge_cases():
    """All-True returns identity arange; all-False returns n_keep=0."""
    idx, n = mask_to_indices(jnp.ones(8, dtype=bool), 8)
    assert int(n) == 8 and jnp.array_equal(idx, jnp.arange(8))

    idx, n = mask_to_indices(jnp.zeros(8, dtype=bool), 8)
    assert int(n) == 0 and jnp.all(idx == 0)


# ---------------------------------------------------------------------------
# Predictor index logic in isolation
# ---------------------------------------------------------------------------


def test_pred_idx_clip():
    """pred_idx = clip(arange - n_ctx, 0, seq_len-1)"""
    """shifts pred_tokens to start at n_ctx."""
    seq_len, n_ctx = 8, 3
    pred_idx = jnp.clip(jnp.arange(seq_len) - n_ctx, 0, seq_len - 1)
    assert jnp.array_equal(pred_idx, jnp.array([0, 0, 0, 0, 1, 2, 3, 4]))


def test_predictor_layout():
    """combined array layout is [ctx_proj[:n_ctx] | pred_tokens[:n_tgt] | zeros]."""
    seq_len, D = 8, 4
    n_ctx, n_tgt = 3, 2
    ctx_proj = jnp.arange(seq_len * D).reshape(seq_len, D).astype(jnp.float32)
    pred_tokens = -1.0 - jnp.arange(seq_len * D).reshape(seq_len, D).astype(jnp.float32)

    combined = _assemble(ctx_proj, pred_tokens, n_ctx, n_tgt, seq_len)
    assert jnp.array_equal(combined[0:n_ctx], ctx_proj[0:n_ctx])
    assert jnp.array_equal(combined[n_ctx : n_ctx + n_tgt], pred_tokens[0:n_tgt])
    assert jnp.all(combined[n_ctx + n_tgt :] == 0)


def test_predictor_padding_no_leak():
    """Garbage values in padded slots of ctx_proj and pred_tokens"""
    """must not appear in combined."""
    seq_len, D = 8, 4
    n_ctx, n_tgt = 2, 2
    ctx_proj = jnp.ones((seq_len, D)).at[n_ctx:].set(1e9)
    pred_tokens = (2 * jnp.ones((seq_len, D))).at[n_tgt:].set(-1e9)

    combined = _assemble(ctx_proj, pred_tokens, n_ctx, n_tgt, seq_len)
    assert jnp.all(jnp.abs(combined) < 1e6)


def test_predictor_no_targets():
    """With n_tgt=0, combined is pure context followed by zeros"""
    """(pred_tokens ignored)."""
    seq_len, D = 8, 4
    combined = _assemble(
        jnp.ones((seq_len, D)),
        jnp.full((seq_len, D), 999.0),
        n_ctx=4,
        n_tgt=0,
        seq_len=seq_len,
    )
    assert jnp.array_equal(combined[:4], jnp.ones((4, D)))
    assert jnp.all(combined[4:] == 0)


def test_predictor_no_context():
    """With n_ctx=0, combined starts directly with pred_tokens (ctx_proj ignored)."""
    seq_len, D = 8, 4
    combined = _assemble(
        jnp.full((seq_len, D), 999.0),
        jnp.ones((seq_len, D)),
        n_ctx=0,
        n_tgt=3,
        seq_len=seq_len,
    )
    assert jnp.array_equal(combined[:3], jnp.ones((3, D)))
    assert jnp.all(combined[3:] == 0)


def test_predictor_full_packing():
    """When n_ctx + n_tgt == seq_len there's no zero padding region."""
    seq_len, D = 8, 4
    combined = _assemble(
        jnp.ones((seq_len, D)),
        2 * jnp.ones((seq_len, D)),
        n_ctx=5,
        n_tgt=3,
        seq_len=seq_len,
    )
    assert jnp.array_equal(combined[:5], jnp.ones((5, D)))
    assert jnp.array_equal(combined[5:], 2 * jnp.ones((3, D)))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def test_encoder_unmasked():
    """With mask=None, encoder returns all patches"""
    """in identity order [0, 1, ..., n_patches)."""
    enc = _make_encoder()
    x = jax.random.normal(jax.random.key(1), IMG_SHAPE)
    out, indices = enc(None, x, mask=None, train=False)
    assert out.shape == (16, 32)
    assert jnp.array_equal(indices, jnp.arange(16))


def test_encoder_indices_match_mask_packing():
    """Returned indices match mask_to_indices output:"""
    """kept patches in ascending order."""
    enc = _make_encoder()
    x = jax.random.normal(jax.random.key(1), IMG_SHAPE)
    keep = [2, 5, 9, 11]
    mask = _bool_mask(16, keep)
    _, indices = enc(None, x, mask=mask, train=False)
    assert [int(indices[i]) for i in range(len(keep))] == keep


def test_encoder_isolation():
    """Two images matching only in kept-patch region must give identical valid outputs.

    Verifies the attention mask + zero-fill actually isolate context tokens from
    padded tokens — i.e., the encoder truly sees only the masked-in patches.
    """
    enc = _make_encoder()
    x1 = jax.random.normal(jax.random.key(1), IMG_SHAPE)
    x2 = jax.random.normal(jax.random.key(2), IMG_SHAPE)
    # Patches 0,1,4,5 in 4x4 grid = top-left 8x8 pixel region (channels-first)
    keep = [0, 1, 4, 5]
    mask = _bool_mask(16, keep)
    x2 = x2.at[:, :8, :8].set(x1[:, :8, :8])

    out1, _ = enc(None, x1, mask=mask, train=False)
    out2, _ = enc(None, x2, mask=mask, train=False)
    n = len(keep)
    assert jnp.allclose(out1[:n], out2[:n], atol=1e-4)


# ---------------------------------------------------------------------------
# Full IJEPA composition
# ---------------------------------------------------------------------------


def test_ijepa_shapes():
    """Forward returns z_pred of shape [seq_len, enc_dim],"""
    """plus tgt_indices and n_tgt."""
    model = _make_model()
    x = jax.random.normal(jax.random.key(1), IMG_SHAPE)
    mask_ctx = _bool_mask(16, list(range(10)))
    mask_pred = jnp.zeros((2, 16), dtype=bool)
    mask_pred = mask_pred.at[0, jnp.array([12, 13])].set(True)
    mask_pred = mask_pred.at[1, jnp.array([14, 15])].set(True)

    z_pred, tgt_indices, n_tgt = model(None, x, mask_ctx, mask_pred, train=False)
    assert z_pred.shape == (16, 64)  # ijepa-test encoder dim
    assert tgt_indices.shape == (16,)
    assert int(n_tgt) == 4


def test_target_indices_union():
    """tgt_indices is the union of all pred masks via jnp.any, in ascending order."""
    model = _make_model()
    x = jax.random.normal(jax.random.key(1), IMG_SHAPE)
    mask_ctx = jnp.ones(16, dtype=bool)
    mask_pred = jnp.zeros((2, 16), dtype=bool)
    mask_pred = mask_pred.at[0, jnp.array([3, 11])].set(True)
    mask_pred = mask_pred.at[1, jnp.array([7, 12])].set(True)

    _, tgt_indices, n_tgt = model(None, x, mask_ctx, mask_pred, train=False)
    assert int(n_tgt) == 4
    assert sorted(int(tgt_indices[i]) for i in range(4)) == [3, 7, 11, 12]


def test_overlapping_pred_masks_dedup():
    """Patches present in multiple pred masks are counted once via jnp.any."""
    model = _make_model()
    x = jax.random.normal(jax.random.key(1), IMG_SHAPE)
    mask_ctx = jnp.ones(16, dtype=bool)
    mask_pred = jnp.zeros((2, 16), dtype=bool)
    mask_pred = mask_pred.at[0, jnp.array([5, 6])].set(True)
    mask_pred = mask_pred.at[1, jnp.array([6, 7])].set(True)  # 6 shared

    _, tgt_indices, n_tgt = model(None, x, mask_ctx, mask_pred, train=False)
    assert int(n_tgt) == 3
    assert sorted(int(tgt_indices[i]) for i in range(3)) == [5, 6, 7]


def test_encoder_excludes_targets():
    """mask_enc = mask_ctx & ~mask_tgt: targets present in mask_ctx are removed
    before encoding.

    This is the key invariant preventing the predictor from trivially copying targets
    it shouldn't see. Two configs producing the same effective mask_enc must yield
    identical predictions.
    """
    model = _make_model()
    x = jax.random.normal(jax.random.key(1), IMG_SHAPE)
    mp = jnp.zeros((1, 16), dtype=bool).at[0, jnp.array([4, 5])].set(True)

    mc_a = _bool_mask(16, list(range(8)))  # {0..7}, includes targets
    mc_b = _bool_mask(16, [0, 1, 2, 3, 6, 7])  # already excludes targets

    z_a, ti_a, n_a = model(None, x, mc_a, mp, train=False)
    z_b, ti_b, n_b = model(None, x, mc_b, mp, train=False)

    assert int(n_a) == int(n_b) == 2
    assert jnp.array_equal(ti_a[: int(n_a)], ti_b[: int(n_b)])
    assert jnp.allclose(z_a[: int(n_a)], z_b[: int(n_b)], atol=1e-4)


def test_roll_alignment():
    """After roll(-n_ctx), predictions for target patches live at z_pred[0:n_tgt].

    Same context with different target sets must produce different predictions in
    those slots — verifies target PE is being used and the roll is applied.
    """
    model = _make_model()
    x = jax.random.normal(jax.random.key(1), IMG_SHAPE)
    mask_ctx = jnp.ones(16, dtype=bool)
    mp1 = jnp.zeros((1, 16), dtype=bool).at[0, jnp.array([3, 7])].set(True)
    mp2 = jnp.zeros((1, 16), dtype=bool).at[0, jnp.array([10, 12])].set(True)

    z1, _, n1 = model(None, x, mask_ctx, mp1, train=False)
    z2, _, n2 = model(None, x, mask_ctx, mp2, train=False)
    assert int(n1) == int(n2) == 2
    assert not jnp.allclose(z1[:2], z2[:2])


def test_jit_forward():
    """Full IJEPA forward compiles and runs under jax.jit without shape errors."""
    model = _make_model()

    @jax.jit
    def forward(x, mc, mp):
        return model(None, x, mc, mp, train=False)

    x = jax.random.normal(jax.random.key(1), IMG_SHAPE)
    mc = _bool_mask(16, list(range(10)))
    mp = jnp.zeros((1, 16), dtype=bool).at[0, jnp.array([12, 13])].set(True)
    z, _, n = forward(x, mc, mp)
    assert z.shape == (16, 64)
    assert int(n) == 2
