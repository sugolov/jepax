import jax
import jax.numpy as jnp
from jepax.model.vit import PatchEmbedding, ViTclassifier

def test_patch_embedding():
    key = jax.random.PRNGKey(0)
    embed = PatchEmbedding(input_channels=3, output_shape=64, patch_size=4, key=key)
    x = jax.random.normal(key, (32, 32, 3))
    out = embed(x)
    assert out.shape == (64, 64)
    print("✓ PatchEmbedding")


def test_vit_classifier():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    
    model = ViTclassifier(
        num_channels=3,
        patch_size=4,
        num_classes=10,
        dim=64,
        num_layers=2,
        num_head=4,
        key=k1
    )
    
    x = jax.random.normal(k2, (32, 32, 3))
    logits = model(x, key=k3, train=True)
    assert logits.shape == (10,)
    print("✓ ViTclassifier")


def test_vit_batched():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    
    model = ViTclassifier(
        num_channels=3,
        patch_size=4,
        num_classes=10,
        dim=64,
        num_layers=2,
        num_head=4,
        key=k1
    )
    
    x_batch = jax.random.normal(k2, (8, 32, 32, 3))
    keys = jax.random.split(k3, 8)
    logits = jax.vmap(lambda x, k: model(x, key=k, train=True))(x_batch, keys)
    assert logits.shape == (8, 10)
    print("✓ ViTclassifier batched")