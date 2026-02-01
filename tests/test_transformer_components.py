import jax
import jax.numpy as jnp
from jepax.model.transformer import FeedForward, Attention, TransformerBlock, Transformer


def test_feedforward():
    key = jax.random.PRNGKey(0)
    ff = FeedForward(dim=64, mlp_ratio=4.0, key=key)
    x = jax.random.normal(key, (10, 64))
    out = ff(x)
    assert out.shape == (10, 64)
    print("✓ FeedForward")


def test_attention():
    key = jax.random.PRNGKey(0)
    attn = Attention(dim=64, num_head=4, key=key)
    x = jax.random.normal(key, (10, 64))
    out, weights = attn(x)
    assert out.shape == (10, 64)
    assert weights.shape == (4, 10, 10)
    print("✓ Attention")


def test_transformer_block():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    block = TransformerBlock(dim=64, num_head=4, key=k1)
    x = jax.random.normal(k2, (10, 64))
    out = block(x, key=k2, train=True)
    assert out.shape == (10, 64)
    print("✓ TransformerBlock")


def test_transformer():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    model = Transformer(dim=64, num_layers=2, num_head=4, key=k1)
    x = jax.random.normal(k2, (10, 64))
    out = model(x, key=k3, train=True)
    assert out.shape == (10, 64)
    print("✓ Transformer")


def test_batched():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    model = Transformer(dim=64, num_layers=2, num_head=4, key=k1)
    
    x_batch = jax.random.normal(k2, (4, 10, 64))
    keys = jax.random.split(k3, 4)
    out = jax.vmap(lambda x, k: model(x, key=k, train=True))(x_batch, keys)
    assert out.shape == (4, 10, 64)
    print("✓ Batched forward")


if __name__ == "__main__":
    test_feedforward()
    test_attention()
    test_transformer_block()
    test_transformer()
    test_batched()
    print("\nAll tests passed!")