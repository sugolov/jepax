import dm_pix
import equinox as eqx
import jax
from jax import numpy as jnp


def solarize(img, threshold=0.5):
    return jnp.where(img < threshold, img, 1.0 - img)


def augment_image(
    rng,
    img,
    color_jitter_prob=0.8,
    grayscale_prob=0.2,
    hflip_prob=0.5,
    solarize_prob=0.0,
):
    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(rng, 8)

    # Horizontal flip
    do_flip = jax.random.uniform(k1) < hflip_prob
    img = jnp.where(do_flip, img[:, ::-1, :], img)

    # Color jitter
    do_jitter = jax.random.uniform(k2) < color_jitter_prob
    jittered = dm_pix.random_brightness(k3, img, max_delta=0.4)
    jittered = dm_pix.random_contrast(k4, jittered, lower=0.6, upper=1.4)
    jittered = dm_pix.random_saturation(k5, jittered, lower=0.8, upper=1.2)
    jittered = dm_pix.random_hue(k6, jittered, max_delta=0.1)
    jittered = jnp.clip(jittered, 0.0, 1.0)
    img = jnp.where(do_jitter, jittered, img)

    # Random grayscale
    do_gray = jax.random.uniform(k7) < grayscale_prob
    gray = jnp.mean(img, axis=-1, keepdims=True)
    gray = jnp.broadcast_to(gray, img.shape)
    img = jnp.where(do_gray, gray, img)

    # Solarization
    do_solar = jax.random.uniform(k8) < solarize_prob
    img = jnp.where(do_solar, solarize(img), img)

    return img


@eqx.filter_jit
def augment_batch(
    rng,
    images,
    color_jitter_prob=0.8,
    grayscale_prob=0.2,
    hflip_prob=0.5,
    solarize_prob=0.0,
):
    keys = jax.random.split(rng, images.shape[0])

    def aug(k, img):
        return augment_image(
            k,
            img,
            color_jitter_prob,
            grayscale_prob,
            hflip_prob,
            solarize_prob,
        )

    return jax.vmap(aug)(keys, images)
