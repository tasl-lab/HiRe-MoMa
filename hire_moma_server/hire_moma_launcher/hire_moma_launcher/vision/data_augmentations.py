from functools import partial

import jax
import jax.numpy as jnp


def random_crop(img, rng, *, padding):
    crop_from = jax.random.randint(rng, (2,), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((1,), dtype=jnp.int32)])
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode="edge")
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=("padding", "num_batch_dims"))
def batched_random_crop(img, rng, *, padding, num_batch_dims: int = 1):
    original_shape = img.shape
    img = jnp.reshape(img, (-1, *img.shape[num_batch_dims:]))
    rngs = jax.random.split(rng, img.shape[0])
    img = jax.vmap(lambda i, r: random_crop(i, r, padding=padding), in_axes=(0, 0), out_axes=0)(img, rngs)
    return jnp.reshape(img, original_shape)


def resize(image, image_dim):
    assert len(image_dim) == 2
    new_shape = list(image.shape)
    new_shape[-3:-1] = image_dim
    return jax.image.resize(image, new_shape, method="bilinear")


def make_batch_augmentation_func(image_keys):
    """Random-shift (pad 4 + crop) augmentation applied to obs and next obs (DrQ style)."""

    def augment_obs(rng, observations):
        for key in image_keys:
            observations = observations.copy(
                add_or_replace={key: batched_random_crop(observations[key], rng, padding=4, num_batch_dims=2)})
        return observations

    def augment_batch(batch, rng):
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        return batch.copy(add_or_replace={
            "observations": augment_obs(obs_rng, batch["observations"]),
            "next_observations": augment_obs(next_obs_rng, batch["next_observations"]),
        })

    return augment_batch
