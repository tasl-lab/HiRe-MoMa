"""Observation encoders.

EncodingWrapper:             images (+ proprio) -> features  (critic input)
BaseResidualEncodingWrapper: + base-head residual          (HiRe-MoMa arm head)
BaseActEncodingWrapper:      + base-policy action (optionally only some dims)  (actor input)
"""
from typing import Dict, Iterable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange


def _flatten_time(x, encoded, unbatched_ndim):
    """Merge the observation-history axis into the feature axis."""
    if len(x.shape) == unbatched_ndim:
        return rearrange(x, "T C -> (T C)"), encoded.reshape(-1)
    if len(x.shape) == unbatched_ndim + 1:
        return rearrange(x, "B T C -> B (T C)"), encoded
    return x, encoded


class EncodingWrapper(nn.Module):
    encoder: Dict[str, nn.Module]
    use_proprio: bool = False
    proprio_latent_dim: int = 64
    enable_stacking: bool = False
    image_keys: Iterable[str] = ("image",)

    @nn.compact
    def __call__(self, observations, train=False, stop_gradient=False, is_encoded=False) -> jnp.ndarray:
        encoded = []
        for image_key in self.image_keys:
            image = observations[image_key]
            if not is_encoded and self.enable_stacking:
                if len(image.shape) == 4:
                    image = rearrange(image, "T H W C -> H W (T C)")
                if len(image.shape) == 5:
                    image = rearrange(image, "B T H W C -> B H W (T C)")
            image = self.encoder[image_key](image, train=train, encode=not is_encoded)
            if stop_gradient:
                image = jax.lax.stop_gradient(image)
            encoded.append(image)
        encoded = jnp.concatenate(encoded, axis=-1)

        if self.use_proprio:
            state = observations["state"]
            if self.enable_stacking:
                state, encoded = _flatten_time(state, encoded, unbatched_ndim=2)
            state = nn.Dense(self.proprio_latent_dim, kernel_init=nn.initializers.xavier_uniform())(state)
            state = nn.tanh(nn.LayerNorm()(state))
            if stop_gradient:
                state = jax.lax.stop_gradient(state)
            encoded = jnp.concatenate([encoded, state], axis=-1)
        return encoded


class _ExtraInputWrapper(nn.Module):
    """Appends an embedding of `observations[key]` (if present) to the wrapped encoding.

    `slices` ([begin, end) pairs) keeps only those last-axis entries of the input.
    The embedding is never stop-gradiented, so the actor loss also trains it.
    """

    encoding_wrapper: nn.Module
    latent_dim: int = 16
    enable_stacking: bool = False
    image_keys: Iterable[str] = ("image",)
    key: str = ""
    slices: Optional[Tuple[Tuple[int, int], ...]] = None

    @nn.compact
    def __call__(self, observations, train=False, stop_gradient=False, is_encoded=False) -> jnp.ndarray:
        encoded = self.encoding_wrapper(observations, train=train, stop_gradient=stop_gradient, is_encoded=is_encoded)
        if self.key in observations:
            x = observations[self.key]
            if self.slices is not None:
                x = jnp.concatenate([x[..., b:e] for b, e in self.slices], axis=-1)
            if self.enable_stacking:
                x, encoded = _flatten_time(x, encoded, unbatched_ndim=2)
            x = nn.Dense(self.latent_dim, kernel_init=nn.initializers.xavier_uniform())(x)
            x = nn.tanh(nn.LayerNorm()(x))
            encoded = jnp.concatenate([encoded, x], axis=-1)
        return encoded


class BaseActEncodingWrapper(_ExtraInputWrapper):
    key: str = "base_actions"


class BaseResidualEncodingWrapper(_ExtraInputWrapper):
    key: str = "base_residual"
