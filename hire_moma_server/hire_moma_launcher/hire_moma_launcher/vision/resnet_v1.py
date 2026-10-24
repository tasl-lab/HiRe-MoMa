"""Frozen ImageNet ResNet-10 + trainable spatial-embedding head (from HIL-SERL).

Module names are kept identical to HIL-SERL so that `resnet10_params.pkl` loads as is.
"""
import functools as ft
from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from hire_moma_launcher.vision.data_augmentations import resize

ModuleDef = Any


class SpatialLearnedEmbeddings(nn.Module):
    height: int
    width: int
    channel: int
    num_features: int = 5
    kernel_init: Callable = nn.initializers.lecun_normal()
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, features):
        """features: (B, H, W, C) or (H, W, C)."""
        kernel = self.param("kernel", self.kernel_init,
                            (self.height, self.width, self.channel, self.num_features), self.param_dtype)
        no_batch_dim = len(features.shape) < 4
        if no_batch_dim:
            features = features[None]
        batch_size = features.shape[0]
        features = jnp.sum(jnp.expand_dims(features, -1) * jnp.expand_dims(kernel, 0), axis=(1, 2))
        features = jnp.reshape(features, [batch_size, -1])
        if no_batch_dim:
            features = features[0]
        return features


class MyGroupNorm(nn.GroupNorm):
    def __call__(self, x):
        if x.ndim == 3:
            return super().__call__(x[jnp.newaxis])[0]
        return super().__call__(x)


class ResNetBlock(nn.Module):
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm()(y)
        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(residual)
            residual = self.norm(name="norm_proj")(residual)
        return self.act(residual + y)


class ResNetEncoder(nn.Module):
    """ResNet-v1 trunk. Returns the (frozen) feature map before pooling."""

    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: str = "relu"
    image_size: tuple = (128, 128)

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = True):
        if observations.shape[-3:-1] != self.image_size:
            observations = resize(observations, self.image_size)
        mean = jnp.array([0.485, 0.456, 0.406])
        std = jnp.array([0.229, 0.224, 0.225])
        x = (observations.astype(jnp.float32) / 255.0 - mean) / std

        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype, kernel_init=nn.initializers.kaiming_normal())
        norm = partial(MyGroupNorm, num_groups=4, epsilon=1e-5, dtype=self.dtype)
        act = getattr(nn, self.act)

        x = conv(self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name="conv_init")(x)
        x = norm(name="norm_init")(x)
        x = act(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                stride = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2**i, strides=stride, conv=conv, norm=norm, act=act)(x)
        return jax.lax.stop_gradient(x)


class PreTrainedResNetEncoder(nn.Module):
    """Frozen trunk + trainable spatial embeddings + bottleneck."""

    num_spatial_blocks: int = 8
    bottleneck_dim: Optional[int] = None
    pretrained_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations: jnp.ndarray, encode: bool = True, train: bool = True):
        x = self.pretrained_encoder(observations, train=train) if encode else observations
        height, width, channel = x.shape[-3:]
        x = SpatialLearnedEmbeddings(height=height, width=width, channel=channel,
                                     num_features=self.num_spatial_blocks)(x)
        x = nn.Dropout(0.1, deterministic=not train)(x)
        if self.bottleneck_dim is not None:
            x = nn.Dense(self.bottleneck_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
        return x


resnetv1_configs = {
    "resnetv1-10-frozen": ft.partial(ResNetEncoder, stage_sizes=(1, 1, 1, 1), block_cls=ResNetBlock),
}


def make_image_encoders(image_keys):
    """One pretrained ResNet-10 encoder per camera."""
    pretrained = resnetv1_configs["resnetv1-10-frozen"](name="pretrained_encoder")
    return {
        key: PreTrainedResNetEncoder(num_spatial_blocks=8, bottleneck_dim=256,
                                     pretrained_encoder=pretrained, name=f"encoder_{key}")
        for key in image_keys
    }
