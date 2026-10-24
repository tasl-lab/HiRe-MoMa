from typing import Optional

import distrax
import flax.linen as nn
import jax.numpy as jnp

# Output layers start at zero, so the residual policy starts as "no correction"
# and the critic starts at Q = 0.
default_init = nn.initializers.zeros_init


class Critic(nn.Module):
    """Q(o, a). The encoder is shared with the actor and is trained by the critic loss."""

    encoder: Optional[nn.Module]
    network: nn.Module

    @nn.compact
    def __call__(self, observations, actions: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        obs_enc = observations if self.encoder is None else self.encoder(observations, train=False, stop_gradient=False)
        outputs = self.network(jnp.concatenate([obs_enc, actions], -1), train)
        value = nn.Dense(1, kernel_init=default_init())(outputs)
        return jnp.squeeze(value, -1)


class GraspCritic(nn.Module):
    """Discrete Q(o, .) over gripper commands (used only by the Flat Residual baseline)."""

    encoder: Optional[nn.Module]
    network: nn.Module
    output_dim: int = 2

    @nn.compact
    def __call__(self, observations, train: bool = False) -> jnp.ndarray:
        obs_enc = observations if self.encoder is None else self.encoder(observations, train=False, stop_gradient=False)
        outputs = self.network(obs_enc, train)
        return nn.Dense(self.output_dim, kernel_init=default_init())(outputs)


def ensemblize(cls, num_qs, out_axes=0):
    class EnsembleModule(nn.Module):
        @nn.compact
        def __call__(self, *args, train=False, **kwargs):
            ensemble = nn.vmap(
                cls,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=None,
                out_axes=out_axes,
                axis_size=num_qs,
            )
            return ensemble()(*args, **kwargs)

    return EnsembleModule


class Policy(nn.Module):
    """Gaussian residual policy.

    With `tanh_squash_distribution=True` and `std_clip=eps`, samples lie in [-eps, eps]
    (eps: residual bound, Sec. 4.1).
    """

    encoder: Optional[nn.Module]
    network: nn.Module
    action_dim: int
    std_parameterization: str = "exp"  # "exp", "softplus", "fixed" or "uniform"
    std_min: Optional[float] = 1e-5
    std_max: Optional[float] = 10.0
    tanh_squash_distribution: bool = False
    fixed_std: Optional[jnp.ndarray] = None
    action_scale: float = 1.0  # scales the pre-squash mean
    std_clip: Optional[float] = None  # residual bound

    @nn.compact
    def __call__(self, observations, temperature: float = 1.0, train: bool = False) -> distrax.Distribution:
        # The actor never updates the shared encoder.
        obs_enc = observations if self.encoder is None else self.encoder(observations, train=False, stop_gradient=True)
        outputs = self.network(obs_enc, train=train)
        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs) * self.action_scale

        if self.fixed_std is None:
            if self.std_parameterization == "exp":
                stds = jnp.exp(nn.Dense(self.action_dim, kernel_init=default_init())(outputs))
            elif self.std_parameterization == "softplus":
                stds = nn.softplus(nn.Dense(self.action_dim, kernel_init=default_init())(outputs))
            elif self.std_parameterization == "uniform":
                stds = jnp.exp(self.param("log_stds", nn.initializers.zeros, (self.action_dim,)))
            else:
                raise ValueError(f"Invalid std_parameterization: {self.std_parameterization}")
        else:
            assert self.std_parameterization == "fixed"
            stds = jnp.full((self.action_dim,), jnp.array(self.fixed_std))
        stds = jnp.clip(stds, self.std_min, self.std_max) * jnp.sqrt(temperature)

        if self.std_clip is not None and self.tanh_squash_distribution:
            return TanhMultivariateNormalDiag(loc=means, scale_diag=stds, low=-self.std_clip, high=self.std_clip)
        if self.std_clip is not None:
            return ClipMultivariateNormalDiag(loc=means, scale_diag=stds, clip=self.std_clip)
        if self.tanh_squash_distribution:
            return TanhMultivariateNormalDiag(loc=means, scale_diag=stds)
        return distrax.MultivariateNormalDiag(loc=means, scale_diag=stds)


class ClipMultivariateNormalDiag(distrax.MultivariateNormalDiag):
    """Diagonal Gaussian whose samples are clipped to [loc - clip, loc + clip]."""

    def __init__(self, loc: jnp.ndarray, scale_diag: jnp.ndarray, clip: Optional[float] = None):
        super().__init__(loc=loc, scale_diag=scale_diag)
        self.clip_loc = jnp.array(loc)
        self.clip = clip

    def sample(self, seed=None, sample_shape=()):
        sample = super().sample(seed=seed, sample_shape=sample_shape)
        if self.clip is not None:
            sample = jnp.clip(sample, self.clip_loc - self.clip, self.clip_loc + self.clip)
        return sample


class TanhMultivariateNormalDiag(distrax.Transformed):
    """Tanh-squashed diagonal Gaussian, optionally rescaled from (-1, 1) to (low, high)."""

    def __init__(self, loc: jnp.ndarray, scale_diag: jnp.ndarray,
                 low: Optional[jnp.ndarray] = None, high: Optional[jnp.ndarray] = None):
        distribution = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
        layers = []
        if not (low is None or high is None):
            def rescale_from_tanh(x):
                return (x + 1) / 2 * (high - low) + low

            def forward_log_det_jacobian(x):
                high_ = jnp.broadcast_to(high, x.shape)
                low_ = jnp.broadcast_to(low, x.shape)
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

            layers.append(distrax.Lambda(rescale_from_tanh, forward_log_det_jacobian=forward_log_det_jacobian,
                                         event_ndims_in=1, event_ndims_out=1))
        layers.append(distrax.Block(distrax.Tanh(), 1))
        super().__init__(distribution=distribution, bijector=distrax.Chain(layers))

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    def stddev(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.stddev())
