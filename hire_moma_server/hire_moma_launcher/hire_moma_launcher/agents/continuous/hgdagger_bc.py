"""HG-DAgger baseline: a residual policy regressed (MSE) onto human corrections."""
from functools import partial
from typing import Iterable, Optional, Sequence

import flax
import jax
import jax.numpy as jnp
import optax

from hire_moma_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from hire_moma_launcher.common.encoding import BaseActEncodingWrapper, EncodingWrapper
from hire_moma_launcher.common.typing import Batch, PRNGKey
from hire_moma_launcher.networks.actor_critic_nets import Policy
from hire_moma_launcher.networks.mlp import MLP
from hire_moma_launcher.utils.train_utils import load_resnet10_params, unpack_batch
from hire_moma_launcher.vision.resnet_v1 import make_image_encoders


class BCAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None, **unused):
        """`batch["actions"]` are residual labels (human action minus base action)."""
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = unpack_batch(batch)
        rng, aug_rng = jax.random.split(self.state.rng)
        if self.config.get("augmentation_function") is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)

        def loss_fn(params, rng):
            rng, key = jax.random.split(rng)
            dist = self.state.apply_fn({"params": params}, batch["observations"], temperature=1.0,
                                       train=True, rngs={"dropout": key}, name="actor")
            targets = batch["actions"]
            if self.config["tanh_squash_distribution"]:
                targets = jnp.clip(targets, -1 + 1e-6, 1 - 1e-6)
            nll = -dist.log_prob(targets).mean()
            mse = ((dist.mode() - targets) ** 2).sum(-1).mean()
            loss = {"likelihood": nll, "mse": mse}[self.config["loss_type"]]
            return loss, {"actor_loss": loss, "likelihood": nll, "mse": mse}

        new_state, info = self.state.apply_loss_fns(loss_fn, pmap_axis=pmap_axis, has_aux=True)
        return self.replace(state=new_state.replace(rng=rng)), info

    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(self, observations, *, seed: Optional[PRNGKey] = None, argmax: bool = False) -> jnp.ndarray:
        dist = self.state.apply_fn({"params": self.state.params}, observations, temperature=1.0, name="actor")
        actions = dist.mode() if argmax else dist.sample(seed=seed)
        for d in self.config["grip_dims"]:
            actions = actions.at[..., d].set(0.0)  # no gripper correction
        return actions

    def get_state(self):
        return self.state

    def replace_with_state(self, state):
        return self.replace(state=state)

    def get_params(self):
        return self.state.params

    def with_params(self, params):
        return self.replace(state=self.state.replace(params=params))

    @classmethod
    def create(cls, rng: PRNGKey, observations, actions: jnp.ndarray, *, grip_dims: Sequence[int],
               image_keys: Iterable[str] = ("image",), loss_type: str = "mse",
               network_kwargs: dict, policy_kwargs: dict, learning_rate: float = 3e-4,
               augmentation_function=None, use_proprio: bool = True, **unused):
        encoder = EncodingWrapper(encoder=make_image_encoders(image_keys), use_proprio=use_proprio,
                                  enable_stacking=True, image_keys=image_keys)
        encoder = BaseActEncodingWrapper(encoding_wrapper=encoder, enable_stacking=True, image_keys=image_keys)
        model_def = ModuleDict({"actor": Policy(encoder, MLP(**{**network_kwargs, "activate_final": True}),
                                                action_dim=actions.shape[-1], **policy_kwargs)})
        rng, init_rng = jax.random.split(rng)
        params = model_def.init(init_rng, actor=[observations])["params"]
        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(apply_fn=model_def.apply, params=params, txs=optax.adam(learning_rate),
                                       target_params=params, rng=create_rng)
        agent = cls(state, dict(
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            tanh_squash_distribution=policy_kwargs["tanh_squash_distribution"],
            loss_type=loss_type,
            grip_dims=tuple(grip_dims),
        ))
        return load_resnet10_params(agent, image_keys)
