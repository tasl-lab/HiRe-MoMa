"""Flat Residual baseline (HIL-SERL residual): one SAC head over all continuous dimensions.

As in HIL-SERL, a discrete grasp critic (DQN) is trained on the gripper dimensions and shares
the image encoder with the critic. Its gripper command is not applied (the gripper residual is
always 0, as for all methods), so it only shapes the shared encoder.
"""
from functools import partial
from typing import FrozenSet, Iterable, Optional, Sequence, Tuple

import chex
import flax
import jax
import jax.numpy as jnp
import numpy as np

from hire_moma_launcher.agents.continuous.sac import merge_last_ndims
from hire_moma_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from hire_moma_launcher.common.encoding import BaseActEncodingWrapper, EncodingWrapper
from hire_moma_launcher.common.optimizers import make_optimizer
from hire_moma_launcher.common.typing import Batch, Data, Params, PRNGKey
from hire_moma_launcher.networks.actor_critic_nets import Critic, GraspCritic, Policy, ensemblize
from hire_moma_launcher.networks.lagrange import GeqLagrangeMultiplier
from hire_moma_launcher.networks.mlp import MLP
from hire_moma_launcher.utils.train_utils import load_resnet10_params, unpack_batch
from hire_moma_launcher.vision.resnet_v1 import make_image_encoders


class FlatResidualAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()

    # ---------------------------------------------------------------- helpers
    def _continuous(self, x):
        """Drop the gripper dimensions."""
        return jnp.concatenate([x[..., b:e] for b, e in self.config["continuous_slices"]], axis=-1)

    def _base_actions(self, observations):
        return merge_last_ndims(self._continuous(observations["base_actions"]), 2)

    def forward_critic(self, observations, actions, rng, *, grad_params=None, train=True):
        return self.state.apply_fn({"params": grad_params or self.state.params}, observations, actions,
                                   name="critic", rngs={"dropout": rng} if train else {}, train=train)

    def forward_target_critic(self, observations, actions, rng):
        return self.forward_critic(observations, actions, rng=rng, grad_params=self.state.target_params)

    def forward_grasp_critic(self, observations, rng, *, grad_params=None, train=True):
        return self.state.apply_fn({"params": grad_params or self.state.params}, observations,
                                   name="grasp_critic", rngs={"dropout": rng} if train else {}, train=train)

    def forward_target_grasp_critic(self, observations, rng):
        return self.forward_grasp_critic(observations, rng=rng, grad_params=self.state.target_params)

    def forward_policy(self, observations, rng=None, *, grad_params=None, train=True):
        return self.state.apply_fn({"params": grad_params or self.state.params}, observations,
                                   name="actor", rngs={"dropout": rng} if train else {}, train=train)

    def forward_temperature(self, *, grad_params=None):
        return self.state.apply_fn({"params": grad_params or self.state.params}, name="temperature")

    def _compute_next_actions(self, batch, rng):
        batch_size = batch["rewards"].shape[0]
        policy_rng, sample_rng = jax.random.split(rng, 2)
        dist = self.forward_policy(batch["next_observations"], rng=policy_rng)
        next_residuals, next_log_probs = dist.sample_and_log_prob(seed=sample_rng)
        chex.assert_shape(next_log_probs, (batch_size,))
        return self._base_actions(batch["next_observations"]) + next_residuals, next_log_probs

    def _grasp_index(self, actions):
        """Gripper commands in {-1, 1} -> joint discrete index in [0, 2^n)."""
        idx = 0
        for d in self.config["grip_dims"]:
            g = (jnp.round(jnp.round(actions[..., d]).astype(jnp.int16)) > 0.0).astype(jnp.int16)
            idx = idx * 2 + g
        return idx

    # ---------------------------------------------------------------- losses
    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        actions = self._continuous(batch["actions"])
        rng, next_action_key = jax.random.split(rng)
        next_actions, next_log_probs = self._compute_next_actions(batch, next_action_key)
        target_next_qs = self.forward_target_critic(batch["next_observations"], next_actions, rng=rng)
        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            idxs = jax.random.randint(subsample_key, (self.config["critic_subsample_size"],), 0,
                                      self.config["critic_ensemble_size"])
            target_next_qs = target_next_qs[idxs]
        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * target_next_qs.min(axis=0)
        if self.config["backup_entropy"]:
            temperature = self.forward_temperature()
            target_q = target_q - self.config["discount"] * batch["masks"] * temperature * next_log_probs
        predicted_qs = self.forward_critic(batch["observations"], actions, rng=rng, grad_params=params)
        chex.assert_shape(predicted_qs, (self.config["critic_ensemble_size"], batch_size))
        target_qs = target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)
        return critic_loss, {"critic_loss": critic_loss, "predicted_qs": jnp.mean(predicted_qs),
                             "target_qs": jnp.mean(target_qs), "rewards": batch["rewards"].mean()}

    def grasp_critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """Double-DQN loss on the (joint) gripper command; reward includes the grasp penalty."""
        batch_size = batch["rewards"].shape[0]
        n = self.config["num_grasp_actions"]
        grasp_action = self._grasp_index(batch["actions"])
        target_next_qs = self.forward_target_grasp_critic(batch["next_observations"], rng=rng)
        chex.assert_shape(target_next_qs, (batch_size, n))
        best_next = self.forward_grasp_critic(batch["next_observations"], rng=rng).argmax(axis=-1)
        target_next_q = target_next_qs[jnp.arange(batch_size), best_next]
        grasp_rewards = batch["rewards"] + batch.get("grasp_penalty", jnp.zeros_like(batch["rewards"]))
        target_q = grasp_rewards + self.config["discount"] * batch["masks"] * target_next_q
        predicted_qs = self.forward_grasp_critic(batch["observations"], rng=rng, grad_params=params)
        chex.assert_shape(predicted_qs, (batch_size, n))
        predicted_q = predicted_qs[jnp.arange(batch_size), grasp_action]
        loss = jnp.mean((predicted_q - target_q) ** 2)
        return loss, {"grasp_critic_loss": loss, "predicted_grasp_qs": jnp.mean(predicted_q),
                      "target_grasp_qs": jnp.mean(target_q), "grasp_rewards": grasp_rewards.mean()}

    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
        dist = self.forward_policy(batch["observations"], rng=policy_rng, grad_params=params)
        residuals, log_probs = dist.sample_and_log_prob(seed=sample_rng)
        actions = self._base_actions(batch["observations"]) + residuals
        predicted_q = self.forward_critic(batch["observations"], actions, rng=critic_rng).mean(axis=0)
        chex.assert_shape(predicted_q, (batch_size,))
        temperature = self.forward_temperature()
        actor_loss = -jnp.mean(predicted_q - temperature * log_probs)
        return actor_loss, {"actor_loss": actor_loss, "temperature": temperature, "entropy": -log_probs.mean()}

    def temperature_loss_fn(self, batch, params: Params, rng: PRNGKey):
        rng, next_action_key = jax.random.split(rng)
        _, next_log_probs = self._compute_next_actions(batch, next_action_key)
        loss = self.state.apply_fn({"params": params}, lhs=-next_log_probs.mean(),
                                   rhs=self.config["target_entropy"], name="temperature")
        return loss, {"temperature_loss": loss}

    def loss_fns(self, batch):
        return {
            "critic": partial(self.critic_loss_fn, batch),
            "grasp_critic": partial(self.grasp_critic_loss_fn, batch),
            "actor": partial(self.policy_loss_fn, batch),
            "temperature": partial(self.temperature_loss_fn, batch),
        }

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(self, batch: Batch, *, pmap_axis: Optional[str] = None,
               networks_to_update: FrozenSet[str] = frozenset({"actor", "critic", "grasp_critic", "temperature"})
               ) -> Tuple["FlatResidualAgent", dict]:
        chex.assert_tree_shape_prefix(batch, (batch["rewards"].shape[0],))
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = unpack_batch(batch)
        rng, aug_rng = jax.random.split(self.state.rng)
        if self.config.get("augmentation_function") is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)
        batch = batch.copy(add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]})

        loss_fns = self.loss_fns(batch)
        assert networks_to_update.issubset(loss_fns.keys()), f"Invalid gradient steps: {networks_to_update}"
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})
        new_state, info = self.state.apply_loss_fns(loss_fns, pmap_axis=pmap_axis, has_aux=True)
        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])
        return self.replace(state=new_state.replace(rng=rng)), info

    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(self, observations: Data, *, seed: Optional[PRNGKey] = None, argmax: bool = False):
        """Full residual action vector (gripper entries are 0)."""
        dist = self.forward_policy(observations, rng=seed, train=False)
        residual = dist.mode() if argmax else dist.sample(seed=seed)
        parts, i = [], 0
        for b, e in self.config["continuous_slices"]:
            if parts or b > 0:
                parts.append(jnp.zeros(residual.shape[:-1] + (1,)))  # gripper residual
            parts.append(residual[..., i:i + (e - b)])
            i += e - b
        return jnp.concatenate(parts, axis=-1)

    def get_state(self):
        return self.state

    def replace_with_state(self, state):
        return self.replace(state=state)

    def get_params(self):
        return self.state.params

    def with_params(self, params):
        return self.replace(state=self.state.replace(params=params))

    # ---------------------------------------------------------------- construction
    @classmethod
    def create_pixels(cls, rng: PRNGKey, observations: Data, actions: jnp.ndarray, *,
                      grip_dims: Sequence[int], image_keys: Iterable[str] = ("image",),
                      critic_network_kwargs: dict, grasp_critic_network_kwargs: dict,
                      policy_network_kwargs: dict, policy_kwargs: dict, learning_rate: float = 3e-4,
                      discount: float = 0.99, reward_bias: float = 0.0, target_entropy: Optional[float] = None,
                      critic_ensemble_size: int = 10, critic_subsample_size: Optional[int] = None,
                      temperature_init: float = 0.1, augmentation_function=None,
                      soft_target_update_rate: float = 0.005, backup_entropy: bool = False, use_proprio: bool = True,
                      **unused):
        action_dim = actions.shape[-1]
        assert list(grip_dims) == sorted(grip_dims) and grip_dims[-1] < action_dim - 1
        bounds = [-1, *grip_dims, action_dim]
        continuous_slices = tuple((bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1))
        num_grasp_actions = 2 ** len(grip_dims)

        policy_network_kwargs = {**policy_network_kwargs, "activate_final": True}
        critic_network_kwargs = {**critic_network_kwargs, "activate_final": True}
        encoder_critic = EncodingWrapper(encoder=make_image_encoders(image_keys), use_proprio=use_proprio,
                                         enable_stacking=True, image_keys=image_keys)
        encoder_actor = BaseActEncodingWrapper(encoding_wrapper=encoder_critic, enable_stacking=True,
                                               image_keys=image_keys)
        critic_backbone = ensemblize(partial(MLP, **critic_network_kwargs), critic_ensemble_size)(name="critic_ensemble")
        critic_def = Critic(encoder=encoder_critic, network=critic_backbone, name="critic")
        grasp_critic_def = GraspCritic(encoder=encoder_critic, network=MLP(**grasp_critic_network_kwargs),
                                       output_dim=num_grasp_actions, name="grasp_critic")
        policy_def = Policy(encoder=encoder_actor, network=MLP(**policy_network_kwargs),
                            action_dim=action_dim - len(grip_dims), **policy_kwargs, name="actor")
        temperature_def = GeqLagrangeMultiplier(init_value=temperature_init, constraint_shape=(), name="temperature")

        observations = {**observations, "base_actions": np.expand_dims(np.zeros_like(actions), axis=-2)}
        model_def = ModuleDict({"actor": policy_def, "critic": critic_def,
                                "grasp_critic": grasp_critic_def, "temperature": temperature_def})
        txs = {k: make_optimizer(learning_rate=learning_rate) for k in ("actor", "critic", "grasp_critic", "temperature")}
        rng, init_rng = jax.random.split(rng)
        continuous_actions = jnp.concatenate([actions[..., b:e] for b, e in continuous_slices], axis=-1)
        params = model_def.init(init_rng, actor=[observations], critic=[observations, continuous_actions],
                                grasp_critic=[observations], temperature=[])["params"]
        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(apply_fn=model_def.apply, params=params, txs=txs,
                                       target_params=params, rng=create_rng)
        if target_entropy is None:
            target_entropy = -action_dim / 2  # counts the gripper dims too (as HIL-SERL)
        agent = cls(state=state, config=dict(
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            discount=discount,
            soft_target_update_rate=soft_target_update_rate,
            target_entropy=target_entropy,
            backup_entropy=backup_entropy,
            image_keys=image_keys,
            reward_bias=reward_bias,
            augmentation_function=augmentation_function,
            grip_dims=tuple(grip_dims),
            continuous_slices=continuous_slices,
            num_grasp_actions=num_grasp_actions,
        ))
        return load_resnet10_params(agent, image_keys)
