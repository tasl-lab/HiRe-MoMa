"""Residual SAC head for one group of action dimensions (e.g. the mobile base or the arm(s))."""
from functools import partial
from typing import FrozenSet, Iterable, NamedTuple, Optional, Sequence, Tuple

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from hire_moma_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from hire_moma_launcher.common.encoding import BaseActEncodingWrapper, BaseResidualEncodingWrapper, EncodingWrapper
from hire_moma_launcher.common.optimizers import make_optimizer
from hire_moma_launcher.common.typing import Batch, Data, Params, PRNGKey
from hire_moma_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
from hire_moma_launcher.networks.lagrange import GeqLagrangeMultiplier
from hire_moma_launcher.networks.mlp import MLP
from hire_moma_launcher.utils.train_utils import load_resnet10_params, unpack_batch
from hire_moma_launcher.vision.resnet_v1 import make_image_encoders


class ActionGroup(NamedTuple):
    """Dimensions of the full action vector controlled by one head, as [begin, end) slices."""

    name: str
    begins: Sequence[int]
    ends: Sequence[int]

    @property
    def dim(self) -> int:
        return sum(e - b for b, e in zip(self.begins, self.ends))

    def extract(self, x: jnp.ndarray) -> jnp.ndarray:
        out = jnp.concatenate([x[..., b:e] for b, e in zip(self.begins, self.ends)], axis=-1)
        assert out.shape[-1] == self.dim, "Action dimension mismatch!"
        return out


def merge_last_ndims(x, n=2):
    return x.reshape(x.shape[:-n] + (-1,))


def slice_batch(batch, start_idx, stop_idx=None):
    return jax.tree_util.tree_map(lambda x: x[start_idx:stop_idx, ...] if isinstance(x, jnp.ndarray) else x, batch)


class SACAgent(flax.struct.PyTreeNode):
    """Residual SAC over one action group.

    The policy outputs a bounded residual; the critic scores the executed action
    a_exe = a_base + a_res (restricted to this group's dimensions).
    """

    state: JaxRLTrainState
    config: dict = nonpytree_field()

    # ---------------------------------------------------------------- forward passes
    def forward_critic(self, observations: Data, actions: jax.Array, rng: PRNGKey, *,
                       grad_params: Optional[Params] = None, train: bool = True) -> jax.Array:
        return self.state.apply_fn({"params": grad_params or self.state.params}, observations, actions,
                                   name="critic", rngs={"dropout": rng} if train else {}, train=train)

    def forward_target_critic(self, observations: Data, actions: jax.Array, rng: PRNGKey) -> jax.Array:
        return self.forward_critic(observations, actions, rng=rng, grad_params=self.state.target_params)

    def forward_policy(self, observations: Data, rng: Optional[PRNGKey] = None, *,
                       grad_params: Optional[Params] = None, train: bool = True) -> distrax.Distribution:
        return self.state.apply_fn({"params": grad_params or self.state.params}, observations,
                                   name="actor", rngs={"dropout": rng} if train else {}, train=train)

    def forward_temperature(self, *, grad_params: Optional[Params] = None):
        return self.state.apply_fn({"params": grad_params or self.state.params}, name="temperature")

    def temperature_lagrange_penalty(self, entropy: jnp.ndarray, *, grad_params: Optional[Params] = None):
        return self.state.apply_fn({"params": grad_params or self.state.params}, lhs=entropy,
                                   rhs=self.config["target_entropy"], name="temperature")

    def _base_actions(self, observations):
        group = self.config["action_group"]
        return merge_last_ndims(group.extract(observations["base_actions"]), 2)

    def _compute_next_actions(self, batch, rng):
        """a_exe' = a_base' + a_res', a_res' ~ pi(.|o')."""
        batch_size = batch["rewards"].shape[0]
        policy_rng, sample_rng = jax.random.split(rng, 2)
        dist = self.forward_policy(batch["next_observations"], rng=policy_rng)
        next_residuals, next_log_probs = dist.sample_and_log_prob(seed=sample_rng)
        chex.assert_shape(next_log_probs, (batch_size,))
        return self._base_actions(batch["next_observations"]) + next_residuals, next_log_probs

    # ---------------------------------------------------------------- losses
    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        actions = self.config["action_group"].extract(batch["actions"])  # executed (incl. human) actions

        rng, next_action_key = jax.random.split(rng)
        next_actions, next_log_probs = self._compute_next_actions(batch, next_action_key)
        target_next_qs = self.forward_target_critic(batch["next_observations"], next_actions, rng=rng)
        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            idxs = jax.random.randint(subsample_key, (self.config["critic_subsample_size"],), 0,
                                      self.config["critic_ensemble_size"])
            target_next_qs = target_next_qs[idxs]
        target_next_min_q = target_next_qs.min(axis=0)  # conservative min over the ensemble
        chex.assert_shape(target_next_min_q, (batch_size,))

        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * target_next_min_q
        if self.config["backup_entropy"]:
            temperature = self.forward_temperature()
            target_q = target_q - self.config["discount"] * batch["masks"] * temperature * next_log_probs

        predicted_qs = self.forward_critic(batch["observations"], actions, rng=rng, grad_params=params)
        chex.assert_shape(predicted_qs, (self.config["critic_ensemble_size"], batch_size))
        target_qs = target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        info = {
            "critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(target_qs),
            "rewards": batch["rewards"].mean(),
        }
        if self.config["lambda_imi"] > 0:
            rng, align_rng = jax.random.split(rng, 2)
            align_loss = self.value_alignment_loss(batch, params, align_rng)
            critic_loss += self.config["lambda_imi"] * align_loss
            info.update(critic_loss=critic_loss, alignment_loss=align_loss)
        return critic_loss, info

    def value_alignment_loss(self, batch, params: Params, rng: PRNGKey):
        """Expert-guided value alignment: E[log(1 + exp(Q(o, a_base + a_res) - Q(o, a_h)))].

        The second half of every batch comes from the intervention (expert) buffer.
        """
        half = batch["rewards"].shape[0] // 2
        obs = slice_batch(batch["observations"], half)
        base_actions = slice_batch(self._base_actions(batch["observations"]), half)
        expert_actions = slice_batch(self.config["action_group"].extract(batch["actions"]), half)

        rng, policy_rng, sample_rng = jax.random.split(rng, 3)
        residuals, _ = self.forward_policy(obs, rng=policy_rng).sample_and_log_prob(seed=sample_rng)

        rng, rng_policy_q, rng_expert_q = jax.random.split(rng, 3)
        q_policy = self.forward_critic(obs, base_actions + residuals, rng=rng_policy_q, grad_params=params)
        q_expert = self.forward_critic(obs, expert_actions, rng=rng_expert_q, grad_params=params)
        chex.assert_shape(q_policy, (self.config["critic_ensemble_size"], half))
        return jnp.mean(jax.nn.softplus(q_policy - q_expert))

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
        loss = self.temperature_lagrange_penalty(-next_log_probs.mean(), grad_params=params)
        return loss, {"temperature_loss": loss}

    def loss_fns(self, batch):
        return {
            "critic": partial(self.critic_loss_fn, batch),
            "actor": partial(self.policy_loss_fn, batch),
            "temperature": partial(self.temperature_loss_fn, batch),
        }

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(self, batch: Batch, *, pmap_axis: Optional[str] = None,
               networks_to_update: FrozenSet[str] = frozenset({"actor", "critic", "temperature"})
               ) -> Tuple["SACAgent", dict]:
        """One gradient step on `networks_to_update` (critic-only steps implement the UTD ratio)."""
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
        new_state = new_state.replace(rng=rng)
        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(self, observations: Data, *, seed: Optional[PRNGKey] = None, argmax: bool = False) -> jnp.ndarray:
        """Residual action for this group (the mean if `argmax`)."""
        dist = self.forward_policy(observations, rng=seed, train=False)
        return dist.mode() if argmax else dist.sample(seed=seed)

    # ---------------------------------------------------------------- construction
    @classmethod
    def create(cls, rng: PRNGKey, observations: Data, actions: jnp.ndarray, *,
               actor_def: nn.Module, critic_def: nn.Module, temperature_def: nn.Module,
               action_group: ActionGroup, learning_rate: float = 3e-4, discount: float = 0.99,
               lambda_imi: float = 0.0, reward_bias: float = 0.0, target_entropy: Optional[float] = None,
               critic_ensemble_size: int = 10, critic_subsample_size: Optional[int] = None,
               image_keys: Iterable[str] = None, augmentation_function=None,
               soft_target_update_rate: float = 0.005, backup_entropy: bool = False):
        model_def = ModuleDict({"critic": critic_def, "actor": actor_def, "temperature": temperature_def})
        txs = {
            "critic": make_optimizer(learning_rate=learning_rate),
            "actor": make_optimizer(learning_rate=learning_rate),
            "temperature": make_optimizer(learning_rate=learning_rate),
        }
        rng, init_rng = jax.random.split(rng)
        params = model_def.init(init_rng, critic=[observations, action_group.extract(actions)],
                                actor=[observations], temperature=[])["params"]
        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(apply_fn=model_def.apply, params=params, txs=txs,
                                       target_params=params, rng=create_rng)
        if target_entropy is None:
            target_entropy = -action_group.dim / 2  # see TrainConfig.target_entropy_per_dim
        return cls(state=state, config=dict(
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            discount=discount,
            lambda_imi=lambda_imi,
            soft_target_update_rate=soft_target_update_rate,
            target_entropy=target_entropy,
            backup_entropy=backup_entropy,
            image_keys=image_keys,
            reward_bias=reward_bias,
            augmentation_function=augmentation_function,
            action_group=action_group,
        ))

    @classmethod
    def create_pixels(cls, rng: PRNGKey, observations: Data, actions: jnp.ndarray, *,
                      action_group: ActionGroup, image_keys: Iterable[str] = ("image",),
                      base_residual_dim: Optional[int] = None, full_base_action: bool = False,
                      use_proprio: bool = True, critic_network_kwargs: dict, policy_network_kwargs: dict,
                      policy_kwargs: dict, temperature_init: float = 0.1, **kwargs):
        """Image-based head.

        The actor sees the base-policy action of this head's dimensions only (Eq. 3:
        pi_mo(o, a_base_mo), pi_ma(a_res_mo, o, a_base_ma)); `full_base_action=True` gives it the
        whole base action instead.
        If `base_residual_dim` is set, actor and critic are additionally conditioned on the
        base residual stored under `observations["base_residual"]` (HiRe-MoMa arm head).
        """
        policy_network_kwargs = {**policy_network_kwargs, "activate_final": True}
        critic_network_kwargs = {**critic_network_kwargs, "activate_final": True}

        # Shared encoder: only the critic loss updates the image/proprio features.
        encoder_critic = EncodingWrapper(encoder=make_image_encoders(image_keys), use_proprio=use_proprio,
                                         enable_stacking=True, image_keys=image_keys)
        if base_residual_dim is not None:
            encoder_critic = BaseResidualEncodingWrapper(encoding_wrapper=encoder_critic, enable_stacking=True,
                                                         image_keys=image_keys)
            observations = {**observations, "base_residual": jnp.zeros((1, base_residual_dim))}
        base_slices = None if full_base_action else tuple(zip(action_group.begins, action_group.ends))
        encoder_actor = BaseActEncodingWrapper(encoding_wrapper=encoder_critic, enable_stacking=True,
                                               image_keys=image_keys, slices=base_slices)

        critic_backbone = ensemblize(partial(MLP, **critic_network_kwargs),
                                     kwargs.get("critic_ensemble_size", 10))(name="critic_ensemble")
        critic_def = Critic(encoder=encoder_critic, network=critic_backbone, name="critic")
        policy_def = Policy(encoder=encoder_actor, network=MLP(**policy_network_kwargs),
                            action_dim=action_group.dim, **policy_kwargs, name="actor")
        temperature_def = GeqLagrangeMultiplier(init_value=temperature_init, constraint_shape=(), name="temperature")

        agent = cls.create(rng, observations, actions, actor_def=policy_def, critic_def=critic_def,
                           temperature_def=temperature_def, action_group=action_group,
                           image_keys=image_keys, **kwargs)
        return load_resnet10_params(agent, image_keys)
