"""HiRe-MoMa: hierarchical residual policy with a mobile-base head and an arm head.

    a_res_base = pi_base(o, a_base_base)
    a_res_arm  = pi_arm(a_res_base, o, a_base_arm)        (one-way conditioning)
    a_exe      = a_base + [a_res_arm, 0 (gripper), a_res_base]

Each head is a residual SAC agent with its own critic. With `conditioning=False` the
arm head ignores the base residual (+Decomposition Only, App. A.1).
The gripper is not corrected: its residual is always 0.
"""
from functools import partial
from typing import FrozenSet, Optional, Sequence

import flax
import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from hire_moma_launcher.agents.continuous.sac import ActionGroup, SACAgent
from hire_moma_launcher.common.common import JaxRLTrainState, nonpytree_field
from hire_moma_launcher.common.typing import Batch, Data, PRNGKey
from hire_moma_launcher.utils.train_utils import unpack_batch


@flax.struct.dataclass
class HiReMoMaState:
    """Checkpointable view of both heads."""

    base: JaxRLTrainState
    arm: JaxRLTrainState

    def get_params(self) -> FrozenDict:
        return FrozenDict({"base": self.base.params, "arm": self.arm.params})

    def replace_with_params(self, *, params: FrozenDict):
        return self.replace(base=self.base.replace(params=params["base"]),
                            arm=self.arm.replace(params=params["arm"]))


class HiReMoMaAgent(flax.struct.PyTreeNode):
    agent_base: SACAgent
    agent_arm: SACAgent
    rng: PRNGKey
    config: dict = nonpytree_field()

    def _with_base_residual(self, observations, base_residual):
        return observations.copy(add_or_replace={"base_residual": jnp.expand_dims(base_residual, axis=-2)})

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(self, batch: Batch, *, pmap_axis: Optional[str] = None,
               networks_to_update: FrozenSet[str] = frozenset({"actor", "critic", "temperature"})):
        rng = self.rng
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = unpack_batch(batch)

        new_base, info_base = self.agent_base.update(batch, pmap_axis=pmap_axis,
                                                     networks_to_update=networks_to_update)

        batch_arm = batch
        if self.config["conditioning"]:
            # Condition the arm head on residuals sampled from the (just updated) base head.
            rng, base_rng, next_base_rng = jax.random.split(rng, 3)
            base_res = jax.lax.stop_gradient(new_base.sample_actions(batch["observations"], seed=base_rng))
            next_base_res = jax.lax.stop_gradient(new_base.sample_actions(batch["next_observations"], seed=next_base_rng))
            batch_arm = batch.copy(add_or_replace={
                "observations": self._with_base_residual(batch["observations"], base_res),
                "next_observations": self._with_base_residual(batch["next_observations"], next_base_res),
            })
        new_arm, info_arm = self.agent_arm.update(batch_arm, pmap_axis=pmap_axis,
                                                  networks_to_update=networks_to_update)

        return self.replace(agent_base=new_base, agent_arm=new_arm, rng=rng), {"base": info_base, "arm": info_arm}

    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(self, observations: Data, *, seed: PRNGKey, argmax: bool = False) -> jnp.ndarray:
        """Full residual action vector (gripper entries are 0)."""
        rng_base, rng_arm, _ = jax.random.split(seed, 3)
        base_res = self.agent_base.sample_actions(observations, seed=rng_base, argmax=argmax)
        observations = {**observations, "base_residual": jnp.expand_dims(base_res, axis=-2)}
        arm_res = self.agent_arm.sample_actions(observations, seed=rng_arm, argmax=argmax)

        full = jnp.zeros(arm_res.shape[:-1] + (self.config["action_dim"],))
        for group, part in ((self.config["arm_group"], arm_res), (self.config["base_group"], base_res)):
            idx = jnp.concatenate([jnp.arange(b, e) for b, e in zip(group.begins, group.ends)])
            full = full.at[..., idx].set(part)
        return full

    def get_state(self) -> HiReMoMaState:
        return HiReMoMaState(base=self.agent_base.state, arm=self.agent_arm.state)

    def replace_with_state(self, state: HiReMoMaState):
        return self.replace(agent_base=self.agent_base.replace(state=state.base),
                            agent_arm=self.agent_arm.replace(state=state.arm))

    def get_params(self):
        return self.get_state().get_params()

    def with_params(self, params):
        return self.replace_with_state(self.get_state().replace_with_params(params=params))

    @classmethod
    def create_pixels(cls, rng: PRNGKey, observations: Data, actions: jnp.ndarray, *,
                      base_group: ActionGroup, arm_group: ActionGroup, conditioning: bool = True,
                      image_keys: Sequence[str] = ("image",), target_entropy_per_dim: float = -0.5,
                      **head_kwargs):
        """`head_kwargs` go to `SACAgent.create_pixels` (e.g. `full_base_action`)."""
        rng, rng_base, rng_arm, _ = jax.random.split(rng, 4)
        agent_base = SACAgent.create_pixels(rng_base, observations, actions, action_group=base_group,
                                            image_keys=image_keys,
                                            target_entropy=target_entropy_per_dim * base_group.dim, **head_kwargs)
        agent_arm = SACAgent.create_pixels(rng_arm, observations, actions, action_group=arm_group,
                                           image_keys=image_keys,
                                           base_residual_dim=base_group.dim if conditioning else None,
                                           target_entropy=target_entropy_per_dim * arm_group.dim, **head_kwargs)
        return cls(agent_base=agent_base, agent_arm=agent_arm, rng=rng, config=dict(
            image_keys=image_keys, conditioning=conditioning, action_dim=actions.shape[-1],
            base_group=base_group, arm_group=arm_group))
