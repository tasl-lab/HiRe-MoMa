"""Task, method and training configuration (paper values) and environment construction."""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from hire_moma_launcher.agents.continuous.sac import ActionGroup


@dataclass(frozen=True)
class TaskConfig:
    name: str
    arms: Tuple[str, ...]                 # arms in action order, e.g. ("right",) or ("left", "right")
    max_episode_length: int               # steps at 10 Hz
    reset_arms: Dict[str, list]           # per arm: 7 joint angles + gripper (0 close / 1 open)
    torso: float                          # fixed torso height
    head: Tuple[float, float] = (0.0, -0.90)  # fixed head pan / tilt
    grippers_closed: bool = False         # keep grippers closed (tools held in hand)
    grasp_penalty: Optional[float] = -0.02  # reward of the Flat Residual grasp critic (single-arm tasks)
    image_key: str = "tiago_head_image"
    image_shape: Tuple[int, int, int] = (160, 208, 3)  # (H, W, C), App. C.1
    # Base policy (DP, App. D.3): observation 1, prediction 8, action horizon 6. Each query returns
    # `bp_act_horizon` actions, of which the first `bp_skip` are dropped (0: paper; 3: latency hiding).
    bp_act_horizon: int = 6
    bp_skip: int = 0
    obs_horizon: int = 1

    @property
    def proprio_keys(self):
        return (*self.arms, "base_pose", "base_velocity", "torso")

    @property
    def act_keys(self):
        return (*self.arms, "base")

    @property
    def state_dim(self) -> int:  # per arm: xyz + quat + gripper; base pose (3) + velocity (3) + torso (1)
        return 8 * len(self.arms) + 7

    @property
    def action_dim(self) -> int:  # per arm: 6-D delta pose + gripper; base: 3-D velocity
        return 7 * len(self.arms) + 3

    @property
    def grip_dims(self) -> Tuple[int, ...]:
        return tuple(7 * i + 6 for i in range(len(self.arms)))

    @property
    def base_group(self) -> ActionGroup:
        n = 7 * len(self.arms)
        return ActionGroup("base", (n,), (n + 3,))

    @property
    def arm_group(self) -> ActionGroup:
        return ActionGroup("arm", tuple(7 * i for i in range(len(self.arms))),
                           tuple(7 * i + 6 for i in range(len(self.arms))))

    @property
    def reset_pose(self) -> dict:
        return {**{arm: list(self.reset_arms[arm]) for arm in ("left", "right")},
                "torso": self.torso, "head": list(self.head)}


@dataclass(frozen=True)
class MethodConfig:
    agent: str                    # "hire_moma" | "flat_residual" | "hg_dagger"
    conditioning: bool = True     # arm head conditioned on a_res_mo (Eq. 3)
    lambda_imi: float = 0.0       # weight of the value-alignment loss (Eq. 4)


METHODS = {
    "hire_moma": MethodConfig("hire_moma", conditioning=True, lambda_imi=1e-3),           # HiRe-MoMa (Full)
    "hire_moma_no_align": MethodConfig("hire_moma", conditioning=True, lambda_imi=0.0),   # +Hierarchical Decoupling (Tab. 2)
    "decomposed": MethodConfig("hire_moma", conditioning=False, lambda_imi=0.0),          # +Decomposition Only (Tab. 4)
    "flat_residual": MethodConfig("flat_residual"),                                       # +Flat Residual (HIL-SERL)
    "hg_dagger": MethodConfig("hg_dagger"),                                               # +HG-DAgger
}


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256            # balanced sampling: half replay, half intervention buffer
    cta_ratio: int = 4               # critic updates per actor update
    discount: float = 0.99
    residual_bound: float = 0.3      # epsilon, per dimension of the scaled action space
    action_range_margin: float = 0.2  # demonstrated action range widened by 20% before scaling
    # Target entropy per action dimension (H_bar = value * dim). Options:
    #   -0.5                        default (as in HIL-SERL); unreachable for |a_res| <= 0.3
    #                               (max. entropy ~ -0.52 / dim), so alpha increases throughout training
    #   -1.0                        reachable; alpha converges
    #   -0.5 + log(residual_bound)  (= -1.70) HIL-SERL default rescaled to the residual range
    #   fixed alpha                 drop "temperature" from the learner's networks_to_update
    target_entropy_per_dim: float = -0.5
    buffer_capacity: int = 200_000
    training_starts: int = 256       # minimum size of each sampled buffer
    steps_per_update: int = 50       # learner steps between parameter broadcasts
    actor_sync_period: int = 50      # environment steps between policy syncs (App. C.2)
    checkpoint_period: int = 1000
    buffer_period: int = 1000        # actor steps between buffer dumps
    log_period: int = 10
    max_steps: int = 1_000_000


@dataclass(frozen=True)
class Config:
    task: TaskConfig
    method: MethodConfig
    train: TrainConfig = field(default_factory=TrainConfig)


def get_spaces(cfg: Config):
    """Observation / action spaces of the residual env, built without contacting the robot (learner)."""
    t = cfg.task
    h = t.obs_horizon
    obs = gym.spaces.Dict({
        "base_actions": gym.spaces.Box(-np.inf, np.inf, (h, t.action_dim), np.float32),
        "state": gym.spaces.Box(-np.inf, np.inf, (h, t.state_dim), np.float32),
        t.image_key: gym.spaces.Box(0, 255, (h, *t.image_shape), np.uint8),
    })
    return obs, gym.spaces.Box(-np.inf, np.inf, (t.action_dim,), np.float32)


def get_environment(cfg: Config, robot_url: str, *, stats_path: str = None, base_policy_addr: str = None,
                    human_reward: bool = True, listen_esc: bool = True):
    """Robot environment.

    stats_path:       normalization statistics; actions scaled to [-1, 1], states standardized
                      (None: raw actions and states, used for data collection).
    base_policy_addr: "host:port" of the base-policy server; adds `obs["base_actions"]`.
    human_reward:     sparse binary reward from the operator (X: success, Y: failure).
    """
    from diffusion_policy.policy.agent_client import DiffusionPolicyClient
    from hire_moma_launcher.wrappers.chunking import ChunkingWrapper
    from hire_moma_launcher.wrappers.normalization import load_normalizers
    from hire_moma_launcher.wrappers.act_wrappers import FlattenActWrapper
    from hire_moma_launcher.wrappers.obs_wrappers import FlattenObsWrapper
    from tiago_server.tiago_env.envs.tiago_env import TiagoEnv
    from tiago_server.tiago_env.envs.wrappers import (GripperCloseWrapper, GripperPenaltyWrapper,
                                                      HumanClassifierWrapper, ResidualObsWrapper)

    t = cfg.task
    scaler, standardizer = (None, None)
    if stats_path is not None:
        scaler, standardizer = load_normalizers(stats_path, t.grip_dims, cfg.train.action_range_margin)

    env = TiagoEnv(robot_url, obs_keys=(*t.proprio_keys, t.image_key), act_keys=t.act_keys,
                   reset_pose=t.reset_pose, max_episode_length=t.max_episode_length, listen_esc=listen_esc)
    if t.grippers_closed:
        env = GripperCloseWrapper(env, t.arms)
    env = FlattenObsWrapper(env, t.proprio_keys, {t.image_key: t.image_shape})
    env = FlattenActWrapper(env, t.act_keys, scaler)
    env = ChunkingWrapper(env, obs_horizon=t.obs_horizon)
    if base_policy_addr is not None:
        host, port = base_policy_addr.rsplit(":", 1)
        base_policy = DiffusionPolicyClient(host, int(port), act_horizon=t.bp_act_horizon, skip=t.bp_skip)
        env = ResidualObsWrapper(env, base_policy, scaler, standardizer)
    if human_reward:
        env = HumanClassifierWrapper(env)
    if t.grasp_penalty is not None:
        env = GripperPenaltyWrapper(env, penalty=t.grasp_penalty)
    return env
