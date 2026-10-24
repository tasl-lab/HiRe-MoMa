"""TIAGo-specific wrappers: gripper closing, operator reward, grasp penalty and base-policy actions."""
import copy
from typing import Sequence

import gymnasium as gym
import numpy as np

from hire_moma_launcher.wrappers.normalization import ActionScaler, StateStandardizer


class GripperCloseWrapper(gym.Wrapper):
    """Keeps the given grippers closed (also overrides human commands). Used for Sweep Litter."""

    def __init__(self, env, arms: Sequence[str]):
        super().__init__(env)
        self.arms = list(arms)

    def step(self, action, **kwargs):
        for arm in self.arms:
            action[arm][-1] = 0.0  # gripper: 0 close / 1 open
        obs, rew, done, truncated, info = self.env.step(action, **kwargs)
        if "intervene_action" in info:
            for arm in self.arms:
                info["intervene_action"][arm][-1] = 0.0
        return obs, rew, done, truncated, info


class ResidualObsWrapper(gym.Wrapper):
    """Adds `obs["base_actions"]`: the base policy's action for the current observation.

    The base policy sees raw observations; the residual policy sees scaled base actions and
    standardized states.
    """

    def __init__(self, env: gym.Env, base_policy, action_scaler: ActionScaler = None,
                 state_standardizer: StateStandardizer = None):
        super().__init__(env)
        self.base_policy = base_policy
        self.action_scaler = action_scaler
        self.state_standardizer = state_standardizer
        shape = (env.observation_space["state"].shape[0], env.action_space.shape[-1])
        dtype = env.action_space.dtype
        self.observation_space = gym.spaces.Dict({
            **env.observation_space.spaces,
            "base_actions": gym.spaces.Box(low=np.full(shape, -np.inf, dtype=dtype),
                                           high=np.full(shape, np.inf, dtype=dtype), shape=shape, dtype=dtype),
        })

    def _augment(self, obs):
        obs["base_actions"] = np.asarray(self.base_policy.predict_action(copy.deepcopy(obs)))
        if self.action_scaler is not None:
            obs["base_actions"] = self.action_scaler.scale(obs["base_actions"])
        if self.state_standardizer is not None:
            obs["state"] = self.state_standardizer.standardize(obs["state"])
        return obs

    def step(self, action, **kwargs):
        obs, reward, done, trunc, info = self.env.step(action, **kwargs)
        return self._augment(obs), reward, done, trunc, info

    def reset(self, **kwargs):
        self.base_policy.reset()
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs), info


class HumanClassifierWrapper(gym.Wrapper):
    """Sparse reward given by the operator on the Quest controller.

    X toggles "success" (reward 1); the episode ends after `rew_threshold` successful steps.
    rew_threshold=1 (default) gives the binary terminal reward of Sec. 3.
    Y ends the episode as a failure.
    """

    def __init__(self, env: gym.Env, rew_threshold: int = 1):
        super().__init__(env)
        self.rew_threshold = rew_threshold
        self._reset_state()

    def _reset_state(self):
        self.rew, self.rew_counter = False, 0
        self.prev_x, self.prev_y = False, False

    def step(self, action, **kwargs):
        obs, _, done, truncated, info = self.env.step(action, **kwargs)
        buttons = info.get("teleop_buttons", {})
        x, y = buttons.get("X", False), buttons.get("Y", False)
        if y and not self.prev_y:
            done = True
        if x and not self.prev_x:
            self.rew = not self.rew
        if not done and self.rew:
            self.rew_counter += 1
            if self.rew_counter >= self.rew_threshold:
                done = True
        info["succeed"] = self.rew
        self.prev_x, self.prev_y = x, y
        return obs, self.rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_state()
        info["succeed"] = False
        return obs, info


class GripperPenaltyWrapper(gym.Wrapper):
    """Single-arm tasks: `info["grasp_penalty"]` when the gripper command flips the gripper state.

    Only the Flat Residual baseline uses it (reward of its grasp critic).
    """

    def __init__(self, env, penalty: float = -0.02):
        super().__init__(env)
        self.penalty = penalty
        self.last_gripper_pos = -1e-5

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 7]  # state = [arm pos(3), quat(4), gripper(1), ...]
        return obs, info

    def step(self, action, **kwargs):
        obs, reward, terminated, truncated, info = self.env.step(action, **kwargs)
        if "intervene_action" in info:
            action = info["intervene_action"]
        flips = (action[6] < 0.5 and self.last_gripper_pos > 0.0) or (action[6] > 0.5 and self.last_gripper_pos < 0.0)
        info["grasp_penalty"] = self.penalty if flips else 0.0
        self.last_gripper_pos = obs["state"][0, 7]
        return obs, reward, terminated, truncated, info
