from collections import OrderedDict
from typing import Sequence

import gymnasium as gym
from gymnasium.spaces import flatten, flatten_space, unflatten

from hire_moma_launcher.wrappers.normalization import ActionScaler


class FlattenActWrapper(gym.Wrapper):
    """Flat action vector in [-1, 1] (if a scaler is given) <-> dict action of the robot.

    Human interventions in `info["intervene_action"]` are converted the same way.
    """

    def __init__(self, env, act_keys: Sequence[str], action_scaler: ActionScaler = None):
        super().__init__(env)
        self.act_space = gym.spaces.Dict(OrderedDict((k, env.action_space.spaces[k]) for k in act_keys))
        self.action_space = flatten_space(self.act_space)
        self.action_scaler = action_scaler

    def step(self, action, **kwargs):
        assert action.shape == self.action_space.shape, f"expected {self.action_space.shape}, got {action.shape}"
        if self.action_scaler is not None:
            action = self.action_scaler.unscale(action)
        obs, reward, done, trunc, info = self.env.step(unflatten(self.act_space, action), **kwargs)
        if isinstance(info.get("intervene_action"), dict):
            intervention = flatten(self.act_space, {k: v for k, v in info["intervene_action"].items()
                                                    if k in self.act_space.spaces})
            if self.action_scaler is not None:
                intervention = self.action_scaler.scale(intervention)
            info["intervene_action"] = intervention
        return obs, reward, done, trunc, info
