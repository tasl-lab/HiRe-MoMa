from collections import OrderedDict
from typing import Dict, Sequence

import cv2
import gymnasium as gym
from gymnasium.spaces import flatten, flatten_space


class FlattenObsWrapper(gym.Wrapper):
    """Dict obs -> {"state": flat proprio, <image keys>: resized images}."""

    def __init__(self, env, proprio_keys: Sequence[str], image_shapes: Dict[str, tuple]):
        super().__init__(env)
        self.image_shapes = image_shapes
        self.proprio_space = gym.spaces.Dict(OrderedDict((k, env.observation_space[k]) for k in proprio_keys))
        image_spaces = {k: gym.spaces.Box(low=env.observation_space[k].low.min(),
                                          high=env.observation_space[k].high.max(),
                                          shape=shape, dtype=env.observation_space[k].dtype)
                        for k, shape in image_shapes.items()}
        self.observation_space = gym.spaces.Dict({"state": flatten_space(self.proprio_space), **image_spaces})

    def observation(self, obs):
        out = {"state": flatten(self.proprio_space, {k: obs[k] for k in self.proprio_space.spaces})}
        for key, (h, w, _) in self.image_shapes.items():
            out[key] = cv2.resize(obs[key], (w, h), interpolation=cv2.INTER_LINEAR)
        return out

    def step(self, action, **kwargs):
        obs, reward, done, trunc, info = self.env.step(action, **kwargs)
        return self.observation(obs), reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
