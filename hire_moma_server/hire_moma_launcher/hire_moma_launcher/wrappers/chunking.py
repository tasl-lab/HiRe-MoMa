from collections import deque

import gymnasium as gym
import numpy as np


def _space_stack(space: gym.Space, repeat: int):
    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(low=np.repeat(space.low[None], repeat, axis=0),
                              high=np.repeat(space.high[None], repeat, axis=0), dtype=space.dtype)
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict({k: _space_stack(v, repeat) for k, v in space.spaces.items()})
    raise TypeError()


class ChunkingWrapper(gym.Wrapper):
    """Stacks the last `obs_horizon` observations along a new leading axis."""

    def __init__(self, env: gym.Env, obs_horizon: int = 1):
        super().__init__(env)
        self.obs_horizon = obs_horizon
        self.current_obs = deque(maxlen=obs_horizon)
        self.observation_space = _space_stack(env.observation_space, obs_horizon)

    def _stacked(self):
        keys = self.current_obs[0].keys()
        return {k: np.stack([o[k] for o in self.current_obs]) for k in keys}

    def step(self, action, **kwargs):
        obs, reward, done, trunc, info = self.env.step(action, **kwargs)
        self.current_obs.append(obs)
        return self._stacked(), reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs.extend([obs] * self.obs_horizon)
        return self._stacked(), info
