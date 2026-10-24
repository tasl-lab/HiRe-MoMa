import copy
from typing import Iterable, Optional, Tuple

import gymnasium as gym
import numpy as np
from flax.core import frozen_dict
from gymnasium.spaces import Box

from hire_moma_launcher.data.dataset import DatasetDict, _sample
from hire_moma_launcher.data.replay_buffer import ReplayBuffer


class MemoryEfficientReplayBuffer(ReplayBuffer):
    """Stores every image frame once; obs/next_obs image stacks are rebuilt at sampling time."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        include_grasp_penalty: bool = False,
    ):
        self.pixel_keys = pixel_keys
        observation_space = copy.deepcopy(observation_space)

        self._num_stack = None
        for pixel_key in self.pixel_keys:
            pixel_obs_space = observation_space.spaces[pixel_key]
            if self._num_stack is None:
                self._num_stack = pixel_obs_space.shape[0]
            else:
                assert self._num_stack == pixel_obs_space.shape[0]
            self._unstacked_dim_size = pixel_obs_space.shape[-1]
            observation_space.spaces[pixel_key] = Box(
                low=pixel_obs_space.low[0], high=pixel_obs_space.high[0], dtype=pixel_obs_space.dtype)

        next_observation_space_dict = copy.deepcopy(observation_space.spaces)
        for pixel_key in self.pixel_keys:
            next_observation_space_dict.pop(pixel_key)
        next_observation_space = gym.spaces.Dict(next_observation_space_dict)

        self._first = True
        self._is_correct_index = np.full(capacity, False, dtype=bool)

        super().__init__(
            observation_space,
            action_space,
            capacity,
            next_observation_space=next_observation_space,
            include_grasp_penalty=include_grasp_penalty,
        )

    def insert(self, data_dict: DatasetDict):
        # Circular-buffer edge case: re-insert the frames needed to rebuild the last stacks.
        if self._insert_index == 0 and self._capacity == len(self) and not self._first:
            for indx in np.arange(len(self) - self._num_stack, len(self)):
                element = super().sample(1, indx=indx)
                self._is_correct_index[self._insert_index] = False
                super().insert(element)

        data_dict = data_dict.copy()
        data_dict["observations"] = data_dict["observations"].copy()
        data_dict["next_observations"] = data_dict["next_observations"].copy()

        obs_pixels, next_obs_pixels = {}, {}
        for pixel_key in self.pixel_keys:
            obs_pixels[pixel_key] = data_dict["observations"].pop(pixel_key)
            next_obs_pixels[pixel_key] = data_dict["next_observations"].pop(pixel_key)

        # First step of an episode: store the initial frames of the stack.
        if self._first:
            for i in range(self._num_stack):
                for pixel_key in self.pixel_keys:
                    data_dict["observations"][pixel_key] = obs_pixels[pixel_key][i]
                self._is_correct_index[self._insert_index] = False
                super().insert(data_dict)

        for pixel_key in self.pixel_keys:
            data_dict["observations"][pixel_key] = next_obs_pixels[pixel_key][-1]
        self._first = data_dict["dones"]
        self._is_correct_index[self._insert_index] = True
        super().insert(data_dict)

        # Indices whose stacks would be incomplete must not be sampled.
        for i in range(self._num_stack):
            self._is_correct_index[(self._insert_index + i) % len(self)] = False

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        pack_obs_and_next_obs: bool = False,
    ) -> frozen_dict.FrozenDict:
        """With `pack_obs_and_next_obs`, images hold num_stack + 1 frames (obs and next obs share frames)."""
        if indx is not None:
            raise NotImplementedError()
        indx = self.np_random.integers(len(self), size=batch_size)
        for i in range(batch_size):
            while not self._is_correct_index[indx[i]]:
                indx[i] = self.np_random.integers(len(self))

        if keys is None:
            keys = self.dataset_dict.keys()
        else:
            assert "observations" in keys
        keys = list(keys)
        keys.remove("observations")
        batch = super().sample(batch_size, keys, indx).unfreeze()

        obs_keys = [k for k in self.dataset_dict["observations"].keys() if k not in self.pixel_keys]
        batch["observations"] = {k: _sample(self.dataset_dict["observations"][k], indx) for k in obs_keys}

        for pixel_key in self.pixel_keys:
            obs_pixels = self.dataset_dict["observations"][pixel_key]
            obs_pixels = np.lib.stride_tricks.sliding_window_view(obs_pixels, self._num_stack + 1, axis=0)
            obs_pixels = obs_pixels[indx - self._num_stack]
            obs_pixels = obs_pixels.transpose((0, 4, 1, 2, 3))  # (B, H, W, C, T) -> (B, T, H, W, C)
            if pack_obs_and_next_obs:
                batch["observations"][pixel_key] = obs_pixels
            else:
                batch["observations"][pixel_key] = obs_pixels[:, :-1, ...]
                if "next_observations" in keys:
                    batch["next_observations"][pixel_key] = obs_pixels[:, 1:, ...]

        return frozen_dict.freeze(batch)
