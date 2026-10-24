import collections
from typing import Optional, Union

import gymnasium as gym
import jax
import numpy as np

from hire_moma_launcher.data.dataset import Dataset, DatasetDict


def _init_replay_dict(obs_space: gym.Space, capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    if isinstance(obs_space, gym.spaces.Dict):
        return {k: _init_replay_dict(v, capacity) for k, v in obs_space.spaces.items()}
    raise TypeError()


def _insert_recursively(dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        include_grasp_penalty: bool = False,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space
        dataset_dict = dict(
            observations=_init_replay_dict(observation_space, capacity),
            next_observations=_init_replay_dict(next_observation_space, capacity),
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )
        if include_grasp_penalty:
            dataset_dict["grasp_penalty"] = np.empty((capacity,), dtype=np.float32)
        super().__init__(dataset_dict)
        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)
        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}, device=None):
        """Prefetches `queue_size` batches to `device`."""
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                queue.append(jax.device_put(self.sample(**sample_args), device=device))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)
