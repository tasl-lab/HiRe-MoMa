from threading import Lock
from typing import Iterable

import gymnasium as gym
from agentlace.data.data_store import DataStoreBase

from hire_moma_launcher.data.memory_efficient_replay_buffer import MemoryEfficientReplayBuffer


class MemoryEfficientReplayBufferDataStore(MemoryEfficientReplayBuffer, DataStoreBase):
    """Thread-safe replay buffer that the learner exposes to the actor through agentlace."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, capacity: int,
                 image_keys: Iterable[str] = ("image",), **kwargs):
        MemoryEfficientReplayBuffer.__init__(self, observation_space, action_space, capacity,
                                             pixel_keys=image_keys, **kwargs)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    def insert(self, *args, **kwargs):
        with self._lock:
            super().insert(*args, **kwargs)

    def sample(self, *args, **kwargs):
        with self._lock:
            return super().sample(*args, **kwargs)

    def latest_data_id(self):
        return self._insert_index

    def get_latest_data(self, from_id: int):
        raise NotImplementedError
