"""Action scaling to [-1, 1] and state standardization, both fitted on demonstration data."""
import glob
import json
import os
import pickle as pkl
from typing import List, Sequence

import numpy as np


class ActionScaler:
    """Min-max scaling to [-1, 1].

    Continuous ranges are widened by `action_scale` on each side so that base + residual
    can go slightly beyond the demonstrated range. Gripper dims are binarized: {0, 1} -> {-1, 1}.
    """

    def __init__(self, action_min, action_max, action_scale: float = 0.2,
                 grip_dims: Sequence[int] = (), min_range_per_dim: float = 1e-2):
        action_min = np.asarray(action_min, dtype=np.float32)
        action_max = np.asarray(action_max, dtype=np.float32)
        self.grip_dims = list(grip_dims)
        mid = (action_min + action_max) / 2.0
        half_range = np.maximum((action_max - action_min) / 2.0, min_range_per_dim / 2.0)
        is_grip = np.zeros_like(action_min, dtype=bool)
        is_grip[self.grip_dims] = True
        half_range = half_range * np.where(is_grip, 1.0, 1.0 + action_scale)
        self.min = mid - half_range
        self.max = mid + half_range
        self._range = np.maximum(self.max - self.min, 1e-8)

    def scale(self, action: np.ndarray) -> np.ndarray:
        action = np.array(action, dtype=np.float32)
        for d in self.grip_dims:
            action[..., d] = (action[..., d] >= 0.5).astype(action.dtype)
        action = np.clip(action, self.min, self.max)
        return 2.0 * (action - self.min) / self._range - 1.0

    def unscale(self, scaled: np.ndarray) -> np.ndarray:
        action = self.min + (np.clip(scaled, -1.0, 1.0) + 1.0) * self._range / 2.0
        for d in self.grip_dims:
            action[..., d] = (action[..., d] >= 0.5).astype(action.dtype)
        return action


class StateStandardizer:
    def __init__(self, mean, std, min_std: float = 1e-2):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), min_std)

    def standardize(self, state: np.ndarray) -> np.ndarray:
        return (state - self.mean) / self.std


# ---------------------------------------------------------------- statistics
def load_transitions(dirs: List[str]) -> list:
    """All transitions from every `*.pkl` in `dirs` (imitation episodes or warm-up demos)."""
    transitions = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.pkl"))):
            with open(path, "rb") as f:
                transitions.extend(pkl.load(f))
    return transitions


def compute_stats(transitions: list) -> dict:
    """Action min/max (0.25 / 99.75 percentiles) and state mean/std, on raw (unscaled) data."""
    actions = np.array([t["actions"] if "actions" in t else t["action"] for t in transitions])
    states = np.array([t["observations"]["state"] if "observations" in t else t["observation"]["state"]
                       for t in transitions])
    return {
        "action": {"min": np.percentile(actions, 0.25, axis=0).tolist(),
                   "max": np.percentile(actions, 99.75, axis=0).tolist()},
        "state": {"mean": states.mean(axis=(0, 1)).tolist(), "std": states.std(axis=(0, 1)).tolist()},
    }


def save_stats(stats: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_normalizers(path: str, grip_dims: Sequence[int], action_scale: float = 0.2):
    with open(path) as f:
        stats = json.load(f)
    return (ActionScaler(stats["action"]["min"], stats["action"]["max"], action_scale=action_scale, grip_dims=grip_dims),
            StateStandardizer(stats["state"]["mean"], stats["state"]["std"]))
