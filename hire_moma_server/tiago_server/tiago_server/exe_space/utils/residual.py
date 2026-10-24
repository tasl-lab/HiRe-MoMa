"""Base + residual composition in the scaled action space.

Action layout: [arm_1 (dx dy dz drx dry drz, gripper), (arm_2 ...), base (vx vy wz)].
Gripper entries are binary commands in {-1 (close), 1 (open)}; their residuals lie in {-2, 0, 2}.
"""
from typing import Sequence

import numpy as np


def combine(base: np.ndarray, residual: np.ndarray, grip_dims: Sequence[int]) -> np.ndarray:
    assert base.shape == residual.shape, "Base and residual actions must have the same shape."
    action = base + residual
    for d in grip_dims:
        action[d] = float(int(np.clip(action[d], -1, 1)))
    return action


def deduct(action: np.ndarray, base: np.ndarray, grip_dims: Sequence[int]) -> np.ndarray:
    """Inverse of `combine`: the residual that turns `base` into `action` (e.g. a human command)."""
    assert action.shape == base.shape, "Base and residual actions must have the same shape."
    residual = action - base
    for d in grip_dims:
        residual[d] = float(int(np.clip(residual[d], -2, 2)))
    return residual
