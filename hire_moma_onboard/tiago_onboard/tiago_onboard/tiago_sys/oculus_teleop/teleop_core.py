"""Teleoperation interface (adapted from TeleMoMa)."""
from dataclasses import dataclass, field

import numpy as np

from tiago_onboard.tiago_sys.utils.general_utils import AttrDict


@dataclass
class TeleopAction(AttrDict):
    left: np.ndarray = field(default_factory=lambda: np.r_[np.zeros(6), np.ones(1)])
    right: np.ndarray = field(default_factory=lambda: np.r_[np.zeros(6), np.ones(1)])
    base: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torso: float = field(default_factory=lambda: 0.)
    extra: dict = field(default_factory=dict)


@dataclass
class TeleopObservation(AttrDict):
    left: np.ndarray = field(default_factory=lambda: np.r_[np.zeros(6), np.ones(2)])
    right: np.ndarray = field(default_factory=lambda: np.r_[np.zeros(6), np.ones(2)])
    base: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torso: float = field(default_factory=lambda: 0.)
    extra: dict = field(default_factory=dict)


class BaseTeleopInterface:
    """Teleoperation device: a background thread updates the device state, `get_action` maps it to robot actions."""

    def __init__(self, *args, **kwargs) -> None:
        self.raw_data = {}

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def reset_state(self) -> None:
        raise NotImplementedError

    def _update_internal_state(self) -> None:
        raise NotImplementedError

    def get_action(self, obs: TeleopObservation, is_filter=False) -> TeleopAction:
        raise NotImplementedError

    def get_default_action(self) -> TeleopAction:
        return TeleopAction()
