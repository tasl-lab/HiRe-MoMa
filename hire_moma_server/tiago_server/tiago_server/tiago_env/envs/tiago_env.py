"""Gym client of the onboard robot server (`hire_moma_onboard/.../tiago_env/envs/tiago_env.py`)."""
import time
from collections import OrderedDict
from typing import Dict, Sequence

import gymnasium as gym
import numpy as np
import requests

from tiago_server.tiago_env.utils.flask_comm import decode4json, encode2json, reconstruct_space_dict


class TiagoEnv(gym.Env):
    """Dict observations and delta actions at `hz`.

    The onboard server arbitrates shared control: a Quest 3 takeover overrides the action and the
    executed human action is returned in `info["intervene_action"]`. `reset` moves arms, torso and
    head to `reset_pose`; the operator then places the base and presses A to start the episode.
    """

    def __init__(self, url: str, obs_keys: Sequence[str], act_keys: Sequence[str], reset_pose: Dict,
                 max_episode_length: int, hz: int = 10, listen_esc: bool = True):
        self.url = url.rstrip("/") + "/"
        self.obs_keys = list(obs_keys)
        self.act_keys = list(act_keys)
        self.reset_pose = reset_pose
        self.max_episode_length = max_episode_length
        self.hz = hz
        self.observation_space = self._space("tiago_get_observation_space", self.obs_keys)
        self.action_space = self._space("tiago_get_action_space", self.act_keys)
        self._steps = 0
        self.terminate = False
        if listen_esc:  # ESC ends the current episode
            from pynput import keyboard

            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True

            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()

    def _post(self, route, **payload):
        return requests.post(self.url + route, json=payload or None).json()

    def _space(self, route, keys):
        space = reconstruct_space_dict(self._post(route))
        return gym.spaces.Dict(OrderedDict((k, space[k]) for k in keys if k in space.spaces))

    def _robot_step(self, action, is_only_teleop):
        start = time.time()
        resp = self._post("tiago_step", action=encode2json({k: action[k] for k in self.act_keys if k in action}),
                          is_only_teleop=is_only_teleop)
        obs, info = decode4json(resp["obs"]), decode4json(resp["info"])
        time.sleep(max(0.0, 1.0 / self.hz - (time.time() - start)))
        return {k: obs[k] for k in self.obs_keys if k in obs}, info

    def step(self, action, is_only_teleop: bool = False):
        """`is_only_teleop=True`: ignore `action` and execute only the operator's command."""
        obs, info = self._robot_step(action, is_only_teleop)
        self._steps += 1
        done = self._steps >= self.max_episode_length or self.terminate
        info["succeed"] = 0
        return obs, 0, done, False, info

    def reset(self, **kwargs):
        self._post("tiago_reset", reset_pose=encode2json(self.reset_pose))
        print("[TiagoEnv] Reset done. Place the robot with the VR controller, then press A.")
        idle = {k: np.zeros(self.action_space[k].shape, dtype=np.float32) for k in self.act_keys}
        while True:
            obs, info = self._robot_step(idle, is_only_teleop=True)
            if info.get("teleop_buttons", {}).get("A", False):
                break
        self._steps = 0
        self.terminate = False
        return obs, {"succeed": False}

    def close(self):
        if hasattr(self, "listener"):
            self.listener.stop()
        requests.post(self.url + "tiago_close")
