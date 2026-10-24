#!/usr/bin/env python3
"""Record teleoperated demonstrations for training the base policy (one .pkl per successful episode).

  python -m tiago_server.exe_space.data_collection.record_imitation_demos --task place_only --robot_url http://<onboard-ip>:1234 \
      --out_dir data/place_only/imitation --episodes 80

Quest 3: teleoperate the robot, press X to mark the episode as successful, Y to finish it.
Only episodes marked successful are saved.
"""
import copy
import datetime
import os
import pickle as pkl

import numpy as np
from absl import app, flags
from tqdm import tqdm

from tiago_server.exe_space.experiments.config import get_environment
from tiago_server.exe_space.experiments.mappings import get_config

FLAGS = flags.FLAGS
flags.DEFINE_string("task", None, "place_only | sweep_litter | pick_place")
flags.DEFINE_string("robot_url", "http://localhost:1234", "Onboard server.")
flags.DEFINE_string("out_dir", None, "Output folder.")
flags.DEFINE_integer("episodes", 80, "Number of successful episodes to record.")


def main(_):
    cfg = get_config(FLAGS.task, "hire_moma")
    env = get_environment(cfg, FLAGS.robot_url, human_reward=False)  # raw actions, no base policy
    os.makedirs(FLAGS.out_dir, exist_ok=True)

    saved = 0
    pbar = tqdm(total=FLAGS.episodes, desc="successful episodes")
    while saved < FLAGS.episodes:
        obs, _ = env.reset()
        trajectory, success, prev_x, prev_y = [], False, False, False
        while True:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            next_obs, _, done, truncated, info = env.step(action, is_only_teleop=True)
            action = info.get("intervene_action", action)
            buttons = info.get("teleop_buttons", {})
            x, y = buttons.get("X", False), buttons.get("Y", False)
            if x and not prev_x:
                success = not success
            trajectory.append(copy.deepcopy(dict(observation=obs, action=action, reward=float(success))))
            obs = next_obs
            if (y and not prev_y) or done or truncated:
                break
            prev_x, prev_y = x, y

        if success and trajectory:
            saved += 1
            pbar.update(1)
            stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open(os.path.join(FLAGS.out_dir, f"{FLAGS.task}_{saved:03d}_{stamp}.pkl"), "wb") as f:
                pkl.dump(trajectory, f)
    env.close()


if __name__ == "__main__":
    app.run(main)
