#!/usr/bin/env python3
"""Record the warm-up buffer: successful base-policy rollouts (with optional human interventions).

  python -m tiago_server.exe_space.data_collection.record_demos --task place_only \
      --robot_url http://<onboard-ip>:1234 --base_policy localhost:5555 --out_dir data/place_only/warmup

The base-policy server must be running. Press X when the task succeeds, Y to abort.
Transitions are stored raw (unscaled); the learner normalizes them when loading.
"""
import copy
import datetime
import os
import pickle as pkl

from absl import app, flags
from tqdm import tqdm

from tiago_server.exe_space.experiments.config import get_environment
from tiago_server.exe_space.experiments.mappings import get_config

FLAGS = flags.FLAGS
flags.DEFINE_string("task", None, "place_only | sweep_litter | pick_place")
flags.DEFINE_string("robot_url", "http://localhost:1234", "Onboard server.")
flags.DEFINE_string("base_policy", "localhost:5555", "Base-policy server host:port.")
flags.DEFINE_string("out_dir", None, "Output folder.")
flags.DEFINE_integer("min_transitions", 500, "Record successful episodes until this many transitions (paper: ~500-600).")


def main(_):
    cfg = get_config(FLAGS.task, "hire_moma")
    env = get_environment(cfg, FLAGS.robot_url, base_policy_addr=FLAGS.base_policy)
    grip = list(cfg.task.grip_dims)

    transitions, trajectory, saved = [], [], 0
    pbar = tqdm(total=FLAGS.min_transitions, desc="transitions")
    obs, _ = env.reset()
    while len(transitions) < FLAGS.min_transitions:
        action = obs["base_actions"][0].copy()
        action[grip] = (action[grip] >= 0.5).astype(action.dtype)  # gripper command: 0 close / 1 open
        next_obs, reward, done, truncated, info = env.step(action)
        if "intervene_action" in info:  # store what the robot executed
            action = info.pop("intervene_action")
        transition = dict(observations=obs, actions=action, next_observations=next_obs,
                          rewards=reward, masks=1.0 - done, dones=done)
        if "grasp_penalty" in info:
            transition["grasp_penalty"] = info["grasp_penalty"]
        trajectory.append(copy.deepcopy(transition))
        obs = next_obs
        if done or truncated:
            if info["succeed"]:
                transitions.extend(trajectory)
                saved += 1
                pbar.update(len(trajectory))
            trajectory = []
            obs, _ = env.reset()

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(FLAGS.out_dir, f"{FLAGS.task}_{saved}_episodes_{stamp}.pkl")
    with open(path, "wb") as f:
        pkl.dump(transitions, f)
    print(f"Saved {len(transitions)} transitions to {path}")


if __name__ == "__main__":
    app.run(main)
