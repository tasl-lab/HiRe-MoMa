#!/usr/bin/env python3
"""Asynchronous actor / learner (and evaluation) for HiRe-MoMa and the baselines.

Run the learner and the actor in two terminals on the workstation:

  python -m tiago_server.exe_space.train_residual --task place_only --method hire_moma --learner \
      --exp_dir runs/place_only/hire_moma --demo_dir data/place_only/warmup
  python -m tiago_server.exe_space.train_residual --task place_only --method hire_moma --actor \
      --exp_dir runs/place_only/hire_moma --robot_url http://<onboard-ip>:1234

Evaluate a checkpoint (or the base policy alone with --base_only):

  python -m tiago_server.exe_space.train_residual --task place_only --method hire_moma --actor \
      --exp_dir runs/place_only/hire_moma --robot_url http://<onboard-ip>:1234 --eval_step 5000
"""
import copy
import dataclasses
import glob
import os
import pickle as pkl
import time

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
from gymnasium.wrappers import RecordEpisodeStatistics
from natsort import natsorted

from hire_moma_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from hire_moma_launcher.utils.launcher import make_agent, make_trainer_config, make_wandb_logger, networks_to_update
from hire_moma_launcher.utils.timer_utils import Timer
from hire_moma_launcher.utils.train_utils import concat_batches
from hire_moma_launcher.wrappers.normalization import load_normalizers
from tiago_server.exe_space.experiments.config import get_environment, get_spaces
from tiago_server.exe_space.experiments.mappings import get_config
from tiago_server.exe_space.utils.residual import combine, deduct

FLAGS = flags.FLAGS
flags.DEFINE_string("task", None, "place_only | sweep_litter | pick_place")
flags.DEFINE_string("method", "hire_moma", "hire_moma | hire_moma_no_align | decomposed | flat_residual | hg_dagger")
flags.DEFINE_string("exp_dir", None, "Checkpoints and buffer dumps are written here.")
flags.DEFINE_boolean("learner", False, "Run the learner.")
flags.DEFINE_boolean("actor", False, "Run the actor (connects to the robot).")
flags.DEFINE_string("learner_ip", "localhost", "IP of the learner (actor side).")
flags.DEFINE_string("robot_url", "http://localhost:1234", "Onboard server (actor side).")
flags.DEFINE_string("base_policy", "localhost:5555", "Base-policy server host:port (actor side).")
flags.DEFINE_string("stats", None, "Normalization stats json. Default: stats/<task>.json")
flags.DEFINE_multi_string("demo_dir", None, "Warm-up demo folders (learner side).")
flags.DEFINE_integer("eval_step", 0, "Evaluate this checkpoint instead of training (actor side).")
flags.DEFINE_integer("eval_episodes", 20, "Number of evaluation episodes.")
flags.DEFINE_boolean("base_only", False, "Evaluate the base policy without the residual.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("debug", False, "Disable wandb.")
DEBUG_OVERRIDES = ("batch_size", "training_starts", "max_steps", "steps_per_update", "checkpoint_period",
                   "buffer_period")
for _name in DEBUG_OVERRIDES:
    flags.DEFINE_integer(_name, None, f"Override TrainConfig.{_name} (debugging).")


def print_green(x):
    print(f"\033[92m{x}\033[00m")


# ================================================================ actor
def evaluate(cfg, agent, env, rng):
    if not FLAGS.base_only:
        state = checkpoints.restore_checkpoint(os.path.abspath(FLAGS.exp_dir), agent.get_state(), step=FLAGS.eval_step)
        agent = agent.replace_with_state(state)
    grip = cfg.task.grip_dims
    successes, lengths, transitions = 0, [], []
    for episode in range(FLAGS.eval_episodes):
        obs, _ = env.reset()
        for step in tqdm.trange(cfg.task.max_episode_length, dynamic_ncols=True):
            base = obs["base_actions"][0]
            if FLAGS.base_only:
                residual = np.zeros_like(base)
            else:
                rng, key = jax.random.split(rng)
                residual = np.asarray(jax.device_get(agent.sample_actions(jax.device_put(obs), seed=key, argmax=True)))
            actions = combine(base, residual, grip)
            next_obs, reward, done, truncated, info = env.step(actions)
            transitions.append(copy.deepcopy(dict(observations=obs, residual_actions=residual, actions=actions,
                                                  rewards=reward, masks=1.0 - done, dones=done)))
            obs = next_obs
            if done or truncated:
                successes += int(reward)
                if reward:
                    lengths.append(step + 1)
                break
        print(f"episode {episode + 1}: success={bool(reward)}  ({successes}/{episode + 1})")
    out = os.path.join(FLAGS.exp_dir, f"eval_{'base' if FLAGS.base_only else FLAGS.eval_step}.pkl")
    with open(out, "wb") as f:
        pkl.dump(transitions, f)
    print_green(f"success rate {successes / FLAGS.eval_episodes:.2f}, "
                f"mean steps (successes) {np.mean(lengths) if lengths else float('nan'):.1f}; saved {out}")


def actor(cfg, agent, env, rng):
    from agentlace.data.data_store import QueuedDataStore
    from agentlace.trainer import TrainerClient

    if FLAGS.eval_step or FLAGS.base_only:
        return evaluate(cfg, agent, env, rng)

    grip = cfg.task.grip_dims
    is_bc = cfg.method.agent == "hg_dagger"
    buffer_files = natsorted(glob.glob(os.path.join(FLAGS.exp_dir, "buffer/*.pkl")))
    start_step = int(os.path.basename(buffer_files[-1])[12:-4]) + 1 if buffer_files else 0

    data_store, intvn_store = QueuedDataStore(50000), QueuedDataStore(50000)
    client = TrainerClient("actor_env", FLAGS.learner_ip, make_trainer_config(),
                           data_stores={"actor_env": data_store, "actor_env_intvn": intvn_store},
                           wait_for_server=True, timeout_ms=3000)

    latest = {}

    def receive_params(params):  # applied every `actor_sync_period` environment steps
        latest["params"] = params

    client.recv_network_callback(receive_params)

    transitions, intvn_transitions = [], []
    obs, _ = env.reset()
    timer = Timer()
    running_return, intervened, n_interventions, n_intervention_steps = 0.0, False, 0, 0
    pbar = tqdm.tqdm(range(start_step, cfg.train.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")
        if step % cfg.train.actor_sync_period == 0 and "params" in latest:
            agent = agent.with_params(latest.pop("params"))
        with timer.context("sample_actions"):
            base = obs["base_actions"][0]
            rng, key = jax.random.split(rng)
            residual = np.asarray(jax.device_get(agent.sample_actions(jax.device_put(obs), seed=key, argmax=is_bc)))
            actions = combine(base, residual, grip)

        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            info.pop("teleop_buttons", None)
            if "intervene_action" in info:  # takeover: a_res = a_h - a_base (Alg. 1, l.9)
                residual = deduct(info.pop("intervene_action"), base, grip)
                actions = combine(base, residual, grip)
                n_intervention_steps += 1
                n_interventions += int(not intervened)
                intervened = True
            else:
                intervened = False

            running_return += reward
            transition = dict(observations=obs, actions=residual if is_bc else actions, next_observations=next_obs,
                              rewards=reward, masks=1.0 - done, dones=done)
            if "grasp_penalty" in info:
                transition["grasp_penalty"] = info["grasp_penalty"]
            data_store.insert(transition)  # Alg. 1, l.16-17: all transitions -> D_replay, interventions also -> D_expert
            transitions.append(copy.deepcopy(transition))
            if intervened:
                intvn_store.insert(transition)
                intvn_transitions.append(copy.deepcopy(transition))
            obs = next_obs

            if done or truncated:
                info["episode"]["intervention_count"] = n_interventions
                info["episode"]["intervention_steps"] = n_intervention_steps
                info["succeed"] = int(info.get("succeed", 0))
                client.request("send-stats", {"environment": info})
                pbar.set_description(f"last return: {running_return}")
                running_return, intervened, n_interventions, n_intervention_steps = 0.0, False, 0, 0
                client.update()
                obs, _ = env.reset()

        if step > 0 and step % cfg.train.buffer_period == 0:  # buffer dumps for resuming the learner
            for name, data in (("buffer", transitions), ("demo_buffer", intvn_transitions)):
                os.makedirs(os.path.join(FLAGS.exp_dir, name), exist_ok=True)
                with open(os.path.join(FLAGS.exp_dir, name, f"transitions_{step}.pkl"), "wb") as f:
                    pkl.dump(data, f)
            transitions, intvn_transitions = [], []

        timer.tock("total")
        if step % cfg.train.log_period == 0:
            client.request("send-stats", {"timer": {"actor": timer.get_average_times()}})


# ================================================================ learner
def learner(cfg, agent, replay_buffer, intvn_buffer, logger):
    from agentlace.trainer import TrainerServer

    latest = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.exp_dir))
    step = int(os.path.basename(latest)[11:]) + 1 if latest else 0
    is_bc = cfg.method.agent == "hg_dagger"

    def stats_callback(type: str, payload: dict) -> dict:
        assert type == "send-stats", f"Invalid request type: {type}"
        if logger is not None:
            logger.log(payload, step=step)
        return {}

    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", intvn_buffer)
    server.start(threaded=True)

    def n_ready():  # training starts once every sampled buffer holds `training_starts` transitions
        return len(intvn_buffer) if is_bc else min(len(replay_buffer), len(intvn_buffer))

    starts = cfg.train.training_starts
    pbar = tqdm.tqdm(total=starts, desc="Waiting for data")
    while n_ready() < starts:
        pbar.update(min(n_ready(), starts) - pbar.n)
        time.sleep(1)
    pbar.close()
    server.publish_network(agent.get_params())
    print_green("sent initial network to actor")

    def iterator(buffer, batch_size):
        return buffer.get_iterator(sample_args={"batch_size": batch_size, "pack_obs_and_next_obs": True})

    if is_bc:
        intvn_iter = iterator(intvn_buffer, cfg.train.batch_size)
    else:  # balanced sampling: half replay buffer, half intervention buffer
        replay_iter = iterator(replay_buffer, cfg.train.batch_size // 2)
        intvn_iter = iterator(intvn_buffer, cfg.train.batch_size // 2)
        critic_nets, all_nets = networks_to_update(agent)

    timer = Timer()
    for step in tqdm.tqdm(range(step, cfg.train.max_steps), dynamic_ncols=True, desc="learner"):
        if is_bc:
            with timer.context("train"):
                agent, info = agent.update(next(intvn_iter))
        else:
            for _ in range(cfg.train.cta_ratio - 1):  # critic-only updates
                with timer.context("sample"):
                    batch = concat_batches(next(replay_iter), next(intvn_iter))
                with timer.context("train_critics"):
                    agent, info = agent.update(batch, networks_to_update=critic_nets)
            with timer.context("train"):
                batch = concat_batches(next(replay_iter), next(intvn_iter))
                agent, info = agent.update(batch, networks_to_update=all_nets)

        if step > 0 and step % cfg.train.steps_per_update == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.get_params())
        if step % cfg.train.log_period == 0 and logger is not None:
            logger.log(info, step=step)
            logger.log({"timer": {"learner": timer.get_average_times()}}, step=step)
        if step > 0 and step % cfg.train.checkpoint_period == 0:
            checkpoints.save_checkpoint(os.path.abspath(FLAGS.exp_dir), agent.get_state(), step=step, keep=100)


def load_demos(cfg, paths, buffer, scaler, standardizer):
    """Warm-up demos are stored raw; scale/standardize them like the actor does online."""
    is_bc = cfg.method.agent == "hg_dagger"
    for path in paths:
        with open(path, "rb") as f:
            for t in pkl.load(f):
                t["actions"] = scaler.scale(t["actions"])
                for key in ("observations", "next_observations"):
                    t[key]["base_actions"] = scaler.scale(t[key]["base_actions"])
                    t[key]["state"] = standardizer.standardize(t[key]["state"])
                if is_bc:  # HG-DAgger regresses residuals: label = executed - base
                    t["actions"] = deduct(t["actions"], t["observations"]["base_actions"][0], cfg.task.grip_dims)
                buffer.insert(t)


def load_dumps(path_pattern, buffer):
    for path in glob.glob(path_pattern):
        with open(path, "rb") as f:
            for t in pkl.load(f):
                buffer.insert(t)


# ================================================================ main
def main(_):
    assert FLAGS.learner != FLAGS.actor, "Pass exactly one of --learner / --actor."
    cfg = get_config(FLAGS.task, FLAGS.method)
    overrides = {k: getattr(FLAGS, k) for k in DEBUG_OVERRIDES if getattr(FLAGS, k)}
    if overrides:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, **overrides))
    stats_path = FLAGS.stats or os.path.join("stats", f"{cfg.task.name}.json")
    os.makedirs(FLAGS.exp_dir, exist_ok=True)

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)
    obs_space, act_space = get_spaces(cfg)
    agent = make_agent(cfg, obs_space.sample(), act_space.sample(), seed=FLAGS.seed)
    agent = jax.device_put(jax.tree.map(jnp.array, agent))

    if checkpoints.latest_checkpoint(os.path.abspath(FLAGS.exp_dir)) and not FLAGS.eval_step:
        agent = agent.replace_with_state(checkpoints.restore_checkpoint(os.path.abspath(FLAGS.exp_dir), agent.get_state()))
        print_green("resumed from the latest checkpoint")

    if FLAGS.learner:
        include_penalty = cfg.task.grasp_penalty is not None
        make_buffer = lambda: MemoryEfficientReplayBufferDataStore(
            obs_space, act_space, capacity=cfg.train.buffer_capacity, image_keys=(cfg.task.image_key,),
            include_grasp_penalty=include_penalty)
        replay_buffer, intvn_buffer = make_buffer(), make_buffer()
        scaler, standardizer = load_normalizers(stats_path, cfg.task.grip_dims, cfg.train.action_range_margin)
        demo_files = [p for d in (FLAGS.demo_dir or []) for p in sorted(glob.glob(os.path.join(d, "*.pkl")))]
        load_demos(cfg, demo_files, intvn_buffer, scaler, standardizer)  # D_expert <- D_warm
        load_dumps(os.path.join(FLAGS.exp_dir, "buffer/*.pkl"), replay_buffer)
        load_dumps(os.path.join(FLAGS.exp_dir, "demo_buffer/*.pkl"), intvn_buffer)
        print_green(f"replay buffer: {len(replay_buffer)}, intervention buffer: {len(intvn_buffer)}")
        logger = make_wandb_logger("hire-moma", f"{FLAGS.task}_{FLAGS.method}", debug=FLAGS.debug)
        learner(cfg, agent, replay_buffer, intvn_buffer, logger)
    else:
        env = get_environment(cfg, FLAGS.robot_url, stats_path=stats_path, base_policy_addr=FLAGS.base_policy)
        for key, space in obs_space.spaces.items():
            assert env.observation_space[key].shape == space.shape, f"{key}: robot {env.observation_space[key].shape} != config {space.shape}"
        assert env.action_space.shape == act_space.shape, f"action: robot {env.action_space.shape} != config {act_space.shape}"
        actor(cfg, agent, RecordEpisodeStatistics(env), sampling_rng)


if __name__ == "__main__":
    app.run(main)
