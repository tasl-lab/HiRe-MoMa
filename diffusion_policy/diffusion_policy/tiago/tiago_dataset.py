"""Teleoperated TIAGo episodes (one .pkl per episode, written by `record_imitation_demos.py`).

Episodes are converted once into a Jpeg2k-compressed zarr cache (`<dataset_dir>.zarr.zip`).
"""
import concurrent.futures
import multiprocessing
import os
import pickle
from typing import Dict

import numpy as np
import torch
import zarr
from filelock import FileLock
from imagecodecs.numcodecs import Jpeg2k, register_codecs
from threadpoolctl import threadpool_limits

from diffusion_policy.policy.normalizer import LinearNormalizer, SingleFieldLinearNormalizer, dict_apply
from diffusion_policy.tiago.replay_buffer import ReplayBuffer
from diffusion_policy.tiago.sampler import SequenceSampler, get_val_mask

register_codecs()


def get_obs_keys(shape_meta):
    visual = [k for k, v in shape_meta["obs"].items() if v["type"] == "visual"]
    lowdim = [k for k, v in shape_meta["obs"].items() if v["type"] == "lowdim"]
    return visual, lowdim


def range_normalizer(arr: np.ndarray, output_max=1, output_min=-1, range_eps=1e-7):
    """Min-max normalizer to [-1, 1] (constant dims are mapped to the middle)."""
    stat = {"min": np.min(arr, axis=0), "max": np.max(arr, axis=0), "mean": np.mean(arr, axis=0),
            "std": np.std(arr, axis=0)}
    input_range = stat["max"] - stat["min"]
    ignore = input_range < range_eps
    input_range[ignore] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * stat["min"]
    offset[ignore] = (output_max + output_min) / 2 - stat["min"][ignore]
    return SingleFieldLinearNormalizer.create_manual(scale=scale, offset=offset, input_stats_dict=stat)


def image_normalizer():
    """[0, 1] -> [-1, 1]."""
    stat = {"min": np.array([0], dtype=np.float32), "max": np.array([1], dtype=np.float32),
            "mean": np.array([0.5], dtype=np.float32), "std": np.array([np.sqrt(1 / 12)], dtype=np.float32)}
    return SingleFieldLinearNormalizer.create_manual(scale=np.array([2], dtype=np.float32),
                                                     offset=np.array([-1], dtype=np.float32), input_stats_dict=stat)


class TiagoDataset(torch.utils.data.Dataset):
    def __init__(self, shape_meta: dict, dataset_dir: str, horizon=8, pad_before=0, pad_after=3,
                 n_obs_steps=1, use_cache=True, seed=42, val_ratio=0.0):
        if use_cache:
            cache_path = dataset_dir.rstrip("/") + ".zarr.zip"
            with FileLock(cache_path + ".lock"):
                if not os.path.exists(cache_path):
                    try:
                        replay_buffer = _convert_to_replay(zarr.MemoryStore(), shape_meta, dataset_dir)
                        with zarr.ZipStore(cache_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        if os.path.exists(cache_path):
                            os.remove(cache_path)
                        raise e
                else:
                    with zarr.ZipStore(cache_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(src_store=zip_store, store=zarr.MemoryStore())
        else:
            replay_buffer = _convert_to_replay(zarr.MemoryStore(), shape_meta, dataset_dir)

        self.visual_keys, self.lowdim_keys = get_obs_keys(shape_meta)
        key_first_k = {k: n_obs_steps for k in self.visual_keys + self.lowdim_keys}
        val_mask = get_val_mask(n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        self.sampler = SequenceSampler(replay_buffer=replay_buffer, sequence_length=horizon, pad_before=pad_before,
                                       pad_after=pad_after, episode_mask=~val_mask, key_first_k=key_first_k)
        self.replay_buffer = replay_buffer
        self.n_obs_steps = n_obs_steps

    def get_normalizer(self) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = range_normalizer(self.replay_buffer["action"][:])
        for key in self.lowdim_keys:
            normalizer["obs_" + key] = range_normalizer(self.replay_buffer["obs_" + key][:])
        for key in self.visual_keys:
            normalizer["obs_" + key] = image_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        t = slice(self.n_obs_steps)
        obs = {}
        for key in self.visual_keys:  # T,H,W,C uint8 -> T,C,H,W float in [0, 1]
            obs[key] = np.moveaxis(data["obs_" + key][t], -1, 1).astype(np.float32) / 255.0
        for key in self.lowdim_keys:
            obs[key] = data["obs_" + key][t].astype(np.float32)
        return {"obs": dict_apply(obs, torch.from_numpy), "action": torch.from_numpy(data["action"].astype(np.float32))}


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _convert_to_replay(store, shape_meta, dataset_dir, n_workers=None):
    """Episodes are loaded one at a time so that the images never all sit in memory."""
    n_workers = n_workers or multiprocessing.cpu_count()
    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)
    files = sorted(os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".pkl"))
    assert files, f"No .pkl files found in {dataset_dir}"
    visual_keys, lowdim_keys = get_obs_keys(shape_meta)

    lowdim = {k: [] for k in lowdim_keys}
    actions, lengths = [], []
    for f in files:
        episode = _load(f)
        lengths.append(len(episode))
        for k in lowdim_keys:
            lowdim[k].extend(d["observation"][k].astype(np.float32) for d in episode)
        actions.extend(d["action"].astype(np.float32) for d in episode)
    episode_ends = np.cumsum(lengths)
    n_steps = int(episode_ends[-1])
    meta_group.array("episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True)

    for k in lowdim_keys:
        arr = np.squeeze(np.stack(lowdim[k]))
        assert arr.shape == (n_steps, *shape_meta["obs"][k]["shape"]), f"{k}: {arr.shape}"
        data_group.array(name="obs_" + k, data=arr, chunks=arr.shape, compressor=None, dtype=arr.dtype)
    arr = np.squeeze(np.stack(actions))
    assert arr.shape == (n_steps, *shape_meta["action"]["shape"]), f"action: {arr.shape}"
    data_group.array(name="action", data=arr, chunks=arr.shape, compressor=None, dtype=arr.dtype)

    for k in visual_keys:
        h, w, c = shape_meta["obs"][k]["shape"]
        img_arr = data_group.require_dataset(name="obs_" + k, shape=(n_steps, h, w, c), chunks=(1, h, w, c),
                                             compressor=Jpeg2k(level=50), dtype=np.uint8)
        start = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            for f, length in zip(files, lengths):
                frames = np.stack([d["observation"][k].astype(np.uint8) for d in _load(f)]).reshape(length, h, w, c)

                def copy_frame(i, frames=frames, start=start):
                    img_arr[start + i] = frames[i]
                    return True

                assert all(executor.map(copy_frame, range(length))), "Failed to encode images."
                start += length
    return ReplayBuffer(root)
