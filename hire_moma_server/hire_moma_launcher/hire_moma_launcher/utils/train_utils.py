import os
import pickle as pkl
from collections import defaultdict

import jax.numpy as jnp
import requests
from flax.core import FrozenDict, frozen_dict

RESNET10_URL = "https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl"
RESNET10_PATH = os.path.expanduser(os.environ.get("HIRE_MOMA_RESNET10", "~/.cache/hire_moma/resnet10_params.pkl"))


def concat_batches(a, b, axis=0):
    """Concatenate two (nested) batches, e.g. the replay and intervention halves of a batch."""
    out = defaultdict(list)
    a = a if isinstance(a, dict) else a.unfreeze()
    b = b if isinstance(b, dict) else b.unfreeze()
    for k, v in a.items():
        out[k] = concat_batches(v, b[k], axis=axis) if type(v) is dict else jnp.concatenate((v, b[k]), axis=axis)
    return frozen_dict.freeze(out)


def unpack_batch(batch):
    """Split packed image stacks (num_stack + 1 frames) into observations / next_observations."""
    for key in batch["observations"].keys():
        if key not in batch["next_observations"]:
            pixels = batch["observations"][key]
            obs = batch["observations"].copy(add_or_replace={key: pixels[:, :-1, ...]})
            next_obs = batch["next_observations"].copy(add_or_replace={key: pixels[:, 1:, ...]})
            batch = batch.copy(add_or_replace={"observations": obs, "next_observations": next_obs})
    return batch


def _find(params, name):
    if not isinstance(params, (dict, FrozenDict)):
        return None
    for k, v in params.items():
        if k == name:
            return v
        found = _find(v, name)
        if found is not None:
            return found
    return None


def load_resnet10_params(agent, image_keys=("image",)):
    """Load the ImageNet ResNet-10 weights released by SERL into every image encoder of `agent`."""
    if not os.path.exists(RESNET10_PATH):
        os.makedirs(os.path.dirname(RESNET10_PATH), exist_ok=True)
        print(f"Downloading ResNet-10 weights to {RESNET10_PATH}")
        response = requests.get(RESNET10_URL, timeout=60)
        response.raise_for_status()
        with open(RESNET10_PATH, "wb") as f:
            f.write(response.content)
    with open(RESNET10_PATH, "rb") as f:
        encoder_params = pkl.load(f)

    new_params = agent.state.params
    for image_key in image_keys:
        target = _find(new_params, f"encoder_{image_key}")
        if "pretrained_encoder" in target:
            target = target["pretrained_encoder"]
        for k in target:
            if k in encoder_params:
                target[k] = encoder_params[k]
    return agent.replace(state=agent.state.replace(params=new_params))
