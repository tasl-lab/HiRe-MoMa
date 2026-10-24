"""JSON codec shared by the onboard server and the workstation client.

Two identical copies: `hire_moma_server/tiago_server/tiago_server/tiago_env/utils/flask_comm.py` and
`hire_moma_onboard/tiago_onboard/tiago_onboard/tiago_env/utils/flask_comm.py`; keep them in sync.
RGB images are sent as JPEG (quality 80), depth images as 16-bit PNG.
"""
import base64
from collections import OrderedDict
from typing import Any

import cv2
import numpy as np

try:
    import gymnasium as gym
except ImportError:  # the onboard computer uses gym
    import gym

_INF = 1e10  # JSON has no inf


def _encode_image(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img.astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buf).decode("utf-8")


def _decode_image(data: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(data), dtype=np.uint8), cv2.IMREAD_COLOR)


def _encode_depth(depth: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", depth.astype(np.uint16))
    return base64.b64encode(buf).decode("utf-8")


def _decode_depth(data: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(data), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def encode2json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: encode2json(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.integer) and obj.ndim == 3 and obj.shape[-1] == 3:
            return {"type": "image", "data": _encode_image(obj)}
        if np.issubdtype(obj.dtype, np.integer) and obj.ndim == 3 and obj.shape[-1] == 1:
            return {"type": "depth", "data": _encode_depth(obj)}
        return {"type": "array", "data": obj.tolist()}
    if isinstance(obj, tuple):
        return {"type": "tuple", "data": [encode2json(v) for v in obj]}
    if isinstance(obj, list):
        return {"type": "list", "data": [encode2json(v) for v in obj]}
    return obj


def decode4json(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    kind = obj.get("type")
    if kind == "image":
        return _decode_image(obj["data"])
    if kind == "depth":
        return _decode_depth(obj["data"])
    if kind == "array":
        return np.array(obj["data"], dtype=np.float32)
    if kind == "tuple":
        return tuple(decode4json(v) for v in obj["data"])
    if kind == "list":
        return [decode4json(v) for v in obj["data"]]
    return {k: decode4json(v) for k, v in obj.items()}


def serialize_space_dict(space_dict) -> dict:
    """gym Dict of Box spaces -> JSON-serializable dict."""
    def clean(arr, dtype):
        return np.nan_to_num(np.array(arr, dtype=dtype), nan=0.0, posinf=_INF, neginf=-_INF).tolist()

    return OrderedDict((key, {"type": "Box", "low": clean(box.low, box.dtype), "high": clean(box.high, box.dtype),
                              "shape": box.shape, "dtype": str(box.dtype)})
                       for key, box in space_dict.spaces.items())


def reconstruct_space_dict(data: dict):
    def restore(arr, dtype):
        arr = np.array(arr, dtype=dtype)
        if np.issubdtype(arr.dtype, np.floating):
            arr[np.isclose(arr, _INF)] = np.inf
            arr[np.isclose(arr, -_INF)] = -np.inf
        return arr

    spaces = OrderedDict()
    for key, val in data.items():
        if val["type"] == "Box":
            spaces[key] = gym.spaces.Box(low=restore(val["low"], val["dtype"]), high=restore(val["high"], val["dtype"]),
                                         shape=tuple(val["shape"]), dtype=np.dtype(val["dtype"]))
    return gym.spaces.Dict(spaces)
