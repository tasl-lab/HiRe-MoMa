import queue
import threading
import time

import numpy as np
from agentlace.inference import InferenceClient


class DiffusionPolicyClient:
    """Streams base-policy actions from `agent_server.py` (used by the RL actor; needs only agentlace).

    The server returns a chunk of `act_horizon` actions per query; the client queues them after
    dropping the first `skip`, and a background thread re-queries with the latest observation once
    the queue holds <= `skip` actions. skip=0 (paper): execute all 6 actions, then re-plan;
    skip > 0 re-plans earlier and drops the stale actions, hiding the inference latency.
    """

    def __init__(self, host: str = "localhost", port: int = 5555, act_horizon: int = 6, skip: int = 0,
                 timeout_ms: int = 5000):
        self.client = InferenceClient(host, port)
        self.client.client.timeout_ms = timeout_ms  # agentlace default (800 ms) is too short for CPU inference
        self.client.client.reset_socket()
        self.skip = skip
        self.action_buffer = queue.Queue(maxsize=act_horizon)
        self.predicting = False
        self.last_obs = None
        threading.Thread(target=self._background_loop, daemon=True).start()

    def _background_loop(self):
        while True:
            time.sleep(0.01)
            if self.action_buffer.qsize() <= self.skip and self.last_obs is not None and not self.predicting:
                self.predicting = True
                try:
                    self._predict(self.last_obs)
                finally:
                    self.predicting = False

    def _predict(self, obs):
        response = None
        while response is None:
            response = self.client.call("predict", obs)
        actions = response["actions"][:, self.skip:, ...]  # (B=1, T, D)
        for act in actions.transpose(1, 0, 2):
            self.action_buffer.put(act)

    def predict_action(self, obs) -> np.ndarray:
        """Next base action, shape (1, action_dim). Blocks if no action is ready."""
        self.last_obs = obs
        return self.action_buffer.get()

    def reset(self):
        while not self.action_buffer.empty():
            try:
                self.action_buffer.get_nowait()
            except queue.Empty:
                break
        self.last_obs = None
        self.predicting = False
