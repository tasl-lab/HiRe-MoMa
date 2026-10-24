import numpy as np
import torch

from diffusion_policy.policy.architecture import IMAGE_KEY, build_nets, make_inference_scheduler
from diffusion_policy.policy.normalizer import LinearNormalizer


def _to_btchw(x: torch.Tensor) -> torch.Tensor:
    """Image -> (B, T, C, H, W). Accepts (T,H,W,C), (B,T,H,W,C), (B,T,C,H,W)."""
    if x.ndim == 4:
        x = x.unsqueeze(0)
    if x.shape[2] not in (1, 3):
        x = x.permute(0, 1, 4, 2, 3)
    return x


def _to_btd(x: torch.Tensor) -> torch.Tensor:
    """State -> (B, T, D). Accepts (T,D) or (B,T,D)."""
    return x.unsqueeze(0) if x.ndim == 2 else x


class DiffusionPolicyAgent:
    """Frozen base policy from a `tiago_train.py` checkpoint; predicts action chunks with 16 DDIM steps."""

    def __init__(self, ckpt_path: str, act_horizon: int = 6, device: str = "cuda", num_inference_steps: int = 16):
        ckpt = torch.load(ckpt_path, map_location=device)
        self.shape_meta = ckpt["shape_meta"]
        self.obs_horizon = ckpt["obs_horizon"]
        self.pred_horizon = ckpt["pred_horizon"]
        self.act_horizon = act_horizon
        self.device = torch.device(device)
        self.nets = build_nets(self.shape_meta, self.obs_horizon).to(self.device)
        self.nets.load_state_dict(ckpt["ema_nets"])
        self.nets.eval()
        self.normalizer = LinearNormalizer()
        self.normalizer.load_state_dict(ckpt["normalizer"])
        self.normalizer.to(self.device)
        self.scheduler = make_inference_scheduler(num_inference_steps)
        self.act_dim = self.shape_meta["action"]["shape"][0]

    @torch.no_grad()
    def predict_chunk(self, obs: dict) -> np.ndarray:
        """obs: {"tiago_head_image": uint8 (T,H,W,C), "state": (T,D)} (optionally batched).

        Returns (B, act_horizon, action_dim) raw actions.
        """
        image = _to_btchw(torch.as_tensor(np.asarray(obs[IMAGE_KEY]), dtype=torch.float32, device=self.device))
        image = image / 255.0  # uint8 -> [0, 1], as in training
        state = _to_btd(torch.as_tensor(np.asarray(obs["state"]), dtype=torch.float32, device=self.device))
        image = self.normalizer["obs_" + IMAGE_KEY].normalize(image[:, :self.obs_horizon])
        state = self.normalizer["obs_state"].normalize(state[:, :self.obs_horizon])

        features = self.nets["vision_encoder"](image.flatten(end_dim=1)).reshape(*image.shape[:2], -1)
        obs_cond = torch.cat([features, state], dim=-1).flatten(start_dim=1)
        naction = torch.randn((obs_cond.shape[0], self.pred_horizon, self.act_dim), device=self.device)
        for k in self.scheduler.timesteps:
            noise = self.nets["noise_pred_net"](sample=naction, timestep=k, global_cond=obs_cond)
            naction = self.scheduler.step(model_output=noise, timestep=k, sample=naction, eta=0.0).prev_sample
        action = self.normalizer["action"].unnormalize(naction)
        start = self.obs_horizon - 1
        return action[:, start:start + self.act_horizon].cpu().numpy()
