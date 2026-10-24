"""Train the diffusion base policy on teleoperated episodes.

  python tiago_train.py --data_dir data/place_only/imitation --state_dim 15 --action_dim 10 \
      --out_dir ckpt/place_only
"""
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm

from diffusion_policy.tiago.tiago_dataset import TiagoDataset
from diffusion_policy.policy.architecture import IMAGE_KEY, build_nets, make_shape_meta, make_train_scheduler


def save(path, ema, nets, normalizer, args, shape_meta):
    ema_nets = copy.deepcopy(nets)  # the online weights keep training
    ema.copy_to(ema_nets.parameters())
    torch.save({"ema_nets": ema_nets.state_dict(), "normalizer": normalizer.state_dict(), "shape_meta": shape_meta,
                "obs_horizon": args.obs_horizon, "pred_horizon": args.pred_horizon}, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--state_dim", type=int, required=True, help="15 (single arm) or 23 (dual arm)")
    parser.add_argument("--action_dim", type=int, required=True, help="10 (single arm) or 17 (dual arm)")
    parser.add_argument("--pred_horizon", type=int, default=8)
    parser.add_argument("--obs_horizon", type=int, default=1)
    parser.add_argument("--action_horizon", type=int, default=6, help="Padding at episode ends (pad_after = horizon - 1).")
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    shape_meta = make_shape_meta(args.state_dim, args.action_dim)
    dataset = TiagoDataset(shape_meta, args.data_dir, horizon=args.pred_horizon, pad_before=args.obs_horizon - 1,
                           pad_after=args.action_horizon - 1, n_obs_steps=args.obs_horizon)
    normalizer = dataset.get_normalizer().to(device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                         shuffle=True, pin_memory=True, persistent_workers=args.num_workers > 0)

    nets = build_nets(shape_meta, args.obs_horizon).to(device)
    noise_scheduler = make_train_scheduler()
    ema = EMAModel(parameters=nets.parameters(), power=0.75)
    optimizer = torch.optim.AdamW(params=nets.parameters(), lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_scheduler(name="cosine", optimizer=optimizer, num_warmup_steps=500,
                                 num_training_steps=len(loader) * args.num_epochs)
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in tqdm(range(args.num_epochs), desc="Epoch"):
        losses = []
        for batch in loader:
            image = normalizer["obs_" + IMAGE_KEY].normalize(batch["obs"][IMAGE_KEY][:, :args.obs_horizon].to(device))
            state = normalizer["obs_state"].normalize(batch["obs"]["state"][:, :args.obs_horizon].to(device))
            naction = normalizer["action"].normalize(batch["action"].to(device))
            features = nets["vision_encoder"](image.flatten(end_dim=1)).reshape(*image.shape[:2], -1)
            obs_cond = torch.cat([features, state], dim=-1).flatten(start_dim=1)

            noise = torch.randn(naction.shape, device=device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (naction.shape[0],),
                                      device=device).long()
            noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
            loss = nn.functional.mse_loss(nets["noise_pred_net"](noisy_actions, timesteps, global_cond=obs_cond), noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            ema.step(nets.parameters())
            losses.append(loss.item())
        tqdm.write(f"epoch {epoch + 1}: loss {np.mean(losses):.4f}")
        if (epoch + 1) % args.save_every == 0:
            save(os.path.join(args.out_dir, f"ep{epoch + 1}_ema.pt"), ema, nets, normalizer, args, shape_meta)
    save(os.path.join(args.out_dir, "last_ema.pt"), ema, nets, normalizer, args, shape_meta)


if __name__ == "__main__":
    main()
