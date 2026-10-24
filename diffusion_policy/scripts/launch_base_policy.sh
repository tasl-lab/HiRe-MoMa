#!/usr/bin/env bash
# Serve the base policy to the actor.   Usage: launch_base_policy.sh <task> [checkpoint]
set -e
TASK=${1:?task}; CKPT=${2:-ckpt/$TASK/ep1000_ema.pt}
python -m diffusion_policy.policy.agent_server --ckpt "$CKPT" --port "${PORT:-5555}" --act_horizon 6
