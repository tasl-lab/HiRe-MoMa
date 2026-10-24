#!/usr/bin/env bash
# Train the Diffusion Policy base policy.   Usage: train_base_policy.sh <task>
set -e
TASK=${1:?task}
case "$TASK" in
    sweep_litter) DIMS="--state_dim 23 --action_dim 17" ;;
    *)            DIMS="--state_dim 15 --action_dim 10" ;;
esac
python diffusion_policy/tiago_train.py --data_dir "data/$TASK/imitation" --out_dir "ckpt/$TASK" $DIMS \
    --pred_horizon 8 --obs_horizon 1 --num_epochs 1000
