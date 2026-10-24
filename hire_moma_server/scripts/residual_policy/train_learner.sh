#!/usr/bin/env bash
# Learner (workstation GPU).   Usage: train_learner.sh <task> <method> [extra flags]
set -e
TASK=${1:?task}; METHOD=${2:?method}; shift 2
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.3
python -m tiago_server.exe_space.train_residual --task "$TASK" --method "$METHOD" --learner \
    --exp_dir "runs/$TASK/$METHOD" --demo_dir "data/$TASK/warmup" "$@"
