#!/usr/bin/env bash
# Actor (talks to the robot).   Usage: train_actor.sh <task> <method> [extra flags]
set -e
TASK=${1:?task}; METHOD=${2:?method}; shift 2
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.1
python -m tiago_server.exe_space.train_residual --task "$TASK" --method "$METHOD" --actor \
    --exp_dir "runs/$TASK/$METHOD" --robot_url "${ROBOT_URL:-http://localhost:1234}" \
    --base_policy "${BASE_POLICY:-localhost:5555}" "$@"
