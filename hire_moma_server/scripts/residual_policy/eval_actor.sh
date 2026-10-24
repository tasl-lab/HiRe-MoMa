#!/usr/bin/env bash
# Evaluate a checkpoint (20 episodes).   Usage: eval_actor.sh <task> <method> <step> [extra flags]
#   e.g. eval_actor.sh place_only hire_moma 5000        |  base policy only: eval_actor.sh place_only hire_moma 0 --base_only
set -e
TASK=${1:?task}; METHOD=${2:?method}; STEP=${3:?checkpoint step}; shift 3
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.1
python -m tiago_server.exe_space.train_residual --task "$TASK" --method "$METHOD" --actor \
    --exp_dir "runs/$TASK/$METHOD" --robot_url "${ROBOT_URL:-http://localhost:1234}" \
    --base_policy "${BASE_POLICY:-localhost:5555}" --eval_step "$STEP" --eval_episodes 20 "$@"
