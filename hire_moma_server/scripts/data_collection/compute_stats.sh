#!/usr/bin/env bash
# Normalization statistics from the task's data.   Usage: compute_stats.sh <task>
set -e
TASK=${1:?task}
python -m tiago_server.exe_space.utils.compute_stats --data_dir "data/$TASK/imitation" \
    --data_dir "data/$TASK/warmup" --out "stats/$TASK.json"
