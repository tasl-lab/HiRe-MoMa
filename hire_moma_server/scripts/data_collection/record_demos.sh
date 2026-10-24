#!/usr/bin/env bash
# Warm-up buffer (successful base-policy rollouts).   Usage: record_demos.sh <task>
set -e
TASK=${1:?task}
python -m tiago_server.exe_space.data_collection.record_demos --task "$TASK" \
    --robot_url "${ROBOT_URL:-http://localhost:1234}" --base_policy "${BASE_POLICY:-localhost:5555}" \
    --out_dir "data/$TASK/warmup"
