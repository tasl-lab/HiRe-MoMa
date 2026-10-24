#!/usr/bin/env bash
# Teleoperated demonstrations for the base policy.   Usage: record_imitation_demos.sh <task> [episodes]
set -e
TASK=${1:?task}; EPISODES=${2:-80}
python -m tiago_server.exe_space.data_collection.record_imitation_demos --task "$TASK" \
    --robot_url "${ROBOT_URL:-http://localhost:1234}" --out_dir "data/$TASK/imitation" --episodes "$EPISODES"
