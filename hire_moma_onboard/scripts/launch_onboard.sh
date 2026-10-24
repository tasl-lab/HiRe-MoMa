#!/usr/bin/env bash
# Robot-side server. Source your ROS / PAL environment first, e.g.
#   source /opt/pal/<distro>/setup.bash; export ROS_MASTER_URI=http://<robot>:11311 ROS_IP=<this-computer-ip>
# Usage: launch_onboard.sh <task>        (sets the per-task gripper closing width)
set -e
TASK=${1:?task}
case "$TASK" in
    pick_place)   GRIPPER_MIN=0.04 ;;
    sweep_litter) GRIPPER_MIN=0.01 ;;
    *)            GRIPPER_MIN=0.00 ;;
esac
python -m tiago_onboard.tiago_env.envs.tiago_env --port "${PORT:-1234}" --gripper_min "$GRIPPER_MIN" \
    ${OCULUS_IP:+--oculus_ip "$OCULUS_IP"}
