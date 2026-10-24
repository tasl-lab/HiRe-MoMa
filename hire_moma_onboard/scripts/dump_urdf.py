"""Save the robot's URDF (needed by TRAC-IK) from the ROS parameter server.

  python hire_moma_onboard/scripts/dump_urdf.py
"""
import argparse
import os

import rospy

parser = argparse.ArgumentParser()
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../tiago_onboard/tiago_onboard/tiago_sys/urdf/tiago.urdf"))
args = parser.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w") as f:
    f.write(rospy.get_param("/robot_description"))
print(f"URDF written to {args.out}")
