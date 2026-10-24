import os

import numpy as np
import rospy
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import Header
from tracikpy import TracIKSolver
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from tiago_onboard.tiago_sys.utils.ros_utils import Listener, Publisher, TFTransformListener
from tiago_onboard.tiago_sys.utils.transformations import add_angles, euler_to_quat, quat_to_euler, quat_to_rmat

URDF_PATH = os.environ.get("TIAGO_URDF", os.path.join(os.path.dirname(os.path.realpath(__file__)), "../urdf/tiago.urdf"))


class TiagoArms:
    """End-effector delta control through IK (TRAC-IK) on joint trajectories."""

    def __init__(self, arm_enabled, side="right") -> None:
        self.arm_enabled = arm_enabled
        self.side = side
        self.arm_reader = TFTransformListener("/base_footprint")
        self.joint_reader = Listener(f"/arm_{self.side}_controller/state", JointTrajectoryControllerState,
                                     post_process_func=lambda data: np.array(data.actual.positions))
        self.arm_writer = None
        if self.arm_enabled:
            self.ik_solver = TracIKSolver(urdf_file=URDF_PATH, base_link="torso_lift_link",
                                          tip_link=f"arm_{self.side}_tool_link", timeout=0.025, epsilon=5e-4,
                                          solve_type="Distance")
            self.arm_writer = Publisher(f"/arm_{self.side}_controller/safe_command", JointTrajectory)

    @property
    def arm_pose(self):
        """[x, y, z, qx, qy, qz, qw] of the tool link in /base_footprint."""
        pos, quat = self.arm_reader.get_transform(target_link=f"/arm_{self.side}_tool_link")
        if pos is None:
            return None
        return np.concatenate((pos, quat))

    def process_action(self, action):
        """Delta (xyz, euler) -> absolute target in /torso_lift_link."""
        pos_delta, euler_delta = action[:3], action[3:6]
        cur_pos, cur_quat = self.arm_reader.get_transform(target_link=f"/arm_{self.side}_tool_link",
                                                          base_link="/torso_lift_link")
        target_pos = cur_pos + pos_delta
        target_quat = euler_to_quat(add_angles(euler_delta, quat_to_euler(cur_quat)))
        return target_pos, target_quat

    def create_joint_command(self, joint_goal, duration_scale):
        message = JointTrajectory()
        message.header = Header()
        positions = list(self.joint_reader.get_most_recent_msg())
        message.joint_names = [f"arm_{self.side}_{i}_joint" for i in range(1, 8)]
        positions[:7] = joint_goal[:7]
        message.points.append(JointTrajectoryPoint(positions=positions,
                                                   time_from_start=rospy.Duration(0.7 + duration_scale)))
        return message

    def write(self, joint_goal, duration_scale):
        if self.arm_writer is not None:
            self.arm_writer.write(self.create_joint_command(joint_goal, duration_scale))

    def find_ik(self, target_pos, target_quat):
        ee_pose = np.eye(4)
        ee_pose[:3, :3] = quat_to_rmat(target_quat)
        ee_pose[:3, 3] = np.array(target_pos)
        joint_init = self.joint_reader.get_most_recent_msg()
        joint_goal = self.ik_solver.ik(ee_pose, qinit=joint_init)
        duration_scale = 0 if joint_goal is None else np.linalg.norm(joint_goal - joint_init)
        return joint_goal, duration_scale

    def step(self, action):
        if self.arm_enabled:
            assert len(action) == 6  # dx dy dz drx dry drz
            joint_goal, duration_scale = self.find_ik(*self.process_action(action))
            if joint_goal is not None:
                self.write(joint_goal, duration_scale)

    def reset(self, goal_joints):
        if self.arm_enabled:
            assert len(goal_joints) == 7
            delay_scale = np.linalg.norm(self.joint_reader.get_most_recent_msg() - goal_joints)
            assert delay_scale < 6, "Reset pose too far from the current joints; move the arm closer first."
            self.write(goal_joints, delay_scale * 3)
            rospy.sleep(delay_scale * 4)
