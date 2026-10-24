import numpy as np
import rospy
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from tiago_onboard.tiago_sys.utils.ros_utils import Listener, Publisher


class TiagoTorso:
    def __init__(self, torso_enabled) -> None:
        self.torso_enabled = torso_enabled
        self.torso_listener = Listener("/torso_controller/state", JointTrajectoryControllerState,
                                       post_process_func=lambda m: m.actual.positions[0])
        self.torso_writer = Publisher("/torso_controller/safe_command", JointTrajectory)

    def create_torso_command(self, dist):
        message = JointTrajectory()
        message.header = Header()
        message.joint_names = ["torso_lift_joint"]
        message.points.append(JointTrajectoryPoint(positions=[dist], time_from_start=rospy.Duration(0.5)))
        return message

    def step(self, position_delta):
        if self.torso_enabled and position_delta is not None:
            goal = np.clip(self.get_torso_extension() + position_delta, 0.05, 0.30)
            self.torso_writer.write(self.create_torso_command(goal))

    def get_torso_extension(self):
        return self.torso_listener.get_most_recent_msg()

    def reset(self, abs_pos):
        if abs_pos is not None:
            self.torso_writer.write(self.create_torso_command(abs_pos))
