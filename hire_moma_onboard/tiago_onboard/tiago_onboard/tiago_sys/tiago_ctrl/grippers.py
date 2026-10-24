import rospy
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import Bool, Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from tiago_onboard.tiago_sys.utils.ros_utils import Listener, Publisher


def _command(joint_name, dist):
    message = JointTrajectory()
    message.header = Header()
    message.joint_names = [joint_name]
    message.points.append(JointTrajectoryPoint(positions=[dist], time_from_start=rospy.Duration(0.5)))
    return message


class PALGripper:
    """PAL parallel gripper. Commands / states are normalized: 0 = closed (gripper_min), 1 = open (gripper_max)."""

    def __init__(self, side, gripper_min=0.0, gripper_max=0.09):
        self.side = side
        self.gripper_min = gripper_min
        self.gripper_max = gripper_max
        self.gripper_reader = Listener(f"/parallel_gripper_{side}_controller/state", JointTrajectoryControllerState,
                                       post_process_func=lambda m: m.actual.positions[0])
        self.gripper_writer = Publisher(f"/parallel_gripper_{side}_controller/command", JointTrajectory)

    def get_state(self):
        return (self.gripper_reader.get_most_recent_msg() - self.gripper_min) / (self.gripper_max - self.gripper_min)

    def step(self, gripper_act):
        dist = gripper_act * (self.gripper_max - self.gripper_min) + self.gripper_min
        self.gripper_writer.write(_command("parallel_gripper_joint", dist))


class RobotiqGripper2F:
    gripper_max = None

    def __init__(self, side, gripper_min=0.0, **unused):
        self.side = side
        self.gripper_min = gripper_min
        self.gripper_reader = Listener(f"/gripper_{side}_controller/state", JointTrajectoryControllerState,
                                       post_process_func=lambda m: m.actual.positions[0])
        self.gripper_writer = Publisher(f"/gripper_{side}_controller/command", JointTrajectory)
        self.is_grasped = Listener(f"/gripper_{side}/is_grasped", Bool, post_process_func=lambda x: x.data)

    def get_state(self):
        return (self.gripper_max - self.gripper_reader.get_most_recent_msg()) / (self.gripper_max - self.gripper_min)

    def step(self, gripper_act):
        if self.is_grasped.get_most_recent_msg() and gripper_act < self.get_state():
            gripper_act = self.get_state()
        dist = self.gripper_max - gripper_act * (self.gripper_max - self.gripper_min)
        self.gripper_writer.write(_command(f"gripper_{self.side}_finger_joint", dist))


class RobotiqGripper2F_140(RobotiqGripper2F):
    gripper_max = 0.7


class RobotiqGripper2F_85(RobotiqGripper2F):
    gripper_max = 0.8
