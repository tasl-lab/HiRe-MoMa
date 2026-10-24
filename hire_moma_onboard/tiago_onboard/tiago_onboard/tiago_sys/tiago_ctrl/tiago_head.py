import rospy
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from tiago_onboard.tiago_sys.utils.camera_utils import Camera
from tiago_onboard.tiago_sys.utils.ros_utils import Listener, Publisher


class LookAtFixedPoint:
    """Head policy that keeps the head at a fixed [pan, tilt]."""

    def __init__(self, pose) -> None:
        self.pose = pose

    def get_action(self, env_action):
        return self.pose[:2]


class TiagoHead:
    def __init__(self, head_policy) -> None:
        self.head_enabled = head_policy is not None
        self.head_policy = head_policy
        self.head_camera = Camera(img_topic="/xtion/rgb/image_raw", depth_topic="/xtion/depth/image_raw")
        self.head_listener = Listener("/head_controller/state", JointTrajectoryControllerState,
                                      post_process_func=lambda m: m.actual.positions)
        self.head_writer = Publisher("/head_controller/command", JointTrajectory) if self.head_enabled else None

    def write(self, pose):
        if self.head_enabled:
            message = JointTrajectory()
            message.header = Header()
            message.joint_names = ["head_1_joint", "head_2_joint"]
            message.points.append(JointTrajectoryPoint(positions=pose, time_from_start=rospy.Duration(0.5)))
            self.head_writer.write(message)

    def step(self, env_action):
        pose = self.head_policy.get_action(env_action)
        if pose is not None:
            self.write(pose)

    def reset(self, pose):
        if pose is not None:
            self.write(pose)
