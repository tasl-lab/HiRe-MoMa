import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from tiago_onboard.tiago_sys.utils.ros_utils import Listener, Publisher, create_twist_command
from tiago_onboard.tiago_sys.utils.transformations import quat_diff, quat_to_euler


class TiagoBaseVelocityControl:
    """Velocity control of the mobile base: action = [vx, vy, wz] (vy has no effect on TIAGo)."""

    lin_scale = 0.2
    ang_scale = 0.2

    def __init__(self, base_enabled) -> None:
        self.base_enabled = base_enabled

        def process_odom(message):
            p, o = message.pose.pose.position, message.pose.pose.orientation
            lin, ang = message.twist.twist.linear, message.twist.twist.angular
            return {"pose": np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w]),
                    "velocity": np.array([lin.x, lin.y, lin.z, ang.x, ang.y, ang.z])}

        self.odom_listener = Listener("/mobile_base_controller/odom", Odometry, post_process_func=process_odom)
        # Base pose is reported relative to the odometry when the server starts.
        self.reference_odom = self.odom_listener.get_most_recent_msg()
        self.base_writer = Publisher("/mobile_base_controller/cmd_vel", Twist) if base_enabled else None

    def get_delta_pose(self):
        """[dx, dy, drz] w.r.t. the reference odometry."""
        current = self.odom_listener.get_most_recent_msg()["pose"]
        reference = self.reference_odom["pose"]
        delta_pos = current[:3] - reference[:3]
        delta_euler = quat_to_euler(quat_diff(current[3:], reference[3:]))
        return np.array([delta_pos[0], delta_pos[1], delta_euler[2]])

    def get_velocity(self):
        """[vx, vy, wz]."""
        v = self.odom_listener.get_most_recent_msg()["velocity"]
        return np.array([v[0], v[1], v[5]])

    def step(self, velocity):
        if not self.base_enabled:
            return
        if velocity is None:
            self.base_writer.write(create_twist_command(np.zeros(3), np.zeros(3)))
            return
        lin_cmd = np.zeros(3)
        lin_cmd[:2] = velocity[:2]
        ang_cmd = np.zeros(3)
        ang_cmd[2] = velocity[2]
        self.base_writer.write(create_twist_command(self.lin_scale * lin_cmd, self.ang_scale * ang_cmd))
