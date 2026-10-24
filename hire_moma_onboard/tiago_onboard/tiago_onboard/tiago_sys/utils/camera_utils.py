import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from tiago_onboard.tiago_sys.utils.ros_utils import Listener


def img_processing(data):
    """ROS image -> RGB array."""
    return np.array(cv2.cvtColor(CvBridge().imgmsg_to_cv2(data), cv2.COLOR_BGR2RGB)).astype(int)


def depth_processing(data):
    return np.expand_dims(np.array(CvBridge().imgmsg_to_cv2(data)), -1).astype(int)


class Camera:
    """Latest RGB / depth frames of a ROS camera."""

    def __init__(self, img_topic, depth_topic, input_message_type=Image) -> None:
        self.img_listener = Listener(img_topic, input_message_type, post_process_func=img_processing)
        self.depth_listener = Listener(depth_topic, input_message_type, post_process_func=depth_processing)
        self._img_shape = self.img_listener.get_most_recent_msg().shape
        self._depth_shape = self.depth_listener.get_most_recent_msg().shape

    def get_img(self):
        return self.img_listener.get_most_recent_msg()

    def get_depth(self):
        return self.depth_listener.get_most_recent_msg()

    @property
    def img_shape(self):
        return self._img_shape

    @property
    def depth_shape(self):
        return self._depth_shape
