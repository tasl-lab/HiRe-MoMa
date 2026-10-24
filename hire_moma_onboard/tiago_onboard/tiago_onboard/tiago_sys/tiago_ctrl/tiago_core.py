
import rospy

from tiago_onboard.tiago_sys.tiago_ctrl.tiago_arms import TiagoArms
from tiago_onboard.tiago_sys.tiago_ctrl.tiago_mobile_base import TiagoBaseVelocityControl
from tiago_onboard.tiago_sys.tiago_ctrl.grippers import PALGripper, RobotiqGripper2F_85, RobotiqGripper2F_140
from tiago_onboard.tiago_sys.tiago_ctrl.tiago_head import TiagoHead
from tiago_onboard.tiago_sys.tiago_ctrl.tiago_torso import TiagoTorso


class Tiago:
    gripper_map = {"pal": PALGripper, "robotiq2F-140": RobotiqGripper2F_140, "robotiq2F-85": RobotiqGripper2F_85}

    def __init__(self, head_policy=None, base_enabled=False, torso_enabled=False, right_arm_enabled=True,
                 left_arm_enabled=True, right_gripper_type=None, left_gripper_type=None, reset_pose=None,
                 gripper_kwargs=None):
        self.head_enabled = head_policy is not None
        self.base_enabled = base_enabled
        self.torso_enabled = torso_enabled
        self.head = TiagoHead(head_policy=head_policy)
        self.base = TiagoBaseVelocityControl(base_enabled=base_enabled)
        self.torso = TiagoTorso(torso_enabled=torso_enabled)
        self.arms = {"right": TiagoArms(right_arm_enabled, side="right"),
                     "left": TiagoArms(left_arm_enabled, side="left")}
        self.gripper = {"right": None, "left": None}
        for side, gripper_type in (("right", right_gripper_type), ("left", left_gripper_type)):
            if gripper_type is not None:
                self.gripper[side] = self.gripper_map[gripper_type](side, **(gripper_kwargs or {}))
        self.reset_pose = reset_pose

    @property
    def right_gripper_pos(self):
        return None if self.gripper["right"] is None else self.gripper["right"].get_state()

    @property
    def left_gripper_pos(self):
        return None if self.gripper["left"] is None else self.gripper["left"].get_state()

    def step(self, action):
        """action: {"left"/"right": [dx dy dz drx dry drz, gripper], "base": [vx vy wz], "torso": dz}."""
        if self.torso_enabled and action.get("torso") is not None:
            self.torso.step(action["torso"])
        for side in ("right", "left"):
            if action.get(side) is None:
                continue
            self.arms[side].step(action[side][:6])
            if self.gripper[side] is not None:
                self.gripper[side].step(action[side][6])
        if self.head_enabled:
            self.head.step(action)
        if self.base_enabled and action.get("base") is not None:
            self.base.step(action["base"])

    def reset(self, reset_pose=None, reset_arms=True, is_input_cont=False):
        """Move torso, arms (joint targets) and head to `reset_pose`; the base is placed by the operator."""
        pose = reset_pose if reset_pose is not None else self.reset_pose
        if "torso" in pose and self.torso is not None:
            print("[reset] torso")
            self.torso.reset(pose["torso"])
            rospy.sleep(3)
        for side in ("right", "left"):
            if pose[side] is not None and self.arms[side].arm_enabled:
                self.gripper[side].step(pose[side][-1])
                if reset_arms:
                    print(f"[reset] {side} arm")
                    self.arms[side].reset(pose[side][:-1])
                    rospy.sleep(1)
        if self.head_enabled:
            self.head.reset(pose["head"])
        rospy.sleep(0.5)
        if is_input_cont:
            input("Reset complete. Press ENTER to continue")
