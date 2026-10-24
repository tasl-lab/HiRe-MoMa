"""Onboard server: runs next to the robot (ROS1), executes actions and arbitrates human takeover.

  python -m tiago_onboard.tiago_env.envs.tiago_env --port 1234 --oculus_ip <quest-ip> --gripper_min 0.0

Shared control: whenever the operator holds an arm trigger (RG / LG) or moves a joystick, the
Quest command replaces the policy action and is returned in `info["intervene_action"]`.
"""
import numpy as np
import rospy
from absl import app, flags
from flask import Flask, jsonify, request

from tiago_onboard.tiago_env.utils.flask_comm import decode4json, encode2json, serialize_space_dict
from tiago_onboard.tiago_sys.tiago_ctrl.tiago_head import LookAtFixedPoint
from tiago_onboard.tiago_sys.tiago_ctrl.tiago_gym import TiagoGym
from tiago_onboard.tiago_sys.oculus_teleop.oculus import OculusPolicy

FLAGS = flags.FLAGS
flags.DEFINE_string("host", "0.0.0.0", "Interface the server listens on.")
flags.DEFINE_integer("port", 1234, "Server port.")
flags.DEFINE_string("oculus_ip", None, "Quest 3 IP (adb over Wi-Fi). Leave empty for USB.")
flags.DEFINE_float("gripper_min", 0.0, "PAL gripper opening (m) commanded as 'closed' "
                   "(place_only 0.00, pick_place 0.04, sweep_litter 0.01; set by launch_onboard.sh).")
flags.DEFINE_float("gripper_max", 0.09, "PAL gripper opening (m) commanded as 'open'.")

# Default pose at start-up; every episode the client sends the task's reset pose.
DEFAULT_RESET_POSE = {
    "left": [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1],   # 7 joints + gripper (0 close / 1 open)
    "right": [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1],
    "torso": 0.20,
    "head": [0.0, -0.90],
}


class TiagoEnv(TiagoGym):
    ALL_ACT_KEYS = ("left", "right", "base", "torso")
    BANNED_ACT_KEYS = ("torso",)  # the torso stays at its reset height
    ARM_TRIGGERS = {"left": "LG", "right": "RG"}
    JOYSTICK_THRESHOLD = {"base": 0.001, "torso": 0.001}

    def __init__(self, oculus_ip=None, gripper_min=0.0, gripper_max=0.09, reset_pose=DEFAULT_RESET_POSE):
        super().__init__(head_policy=LookAtFixedPoint(reset_pose["head"]), base_enabled=True, torso_enabled=False,
                         left_arm_enabled=True, right_arm_enabled=True, right_gripper_type="pal",
                         left_gripper_type="pal", reset_pose=reset_pose,
                         gripper_kwargs={"gripper_min": gripper_min, "gripper_max": gripper_max})
        self.teleop = OculusPolicy(ip_address=oculus_ip)
        self.teleop.start()
        self.obs = self.reset(reset_arms=True, is_input_cont=True)

    def _teleop(self):
        signals = self.teleop.get_action(self.obs, is_filter=True)  # delta actions, gripper 0 close / 1 open
        return signals, signals.extra.get("buttons", {})

    def _is_intervening(self, signals, buttons):
        for key in self.ALL_ACT_KEYS:
            if key in self.BANNED_ACT_KEYS:
                continue
            if key in self.ARM_TRIGGERS:
                if buttons[self.ARM_TRIGGERS[key]]:
                    return True
            elif np.linalg.norm(signals[key]) > self.JOYSTICK_THRESHOLD[key]:
                return True
        return False

    def step(self, action, is_only_teleop=False):
        signals, buttons = self._teleop()
        human = {k: signals[k] for k in action.keys()}
        intervened = is_only_teleop or self._is_intervening(signals, buttons)
        exe_action = human if intervened else action
        self.tiago.step(exe_action)
        self.obs = self._observation()
        self.steps += 1
        info = {"teleop_buttons": buttons}
        if intervened:
            info["intervene_action"] = exe_action
        return self.obs, info

    def reset(self, *args, **kwargs):
        self.steps = 0
        self.tiago.reset(*args, **kwargs)
        self.teleop.reset_state()
        self.obs = self._observation()
        return self.obs

    def close(self):
        self.teleop.stop()


def main(_):
    rospy.init_node("tiago_onboard_server")
    env = TiagoEnv(oculus_ip=FLAGS.oculus_ip, gripper_min=FLAGS.gripper_min, gripper_max=FLAGS.gripper_max)
    webapp = Flask(__name__)

    @webapp.route("/tiago_reset", methods=["POST"])
    def tiago_reset():
        reset_pose = decode4json(request.json["reset_pose"])
        return jsonify(encode2json(env.reset(reset_pose=reset_pose, reset_arms=True, is_input_cont=False)))

    @webapp.route("/tiago_step", methods=["POST"])
    def tiago_step():
        obs, info = env.step(decode4json(request.json["action"]), request.json["is_only_teleop"])
        return jsonify({"obs": encode2json(obs), "info": encode2json(info)})

    @webapp.route("/tiago_get_action_space", methods=["POST"])
    def tiago_get_action_space():
        return jsonify(serialize_space_dict(env.action_space))

    @webapp.route("/tiago_get_observation_space", methods=["POST"])
    def tiago_get_observation_space():
        return jsonify(serialize_space_dict(env.observation_space))

    @webapp.route("/tiago_close", methods=["POST"])
    def tiago_close():
        env.close()
        return "success"

    webapp.run(host=FLAGS.host, port=FLAGS.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    app.run(main)
