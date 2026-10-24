"""Pick Place: grasp the bread, drive to a second table and place it on a plate (10-D action, 800 steps)."""
from tiago_server.exe_space.experiments.config import TaskConfig

ARM_RESET = [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1]  # 7 joints + gripper (open)

TASK = TaskConfig(name="pick_place", arms=("right",), max_episode_length=800, torso=0.29,
                  reset_arms={"left": ARM_RESET, "right": ARM_RESET})
