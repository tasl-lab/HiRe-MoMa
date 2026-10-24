"""Sweep Litter: sweep litter into a dustpan with a broom and a dustpan held in closed grippers (15-D, 500 steps)."""
from tiago_server.exe_space.experiments.config import TaskConfig

TASK = TaskConfig(name="sweep_litter", arms=("left", "right"), max_episode_length=500, torso=0.22,
                  reset_arms={"left": [0.732, -0.970, 1.748, 2.120, 1.544, -0.446, 1.879, 0],
                              "right": [0.595, -0.684, 1.665, 1.760, 1.408, -0.546, 1.213, 0]},
                  grippers_closed=True, grasp_penalty=None)
