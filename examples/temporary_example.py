import random
import numpy as np
import matplotlib.pyplot as plt

from flatland.core.env import RailEnv
from flatland.utils.rail_env_generator import *
from flatland.utils.rendertools import *

random.seed(1)
np.random.seed(1)


# Example generate a random rail
rail = generate_random_rail(20, 20)

env = RailEnv(rail, number_of_agents=10)
env.reset()

env_renderer = RenderTool(env)
env_renderer.renderEnv(show=True)


# Example generate a rail given a manual specification,
# a map of tuples (cell_type, rotation)
specs = [[(0, 0), (0, 0), (0, 0), (0, 0), (7, 0), (0, 0)],
         [(7, 270), (1, 90), (1, 90), (1, 90), (2, 90), (7, 90)]]

rail = generate_rail_from_manual_specifications(specs)
env = RailEnv(rail, number_of_agents=1)

handle = env.get_agent_handles()

env.reset()

env.agents_position = [[1, 4]]
env.agents_target = [[1, 1]]
env.agents_direction = [1]

env_renderer = RenderTool(env)
env_renderer.renderEnv(show=True)


print("Manual control: s=perform step, q=quit, [agent id] [1-2-3 action] \
       (turnleft+move, move to front, turnright+move)")
for step in range(100):
    cmd = input(">> ")
    cmds = cmd.split(" ")

    action_dict = {}

    i = 0
    while i < len(cmds):
        if cmds[i] == 'q':
            import sys
            sys.exit()
        elif cmds[i] == 's':
            obs, all_rewards, done, _ = env.step(action_dict)
            action_dict = {}
            print("Rewards: ", all_rewards, "  [done=", done, "]")
        else:
            agent_id = int(cmds[i])
            action = int(cmds[i+1])
            action_dict[agent_id] = action
            i = i+1
        i += 1

    env_renderer.renderEnv(show=True)
