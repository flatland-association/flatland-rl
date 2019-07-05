import random

import numpy as np

from flatland.envs.generators import complex_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool

random.seed(1)
np.random.seed(1)

env = RailEnv(width=7,
              height=7,
              rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=1, min_dist=8, max_dist=99999, seed=0),
              number_of_agents=2,
              obs_builder_object=TreeObsForRailEnv(max_depth=2))

# Print the observation vector for agent 0
obs, all_rewards, done, _ = env.step({0: 0})
for i in range(env.get_num_agents()):
    env.obs_builder.util_print_obs_subtree(tree=obs[i])

env_renderer = RenderTool(env)
env_renderer.renderEnv(show=True, frames=True)
env_renderer.renderEnv(show=True, frames=True)

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
            action = int(cmds[i + 1])
            action_dict[agent_id] = action
            i = i + 1
        i += 1

    env_renderer.renderEnv(show=True, frames=True)
