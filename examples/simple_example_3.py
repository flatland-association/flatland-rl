import random

from flatland.envs.generators import random_rail_generator, random_rail_generator
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from flatland.core.env_observation_builder import ObservationBuilder
import numpy as np

random.seed(100)
np.random.seed(100)

env = RailEnv(width=7,
              height=7,
              rail_generator=random_rail_generator(),
              number_of_agents=2,
              obs_builder_object=TreeObsForRailEnv(max_depth=2))

# Print the distance map of each cell to the target of the first agent
# for i in range(4):
#     print(env.obs_builder.distance_map[0, :, :, i])

# Print the observation vector for agent 0
obs, all_rewards, done, _ = env.step({0: 0})
for i in range(env.get_num_agents()):
    env.obs_builder.util_print_obs_subtree(tree=obs[i], num_features_per_node=5)

env_renderer = RenderTool(env, gl="QT")
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
            action = int(cmds[i + 1])
            action_dict[agent_id] = action
            i = i + 1
        i += 1

    env_renderer.renderEnv(show=True)
