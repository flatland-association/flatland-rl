import random
import numpy as np
import matplotlib.pyplot as plt

from flatland.envs.rail_env import *
from flatland.envs.generators import *
from flatland.core.env_observation_builder import TreeObsForRailEnv
from flatland.utils.rendertools import *

random.seed(0)
np.random.seed(0)

transition_probability = [1.0,  # empty cell - Case 0
                          1.0,  # Case 1 - straight
                          1.0,  # Case 2 - simple switch
                          0.3,  # Case 3 - diamond drossing
                          0.5,  # Case 4 - single slip
                          0.5,  # Case 5 - double slip
                          0.2,  # Case 6 - symmetrical
                          0.0,  # Case 7 - dead end
                          0.2,  # Case 8 - turn left
                          0.2,  # Case 9 - turn right
                          1.0]  # Case 10 - mirrored switch

"""
# Example generate a random rail
env = RailEnv(width=20,
              height=20,
              rail_generator=random_rail_generator(cell_type_relative_proportion=transition_probability),
              number_of_agents=10)

# env = RailEnv(width=20,
#               height=20,
#               rail_generator=rail_from_list_of_saved_GridTransitionMap_generator(['examples/sample_10_10_rail.npy']),
#               number_of_agents=10)

env.reset()

env_renderer = RenderTool(env)
env_renderer.renderEnv(show=True)
"""
"""
# Example generate a rail given a manual specification,
# a map of tuples (cell_type, rotation)
specs = [[(0, 0), (0, 0), (0, 0), (0, 0), (7, 0), (0, 0)],
         [(7, 270), (1, 90), (1, 90), (1, 90), (2, 90), (7, 90)]]

env = RailEnv(width=6,
              height=2,
              rail_generator=rail_from_manual_specifications_generator(specs),
              number_of_agents=1,
              obs_builder_object=TreeObsForRailEnv(max_depth=2))

handle = env.get_agent_handles()
env.agents_position[0] = [1, 4]
env.agents_target[0] = [1, 1]
env.agents_direction[0] = 1
# TODO: watch out: if these variables are overridden, the obs_builder object has to be reset, too!
env.obs_builder.reset()
"""
"""
# INFINITE-LOOP TEST
specs = [[(0, 0), (0, 0), (0, 0), (0, 0), (7, 0), (0, 0)],
         [(7, 270), (1, 90), (1, 90), (2, 270), (2, 0), (0, 0)],
         [(0, 0), (0, 0), (0, 0), (2, 180), (2, 90), (7, 90)],
         [(0, 0), (0, 0), (0, 0), (7, 180), (0, 0), (0, 0)]]

# CURVED RAIL + DEAD-ENDS TEST
specs = [[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
         [(7, 270), (1, 90), (1, 90), (8, 90), (0, 0), (0, 0)],
         [(0, 0),   (7, 270),(1, 90), (8, 180), (0, 00), (0, 0)]]

env = RailEnv(width=6,
              height=4,
              rail_generator=rail_from_manual_specifications_generator(specs),
              number_of_agents=1,
              obs_builder_object=TreeObsForRailEnv(max_depth=2))

handle = env.get_agent_handles()
env.agents_position[0] = [1, 3]
env.agents_target[0] = [1, 1]
env.agents_direction[0] = 1
# TODO: watch out: if these variables are overridden, the obs_builder object has to be reset, too!
env.obs_builder.reset()
"""
env = RailEnv(width=7,
              height=7,
              rail_generator=random_rail_generator(cell_type_relative_proportion=transition_probability),
              # rail_generator=complex_rail_generator(nr_start_goal=2),
              number_of_agents=2)

# Print the distance map of each cell to the target of the first agent
# for i in range(4):
#     print(env.obs_builder.distance_map[0, :, :, i])

# Print the observation vector for agent 0
obs, all_rewards, done, _ = env.step({0:0})
for i in range(env.number_of_agents):
    env.obs_builder.util_print_obs_subtree(tree=obs[i], num_features_per_node=5)

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
