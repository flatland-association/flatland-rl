from flatland.envs.rail_env import *
from flatland.core.env_observation_builder import TreeObsForRailEnv
from flatland.utils.rendertools import *
from flatland.agents.dueling_double_dqn import Agent
random.seed(1)
np.random.seed(1)

"""
transition_probability = [1.0,  # empty cell - Case 0
                          3.0,  # Case 1 - straight
                          1.0,  # Case 2 - simple switch
                          3.0,  # Case 3 - diamond drossing
                          2.0,  # Case 4 - single slip
                          1.0,  # Case 5 - double slip
                          1.0,  # Case 6 - symmetrical
                          1.0]  # Case 7 - dead end
"""
transition_probability = [1.0,  # empty cell - Case 0
                          1.0,  # Case 1 - straight
                          0.5,  # Case 2 - simple switch
                          0.2,  # Case 3 - diamond drossing
                          0.5,  # Case 4 - single slip
                          0.1,  # Case 5 - double slip
                          0.2,  # Case 6 - symmetrical
                          0.01]  # Case 7 - dead end

# Example generate a random rail
env = RailEnv(width=20,
              height=20,
              rail_generator=random_rail_generator(cell_type_relative_proportion=transition_probability),
              number_of_agents=10)
env.reset()

env_renderer = RenderTool(env)
handle = env.get_agent_handles()

state_size = 105
action_size = 4
agent = Agent(state_size, action_size, "FC", 0)

# Example generate a rail given a manual specification,
# a map of tuples (cell_type, rotation)
specs = [[(0, 0), (0, 0), (0, 0), (0, 0), (7, 0), (0, 0)],
         [(7, 270), (1, 90), (1, 90), (1, 90), (2, 90), (7, 90)]]

env = RailEnv(width=6,
              height=2,
              rail_generator=rail_from_manual_specifications_generator(specs),
              number_of_agents=1,
              obs_builder_object=TreeObsForRailEnv(max_depth=2))


env.agents_position[0] = [1, 4]
env.agents_target[0] = [1, 1]
env.agents_direction[0] = 1
# TODO: watch out: if these variables are overridden, the obs_builder object has to be reset, too!
env.obs_builder.reset()

# TODO: delete next line
#for i in range(4):
#    print(env.obs_builder.distance_map[0, :, :, i])

obs, all_rewards, done, _ = env.step({0:0})
#env.obs_builder.util_print_obs_subtree(tree=obs[0], num_elements_per_node=5)

env_renderer = RenderTool(env)
action_dict = {0: 0}

for step in range(100):
    obs, all_rewards, done, _ = env.step(action_dict)
    action = agent.act(np.array(obs[0]),eps=1)

    action_dict = {0 :action}
    print("Rewards: ", all_rewards, "  [done=", done, "]")

