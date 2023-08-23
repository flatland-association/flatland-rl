import numpy as np

from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
import flatland.envs.observations as ObsRepository

# First of all we import the Flatland rail environment
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator

width = 30  # With of map
height = 30  # Height of map
nr_trains = 5  # Number of trains that have an assigned task in the env
cities_in_map = 3  # Number of cities where agents can start or end
seed = 14  # Random seed
grid_distribution_of_cities = False  # Type of city distribution, if False cities are randomly placed
max_rails_between_cities = 2  # Max number of tracks allowed between cities. This is number of entry point to a city
max_rail_in_cities = 2  # Max number of parallel tracks within a city, representing a realistic trainstation

rail_generator = sparse_rail_generator(max_num_cities=cities_in_map,
                                       seed=seed,
                                       grid_mode=grid_distribution_of_cities,
                                       max_rails_between_cities=max_rails_between_cities,
                                       max_rail_pairs_in_city=max_rail_in_cities,
                                       )

line_generator = sparse_line_generator()

# define here the observation to explore
########################################
# observation_builder = ObsRepository.GlobalObsStandardizedForRailEnv()
# observation_builder = ObsRepository.GlobalObsForRailEnv()

# observation_builder = ObsRepository.LocalObsStandardizedForRailEnv(3, 3, 0)
observation_builder = ObsRepository.LocalObsForRailEnv(3, 3, 0)

# observation_builder = ObsRepository.TreeObsStandardizedForRailEnv(max_depth=3)
# observation_builder = ObsRepository.TreeObsForRailEnv(max_depth=3)

# Construct the environment with the given observation, generators, predictors, and stochastic data
env = RailEnv(width=width,
              height=height,
              rail_generator=rail_generator,
              line_generator=line_generator,
              number_of_agents=nr_trains,
              obs_builder_object=observation_builder,
              remove_agents_at_target=True)

states = []
state, _ = env.reset()
states.append(state)

action_dict = {} #dict()
for agent_idx in range(len(env.agents)):
    action_dict[agent_idx] = RailEnvActions.MOVE_FORWARD   # Try to move with the agents


for k in range(10):
    # Do a step in the environment to get the observation
    next_obs, all_rewards, done, _ = env.step(action_dict)
    states.append(next_obs)


print(next_obs)


"""
Observations

1) GlobalObsForRailEnv

dict(0, ... , num_agents-1) with entries for each agent.
agent = tuple: 3
    0 = transition maps for each cell -> ndarray(width, height, 16)     16 = size of transistion map
    1 = agent states ndarray(width, height, 5)      5 states (channels): self agent pos and dir, other agents pos and dirs, self and others malfunctions, fractionals speeds
    2 = ndarray(width, height, 2)      2 channels: this agents target 0/1-flag, ohter agents targets 0/1-flag

2) TreeObsForRailEnv

dict(0, ... , num_agents-1) with entries for each agent.
agent = Node: 13 (was ist ein Node?)
    12 named Attribute (int, float)
    childs = Dict of 4 Nodes {L(eft), F(orward), R(ight), B(ack)} -> tree depth
    13 Attribute 00 .. 11 (int, float), 12 = Dict of 4 Nodes (redundant ?)

2a) FlattenTreeObsForRailEnv

ndarray(num_agents, 935)   935 = muss width, height und depth irgendwie enthalten ...


3) LocalObsForRailEnv (deprecated)

dict(0, ... , num_agents-1) with entries for each agent.
agent = tuple: 4
    0 = ndarray(width of local view, height of local view + 2*padding, 16)
    1 = ndarray(3, 7, 2)
    2 = ndarray(3, 7, 4)
    3 = ndarray(4,)


"""
