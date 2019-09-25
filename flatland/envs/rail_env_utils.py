import numpy as np
import matplotlib.pyplot as plt

from flatland.envs.distance_map import DistanceMap
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_file
from flatland.envs.schedule_generators import schedule_from_file


def load_flatland_environment_from_file(file_name, load_from_package=None, obs_builder_object=None):
    if obs_builder_object is None:
        obs_builder_object = TreeObsForRailEnv(
            max_depth=2,
            predictor=ShortestPathPredictorForRailEnv(max_depth=10))
    environment = RailEnv(width=1,
                          height=1,
                          rail_generator=rail_from_file(file_name, load_from_package),
                          number_of_agents=1,
                          schedule_generator=schedule_from_file(file_name, load_from_package),
                          obs_builder_object=obs_builder_object)
    return environment


def visualize_distance_map(distance_map: DistanceMap, agent_handle: int = 0):
    if agent_handle >= distance_map.get().shape[0]:
        print("Error: agent_handle cannot be larger than actual number of agents")
        return
    # take min value of all 4 directions
    min_distance_map = np.min(distance_map.get(), axis=3)
    plt.imshow(min_distance_map[agent_handle][:][:])
    plt.show()
