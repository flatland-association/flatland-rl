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
