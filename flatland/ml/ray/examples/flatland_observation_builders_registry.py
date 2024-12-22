from ray.tune.registry import register_input

from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.ml.observations.flatten_tree_observation_for_rail_env import FlattenNormalizedTreeObsForRailEnv
from flatland.ml.observations.gym_observation_builder import DummyObservationBuilderGym, GlobalObsForRailEnvGym


def register_flatland_ray_cli_observation_builders():
    register_input("DummyObservationBuilderGym", lambda: DummyObservationBuilderGym()),
    register_input("GlobalObsForRailEnvGym", lambda: GlobalObsForRailEnvGym()),
    register_input("FlattenNormalizedTreeObsForRailEnv_max_depth_3_50",
                   lambda: FlattenNormalizedTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)))
