from flatland.core.env_observation_builder import AgentHandle
from flatland.env_generation.env_creator import env_creator
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.ml.observations.flatten_tree_observation_for_rail_env import FlattenTreeObsForRailEnv


def test_flatten_tree_obs_for_rail_env():
    obs_builder = FlattenTreeObsForRailEnv(max_depth=1, predictor=ShortestPathPredictorForRailEnv(max_depth=10))
    env_creator(n_agents=7, obs_builder_object=obs_builder)
    obs = obs_builder.get()
    assert obs.dtype == float
    assert obs.shape == (5 * 12,)

    env_creator(obs_builder_object=obs_builder)
    obs = obs_builder.get_many(list(range(7)))

    assert len(obs) == 7
    for k in obs.keys():
        assert type(k) == AgentHandle
    for o in obs.values():
        assert o.shape == (5 * 12,)
