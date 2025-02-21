import numpy as np

from flatland.core.env_observation_builder import AgentHandle
from flatland.env_generation.env_generator import env_generator
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.ml.observations.flatten_tree_observation_for_rail_env import FlattenedNormalizedTreeObsForRailEnv


def test_flatten_tree_obs_for_rail_env():
    obs_builder = FlattenedNormalizedTreeObsForRailEnv(max_depth=1, predictor=ShortestPathPredictorForRailEnv(max_depth=10))
    env_generator(n_agents=7, obs_builder_object=obs_builder)
    obs = obs_builder.get()
    assert obs.dtype == float
    assert obs.shape == (5 * 12,)

    env_generator(obs_builder_object=obs_builder, seed=42)
    obs = obs_builder.get_many(list(range(7)))

    assert len(obs) == 7
    for k in obs.keys():
        assert type(k) == AgentHandle
    for o in obs.values():
        assert o.shape == (5 * 12,)
    expected_obs = {
        0: np.array([0., 0., 0., 0., 0.,
                     0., -1., -1., -1., -1.,
                     -1., -1., 1., 1., 1.,
                     1., 1., 1., -1., -1.,
                     -1., -1., -1., -1., -1.,
                     -1., -1., -1., -1., -1.,
                     0.83333333, -1., 0., -1., -1.,
                     0., 0., 0., 0.25, 0.,
                     -1., -1., -1., -1., -1.,
                     0., 0., 0., 1., 0.,
                     -1., -1., -1., -1., -1.,
                     -1., -1., -1., -1., -1.]),
        1: np.array([0., 0., 0., 0., 0., 0., -1., -1., -1., -1., -1.,
                     -1., 1., 1., 1., 1., 1., 1., -1., -1., -1., -1.,
                     -1., -1., -1., -1., -1., -1., -1., -1., 0.8, -1., 0.,
                     -1., -1., 0., 0., 0., 1., 0., -1., -1., -1., -1.,
                     -1., 0., 0., 0., 1., 0., -1., -1., -1., -1., -1.,
                     -1., -1., -1., -1., -1.]),
        2: np.array([0., 0., 0., 0., 0.,
                     0., -1., -1., -1., -1.,
                     -1., -1., 1., 1., 1.,
                     1., 1., 1., -1., -1.,
                     -1., -1., -1., -1., -1.,
                     -1., -1., -1., -1., -1.,
                     0.97368421, -1., 0., -1., -1.,
                     0., 0., 0., 0.25, 0.,
                     -1., -1., -1., -1., -1.,
                     0., 0., 0., 1., 0.,
                     -1., -1., -1., -1., -1.,
                     -1., -1., -1., -1., -1.]),
        3: np.array([0., 0., 0., 0., 0.,
                     0., -1., -1., -1., -1.,
                     -1., -1., 1., 1., 1.,
                     1., 1., 1., -1., -1.,
                     -1., -1., -1., -1., -1.,
                     -1., -1., -1., -1., -1.,
                     0.83333333, -1., 0., -1., -1.,
                     0., 0., 0., 1., 0.,
                     -1., -1., -1., -1., -1.,
                     0., 0., 0., 1., 0.,
                     -1., -1., -1., -1., -1.,
                     -1., -1., -1., -1., -1.]),
        4: np.array([0., 0., 0., 0., 0.,
                     0., -1., -1., -1., -1.,
                     -1., -1., 1., 1., 1.,
                     1., 1., 1., -1., -1.,
                     -1., -1., -1., -1., -1.,
                     -1., -1., -1., -1., -1.,
                     0.85714286, -1., 0., -1., -1.,
                     0., 0., 0., 1., 0.,
                     -1., -1., -1., -1., -1.,
                     0., 0., 0., 1., 0.,
                     -1., -1., -1., -1., -1.,
                     -1., -1., -1., -1., -1.]),
        5: np.array([0., 0., 0., 0., 0.,
                     0., -1., -1., -1., -1.,
                     -1., -1., 1., 1., 1.,
                     1., 1., 1., -1., -1.,
                     -1., -1., -1., -1., -1.,
                     -1., -1., -1., -1., -1.,
                     0.83333333, -1., 0., -1., -1.,
                     0., 0., 0., 0.25, 0.,
                     -1., -1., -1., -1., -1.,
                     0., 0., 0., 1., 0.,
                     -1., -1., -1., -1., -1.,
                     -1., -1., -1., -1., -1.]),
        6: np.array([0., 0., 0., 0., 0.,
                     0., -1., -1., -1., -1.,
                     -1., -1., 1., 1., 1.,
                     1., 1., 1., -1., -1.,
                     -1., -1., -1., -1., -1.,
                     -1., -1., -1., -1., -1.,
                     0.83333333, -1., 0., -1., -1.,
                     0., 0., 0., 0.5, 0.,
                     -1., -1., -1., -1., -1.,
                     0., 0., 0., 1., 0.,
                     -1., -1., -1., -1., -1.,
                     -1., -1., -1., -1., -1.])}
    assert set(obs.keys()) == set(expected_obs.keys())
    for actual, expected in zip(obs.values(), expected_obs.values()):
        assert np.allclose(actual, expected)
