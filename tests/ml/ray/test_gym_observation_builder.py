import pytest

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.ml.observations.flatten_tree_observation_for_rail_env import FlattenTreeObsForRailEnv
from flatland.ml.observations.gym_observation_builder import DummyObservationBuilderGym, GlobalObsForRailEnvGym
from flatland.ml.ray.wrappers import ray_env_creator


@pytest.mark.parametrize(
    "obs_builder,expected_shape",
    [
        pytest.param(obs_builder, expected_shape, id=f"{obid}")
        for obs_builder, obid, expected_shape in
        [
            (FlattenTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)), "FlattenTreeObsForRailEnv_max_depth_3_50", (935,)),
            (DummyObservationBuilderGym(), "DummyObservationBuilderGym", (1,)),
            (GlobalObsForRailEnvGym(), "GlobalObsForRailEnvGym", (20700,)),
        ]
    ]
)
def test_obs_builder_gym(obs_builder: ObservationBuilder, expected_shape):
    expected_dtype = float
    expected_agent_ids = ['0', '1', '2', '3', '4', '5', '6']

    env = ray_env_creator(obs_builder_object=obs_builder)

    assert env.agents == expected_agent_ids, env.agents
    for agent_id in env.agents:
        space_shape = env.get_observation_space(agent_id).shape
        assert space_shape == expected_shape, (expected_shape, space_shape)
        space_dtype = env.get_observation_space(agent_id).dtype
        assert space_dtype == expected_dtype
        sample_shape = env.get_observation_space(agent_id).sample().shape
        assert sample_shape == expected_shape, (expected_shape, sample_shape)
    obs, _ = env.reset()
    assert list(obs.keys()) == expected_agent_ids
    for i in range(7):
        assert obs[str(i)].shape == expected_shape
        assert obs[str(i)].dtype == expected_dtype
    obs, _, _, _, _ = env.step({})
    assert list(obs.keys()) == expected_agent_ids
    for i in range(7):
        assert obs[str(i)].shape == expected_shape
        assert obs[str(i)].dtype == expected_dtype
