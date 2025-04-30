import pytest

from flatland.env_generation.env_generator import env_generator
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.ml.observations.flatten_tree_observation_for_rail_env import FlattenedNormalizedTreeObsForRailEnv
from flatland.ml.ray.wrappers import ray_multi_agent_env_wrapper


def test_ray_multi_agent_env_wrapper_non_compatible_observation_builder():
    env, _, _ = env_generator()
    with pytest.raises(AssertionError) as exc:
        ray_multi_agent_env_wrapper(env)
    assert str(exc.value) == "<class 'flatland.envs.observations.TreeObsForRailEnv'> is not gym-compatible, missing get_observation_space"


def test_ray_multi_agent_env_wrapper():
    raw_env, _, _ = env_generator(obs_builder_object=FlattenedNormalizedTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)))
    env = ray_multi_agent_env_wrapper(raw_env)
    observations, infos = env.reset()
    assert len(observations) == 7
    assert len(infos) == 7
    print(observations)
    for h in [str(h) for h in range(7)]:
        assert observations[h].shape == (1020,)
        assert isinstance(infos[h], dict)
    observations, rewards, terminations, truncations, infos = env.step({h: RailEnvActions.MOVE_FORWARD for h in env.wrap.get_agent_handles()})
    assert len(observations) == 7
    assert len(rewards) == 7
    assert len(terminations) == 8
    assert "__all__" in terminations
    assert len(truncations) == 1
    assert len(infos) == 7
    for h in [str(h) for h in range(7)]:
        assert observations[h].shape == (1020,)
        assert isinstance(rewards[h], int)
        assert isinstance(terminations[h], bool)
        assert isinstance(infos[h], dict)
    for v in terminations.values():
        assert isinstance(v, bool)
    for v in truncations.values():
        assert isinstance(v, bool)
