import pytest

from flatland.env_generation.env_creator import env_creator
from flatland.ml.ray.wrappers import ray_multi_agent_env_wrapper


def test_ray_multi_agent_env_wrapper():
    env = env_creator()
    with pytest.raises(AssertionError) as exc:
        ray_multi_agent_env_wrapper(env)
    assert str(exc.value) == "<class 'flatland.envs.observations.TreeObsForRailEnv'> is not gym-compatible, missing get_observation_space"
