from flatland.core.env_observation_builder import AgentHandle
from flatland.env_generation.env_creator import env_creator
from flatland.ml.observations.gym_observation_builder import DummyObservationBuilderGym, GlobalObsForRailEnvGym


def test_dummy_observation_builder_gym():
    obs_builder = DummyObservationBuilderGym()
    obs = obs_builder.get()
    assert obs.shape == (1,)
    assert obs.dtype == float
    assert obs[0] == 1.0

    obs = obs_builder.get_many(range(7))
    assert len(obs) == 7
    for k in obs.keys():
        assert type(k) == AgentHandle
    for o in obs.values():
        assert o.shape == (1,)
        assert o.dtype == float
        assert o[0] == 1.0


def test_global_obs_for_rail_env():
    obs_builder = GlobalObsForRailEnvGym()
    env = env_creator(obs_builder_object=obs_builder)
    obs = obs_builder.get()
    assert obs.shape == (env.width * env.height * (16 + 5 + 2),)
    assert obs.dtype == float

    obs = obs_builder.get_many(range(7))
    assert len(obs) == 7
    for k in obs.keys():
        assert type(k) == AgentHandle
    for o in obs.values():
        assert o.shape == (env.width * env.height * (16 + 5 + 2),)
        assert o.dtype == float
