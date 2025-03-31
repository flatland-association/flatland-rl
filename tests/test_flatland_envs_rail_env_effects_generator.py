from flatland.core.effects_generator import EffectsGenerator
from flatland.env_generation.env_generator import env_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions


def test_rail_env_effects_generator_on_episode_start():
    class TestMalfunctionEffectsGenerator(EffectsGenerator[RailEnv]):
        def on_episode_start(self, env: RailEnv, *args, **kwargs) -> RailEnv:
            env.agents[0].malfunction_handler._set_malfunction_down_counter(999999)

    env, _, _ = env_generator(effects_generator=TestMalfunctionEffectsGenerator())
    assert env.agents[0].malfunction_handler.malfunction_down_counter == 999999
    env.step({i: RailEnvActions.MOVE_FORWARD for i in env.get_agent_handles()})
    assert env.agents[0].malfunction_handler.malfunction_down_counter == 999998


def test_rail_env_effects_generator_on_episode_step_start():
    class TestMalfunctionEffectsGenerator(EffectsGenerator[RailEnv]):
        def on_episode_step_start(self, env: RailEnv, *args, **kwargs) -> RailEnv:
            env.agents[0].malfunction_handler._set_malfunction_down_counter(999999)

    env, _, _ = env_generator(effects_generator=TestMalfunctionEffectsGenerator())
    assert env.agents[0].malfunction_handler.malfunction_down_counter == 0
    env.step({i: RailEnvActions.MOVE_FORWARD for i in env.get_agent_handles()})
    assert env.agents[0].malfunction_handler.malfunction_down_counter == 999998


def test_rail_env_effects_generator_on_episode_step_end():
    class TestMalfunctionEffectsGenerator(EffectsGenerator[RailEnv]):
        def on_episode_step_end(self, env: RailEnv, *args, **kwargs) -> RailEnv:
            env.agents[0].malfunction_handler._set_malfunction_down_counter(999999)

    env, _, _ = env_generator(effects_generator=TestMalfunctionEffectsGenerator())
    assert env.agents[0].malfunction_handler.malfunction_down_counter == 0
    env.step({i: RailEnvActions.MOVE_FORWARD for i in env.get_agent_handles()})
    assert env.agents[0].malfunction_handler.malfunction_down_counter == 999999
