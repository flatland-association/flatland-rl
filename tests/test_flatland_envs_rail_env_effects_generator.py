from flatland.core.effects_generator import EffectsGenerator
from flatland.env_generation.env_generator import env_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions


class TestMalfunctionEffectsGenerator(EffectsGenerator[RailEnv]):
    def __call__(self, env: RailEnv, *args, **kwargs) -> RailEnv:
        env.agents[0].malfunction_handler._set_malfunction_down_counter(999999)


def test_rail_env_effects_generator():
    env, _, _ = env_generator()
    env.effects_generator = TestMalfunctionEffectsGenerator()
    env.step({i: RailEnvActions.MOVE_FORWARD for i in env.get_agent_handles()})
    assert env.agents[0].malfunction_handler.malfunction_down_counter == 999999
