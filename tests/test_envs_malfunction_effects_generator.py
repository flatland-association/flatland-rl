from flatland.env_generation.env_generator import env_generator
from flatland.envs.malfunction_effects_generators import ConditionalMalfunctionEffectsGenerator, condition_stopped_cells_and_range
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState


def test_conditional_malfunction_effects_generator():
    env, _, _ = env_generator(effects_generator=ConditionalMalfunctionEffectsGenerator(
        malfunction_rate=1,
        min_duration=888,
        max_duration=888,
        # all cells
        condition=condition_stopped_cells_and_range(0, 9999999, [(r, c) for r in range(30) for c in range(30)])
    ))
    env.reset()

    for _ in range(150):
        env.step({agent.handle: RailEnvActions.STOP_MOVING if agent.state == TrainState.MOVING else RailEnvActions.MOVE_FORWARD for agent in env.agents})

    initial_positions = {agent.initial_position for agent in env.agents}
    in_malfunction = dict()
    for agent in env.agents:
        if agent.malfunction_handler.in_malfunction:
            in_malfunction[agent.position] = agent

    # there is an agent stopped by the conditional malfunction generator at each initial position
    assert len(in_malfunction) == len(initial_positions)
    for _, agents in in_malfunction.items():
        assert agent.malfunction_handler.malfunction_down_counter > 700


def test_no_effect_conditional_malfunction_effects_generator():
    env, _, _ = env_generator(effects_generator=ConditionalMalfunctionEffectsGenerator(
        malfunction_rate=0,
        min_duration=888,
        max_duration=888,
        # all cells
        condition=condition_stopped_cells_and_range(0, 9999999, [(r, c) for r in range(30) for c in range(30)])
    ))
    env.reset()

    for _ in range(150):
        env.step({agent.handle: RailEnvActions.STOP_MOVING if agent.state == TrainState.MOVING else RailEnvActions.MOVE_FORWARD for agent in env.agents})

    # no malfunction generated although condition applies as above
    for agent in env.agents:
        assert agent.malfunction_handler.malfunction_down_counter <= 50
