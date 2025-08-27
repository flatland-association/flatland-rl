from flatland.env_generation.env_generator import env_generator
from flatland.envs.malfunction_effects_generators import ConditionalMalfunctionEffectsGenerator, condition_stopped_cells_and_range, \
    condition_stopped_intermediate_and_range, make_multi_malfunction_condition
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.step_utils.states import TrainState
from flatland.utils.rendertools import RenderTool


def test_conditional_stopped_cells_and_range_malfunction_effects_generator():
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


def test_no_effect_conditional_stopped_cells_and_range_malfunction_effects_generator():
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


def test_conditional_stopped_intermediate_and_range_malfunction_effects_generator(rendering: bool = False):
    env, _, _ = env_generator(
        line_length=3,
        n_cities=3,
        n_agents=3,
        effects_generator=ConditionalMalfunctionEffectsGenerator(
            malfunction_rate=1,
            min_duration=888,
            max_duration=888,
            condition=condition_stopped_intermediate_and_range(0, 9999999),
        ))
    env.reset()

    if rendering:
        env_renderer = RenderTool(env)
    for _ in range(400):
        if rendering:
            env_renderer.render_env(show=True)
        if env.dones["__all__"]:
            break
        actions = dict()
        for agent in env.agents:
            if agent.position is None:
                actions[agent.handle] = RailEnvActions.MOVE_FORWARD
            elif agent.position == agent.waypoints[1].position:
                actions[agent.handle] = RailEnvActions.STOP_MOVING
            else:
                p = get_k_shortest_paths(env, agent.position, agent.direction, agent.waypoints[1].position)
                shortest_path = p[0]
                for a in {RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT}:
                    new_cell_valid, new_direction, new_position, transition_valid, preprocessed_action = env.rail.check_action_on_agent(
                        RailEnvActions.from_value(a),
                        agent.position,
                        agent.direction
                    )
                    if new_cell_valid and transition_valid and new_position == shortest_path[1].position and new_direction == shortest_path[1].direction:
                        actions[agent.handle] = a
                        break
        env.step(actions)

    intermediate_waypoints = {w.position for agent in env.agents for w in agent.waypoints[1:-1]}
    in_malfunction = dict()
    for agent in env.agents:
        if agent.malfunction_handler.in_malfunction:
            in_malfunction[agent.position] = agent

    # there is an agent stopped by the conditional malfunction generator at each waypoint
    assert len(intermediate_waypoints) == 3
    assert len(in_malfunction) == len(intermediate_waypoints)
    for _, agents in in_malfunction.items():
        assert agent.malfunction_handler.malfunction_down_counter > 700


def test_make_multi_malfunction_condition():
    env, _, _ = env_generator(
        line_length=3,
        n_cities=3,
        n_agents=3,
        effects_generator=ConditionalMalfunctionEffectsGenerator(
            malfunction_rate=1,
            min_duration=888,
            max_duration=888,
            condition=condition_stopped_intermediate_and_range(0, 9999999),
        ))

    cond = make_multi_malfunction_condition(
        [condition_stopped_intermediate_and_range(44, 99), condition_stopped_cells_and_range(44, 99, [env.agents[0].initial_position])])

    env.agents[0].state_machine.set_state(TrainState.STOPPED)
    env.agents[0].position = env.agents[0].initial_position
    assert cond(env.agents[0], 55)
    assert not cond(env.agents[0], 33)
    assert not cond(env.agents[0], 100)

    env.agents[0].state_machine.set_state(TrainState.STOPPED)
    env.agents[0].position = env.agents[0].waypoints[1].position
    assert cond(env.agents[0], 55)
    assert not cond(env.agents[0], 33)
    assert not cond(env.agents[0], 100)
