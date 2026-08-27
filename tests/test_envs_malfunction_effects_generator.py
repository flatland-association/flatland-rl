import sys

import numpy as np

from flatland.env_generation.env_generator import env_generator, env_generator_legacy
from flatland.envs.malfunction_effects_generators import ConditionalMalfunctionEffectsGenerator, condition_stopped_cells_and_range, \
    condition_stopped_intermediate_and_range, make_multi_malfunction_condition, IntermediateStopMalfunctionEffectsGenerator
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState
from flatland.utils.rendertools import RenderTool


def test_conditional_stopped_cells_and_range_malfunction_effects_generator():
    env, _, _ = env_generator_legacy(seed=42,
                              malfunction_interval=sys.maxsize,  # disable conventional malfunction generator
                              effects_generator=ConditionalMalfunctionEffectsGenerator(
                                  malfunction_rate=1,
                                  min_duration=888,
                                  max_duration=888,
                                  # all cells
                                  condition=condition_stopped_cells_and_range(0, 9999999, [(r, c) for r in range(30) for c in range(30)])
                              ))
    env.reset()

    for _ in range(150):
        env.step({agent.handle: RailEnvActions.STOP_MOVING if agent.state == TrainState.MOVING else RailEnvActions.MOVE_FORWARD for agent in env.agents})

    initial_positions = {agent.initial_entry_point[0] for agent in env.agents}
    in_malfunction = [agent for agent in env.agents if agent.malfunction_handler.in_malfunction]

    # there is an agent stopped by the conditional malfunction generator at each initial position
    assert {agent.initial_entry_point[0] for agent in in_malfunction} == initial_positions
    for agent in in_malfunction:
        # a STOP_MOVING may complete an in-flight crossing before halting (pre-step momentum already
        # past the cell boundary), so the agent may be stopped one cell into the successor entry point
        # reached by that crossing rather than exactly at its initial entry point
        assert agent.current_entry_point == agent.initial_entry_point or agent.current_entry_point in env.rail.get_successor_entry_points(
            agent.initial_entry_point)
        assert agent.malfunction_handler.malfunction_down_counter > 700


def test_no_effect_conditional_stopped_cells_and_range_malfunction_effects_generator():
    env, _, _ = env_generator(
        seed=42,
        malfunction_interval=sys.maxsize,  # disable conventional malfunction generator
        effects_generator=ConditionalMalfunctionEffectsGenerator(
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
    env, _, _ = env_generator_legacy(
        seed=42,
        line_length=3,
        n_cities=3,
        n_agents=3,
        malfunction_interval=sys.maxsize,  # disable conventional malfunction generator
        effects_generator=ConditionalMalfunctionEffectsGenerator(
            malfunction_rate=1,
            min_duration=888,
            max_duration=888,
            condition=condition_stopped_intermediate_and_range(0, 9999999),
        ))
    env.reset()

    _run_with_sthortest_path(env=env, rendering=rendering, num_steps=400)

    intermediate_waypoints = {w.position for agent in env.agents for ws in agent.waypoints[1:-1] for w in ws}
    in_malfunction = dict()
    for agent in env.agents:
        if agent.malfunction_handler.in_malfunction:
            in_malfunction[agent.current_entry_point[0]] = agent

    # there is an agent stopped by the conditional malfunction generator at each waypoint
    assert len(intermediate_waypoints) == 3
    assert len(in_malfunction) == len(intermediate_waypoints)
    for _, agents in in_malfunction.items():
        assert agent.malfunction_handler.malfunction_down_counter > 700


def test_make_multi_malfunction_condition():
    env, _, _ = env_generator_legacy(seed=42,
                              line_length=3,
                              n_cities=3,
                              n_agents=3,
                              malfunction_interval=sys.maxsize,  # disable conventional malfunction generator
                              effects_generator=ConditionalMalfunctionEffectsGenerator(
                                  malfunction_rate=1,
                                  min_duration=888,
                                  max_duration=888,
                                  condition=condition_stopped_intermediate_and_range(0, 9999999),
                              ))

    cond = make_multi_malfunction_condition(
        [condition_stopped_intermediate_and_range(44, 99),
         condition_stopped_cells_and_range(44, 99, [env.agents[0].initial_entry_point[0]])])

    env.agents[0].state_machine.set_state(TrainState.STOPPED)
    direction = env.agents[0].current_entry_point[1] if env.agents[0].current_entry_point is not None else None
    env.agents[0].current_entry_point = (env.agents[0].initial_entry_point[0], direction)
    assert cond(env.agents[0], 55)
    assert not cond(env.agents[0], 33)
    assert not cond(env.agents[0], 100)

    env.agents[0].state_machine.set_state(TrainState.STOPPED)
    direction = env.agents[0].current_entry_point[1] if env.agents[0].current_entry_point is not None else None
    env.agents[0].current_entry_point = (env.agents[0].waypoints[1][0].position, direction)
    assert cond(env.agents[0], 55)
    assert not cond(env.agents[0], 33)
    assert not cond(env.agents[0], 100)


def test_conditional_earliest_and_max_num_malfunction(rendering: bool = False):
    duration = 888
    earliest = 77
    conditional_malfunction_effects_generator = ConditionalMalfunctionEffectsGenerator(
        malfunction_rate=sys.maxsize,
        min_duration=duration, max_duration=duration,
        earliest_malfunction=earliest,
        max_num_malfunctions=2,
    )
    env, _, _ = env_generator_legacy(
        seed=42,
        line_length=3,
        n_cities=3,
        n_agents=3,
        malfunction_interval=sys.maxsize,  # disable conventional malfunction generator
        effects_generator=conditional_malfunction_effects_generator)
    env.reset()

    num_steps_run = 150
    _run_with_sthortest_path(env, rendering, num_steps_run)

    in_malfunction = [agent for agent in env.agents if agent.malfunction_handler.in_malfunction]

    # there is an agent stopped by the conditional malfunction generator at each waypoint
    assert conditional_malfunction_effects_generator._num_malfunctions == 2
    assert len(in_malfunction) == 2
    for agent in in_malfunction:
        assert agent.malfunction_handler.malfunction_down_counter == duration - (num_steps_run - earliest)


# TODO https://github.com/flatland-association/flatland-rl/issues/386 use ShortestPathPolicy instead
def _run_with_sthortest_path(env, rendering, num_steps=400, stop_at_first_intermediate=True):
    if rendering:
        env_renderer = RenderTool(env)
    agents_at = {agent.handle: 0 for agent in env.agents}
    for _ in range(num_steps):
        if rendering:
            env_renderer.render_env(show=True)
        if env.dones["__all__"]:
            break
        actions = dict()
        for agent in env.agents:
            if agent.current_entry_point is None:
                actions[agent.handle] = RailEnvActions.MOVE_FORWARD
            elif agent.state == TrainState.DONE:
                continue
            else:

                next_waypoint_position = agent.waypoints[agents_at[agent.handle] + 1][0].position
                if agent.current_entry_point[0] == next_waypoint_position:
                    if stop_at_first_intermediate:
                        actions[agent.handle] = RailEnvActions.STOP_MOVING
                        continue
                    else:
                        agents_at[agent.handle] += 1
                        actions[agent.handle] = RailEnvActions.STOP_MOVING
                        continue
                # design: actions applied at cell entry -- once the agent's pre-step momentum is
                # about to carry it across the cell boundary into next_entry_point this step
                # (is_cell_exit), that crossing happens regardless of the action given, so if the
                # already-locked-in target is next_waypoint_position, STOP_MOVING must be given NOW,
                # one step before arrival is observable via current_entry_point - otherwise the
                # agent's pre-step momentum carries it one cell past the waypoint before halting.
                target_locked_in = agent.next_entry_point is not None \
                    and agent.next_entry_point[0] == next_waypoint_position
                if target_locked_in and agent.speed_counter.is_cell_exit():
                    actions[agent.handle] = RailEnvActions.STOP_MOVING
                    continue
                # design: actions applied at cell entry -- once next_entry_point is already
                # pending, this step's action decides the look-ahead beyond it, not the arrival
                # at next_waypoint_position itself, so compute it from next_entry_point instead of
                # current_entry_point. The path below is queried fresh starting at lookahead_from
                # itself, so its index 1 (one hop beyond the query's own start) is always the
                # right look-ahead target, whether lookahead_from is current or next_entry_point.
                has_pending_target = agent.next_entry_point is not None and agent.next_entry_point != agent.current_entry_point
                lookahead_from = agent.next_entry_point if has_pending_target else agent.current_entry_point

                p = get_k_shortest_paths(env, lookahead_from[0], lookahead_from[1],
                                         next_waypoint_position)
                shortest_path = p[0]
                if 1 >= len(shortest_path):
                    continue
                for a in {RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT}:
                    new_cell_valid, (new_position, new_direction), transition_valid, preprocessed_action, _ = env.rail._check_action_on_agent(
                        RailEnvActions.from_value(a), lookahead_from)
                    next_wp = shortest_path[1]
                    if (new_cell_valid and transition_valid
                        and new_position == next_wp.position and new_direction == next_wp.direction):
                        actions[agent.handle] = a
                        break
        env.step(actions)
    return agent


def test_intermediate_stop_malfunction_effects_generator(rendering: bool = False):
    # rate=1 gives prob 0.62 =  1 - exp(-rate)
    conditional_malfunction_effects_generator = IntermediateStopMalfunctionEffectsGenerator(np.inf, 1, 3, )
    env, _, _ = env_generator(
        seed=889,
        line_length=3,
        n_cities=5,
        n_agents=3,
        x_dim=50,
        y_dim=50,
        malfunction_interval=sys.maxsize,  # disable conventional malfunction generator
        effects_generator=conditional_malfunction_effects_generator,
    )
    env.reset()

    # `SparseLineGen` now explodes every stop (not just the target) to all reachable direction
    # alternatives, so intermediate waypoints for this seed legitimately gained extra alternatives
    # compared to when this test's expectations were pinned down. That has two knock-on effects computed
    # during `env.reset()`, before the override below runs: (1) `IntermediateStopMalfunctionEffectsGenerator`'s
    # condition matches against the full set of (position, direction) alternatives across all waypoints, so a
    # larger alternative set shifts malfunction timing; (2) the timetable generator computes each agent's
    # departure/arrival window from the (larger) alternative sets too, shifting schedules independent of (1)
    # (e.g. agent 1's earliest departure moves from step 108 to 458). Pin both the waypoints/targets and the
    # schedule back to their pre-generalization values so this test keeps exercising the malfunction-effects
    # logic it's actually about, independent of that unrelated behavior change.
    agent_waypoints = {
        0: [[((15, 29), 1)], [((37, 39), 2)], [((14, 9), 0)]],
        1: [[((14, 8), 2)], [((37, 33), 2)], [((12, 29), 1)]],
        2: [[((14, 6), 0)], [((37, 39), 2)], [((12, 29), 1)]],
    }
    agent_earliest_departure = {0: [491, 526, None], 1: [108, 177, None], 2: [63, 134, None]}
    agent_latest_arrival = {0: [None, 940, 1012], 1: [None, 789, 895], 2: [None, 981, 1087]}
    for agent in env.agents:
        agent.waypoints = [[Waypoint(position, direction) for position, direction in group] for group in agent_waypoints[agent.handle]]
        agent.targets = set(agent_waypoints[agent.handle][-1])
        agent.waypoints_earliest_departure = agent_earliest_departure[agent.handle]
        agent.waypoints_latest_arrival = agent_latest_arrival[agent.handle]
        # the scalar `earliest_departure`/`latest_arrival` (not just the per-leg lists above) gate the
        # WAITING -> READY_TO_DEPART transition (see `rail_env.py`'s `_elapsed_steps >= agent.earliest_departure`)
        # and must be pinned too.
        agent.earliest_departure = agent_earliest_departure[agent.handle][0]
        agent.latest_arrival = agent_latest_arrival[agent.handle][-1]

    num_steps_run = 1200
    _run_with_sthortest_path(env, rendering, num_steps_run, stop_at_first_intermediate=False)

    assert conditional_malfunction_effects_generator._num_malfunctions == 3
    for agent in env.agents:
        assert agent.state == TrainState.DONE
