from fractions import Fraction

import numpy as np
import pytest

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.agent_utils import _sanitize_entry_point
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_env_state_machine_wrapper import RailEnvStateMachineWrapper
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.core.transition_map import DEFAULT_EDGE_LENGTH
from flatland.envs.step_utils.speed_counter import SpeedCounter
from flatland.envs.step_utils.states import TrainState
from flatland.utils.simple_rail import make_simple_rail
from tests.test_utils import ReplayConfig, Replay, run_replay_config, set_penalties_for_replay


def test_initial_status():
    """Test that agent lifecycle works correctly ready-to-depart -> active -> done."""
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  remove_agents_at_target=False)
    env = RailEnvStateMachineWrapper(env)
    env.reset()

    env._max_episode_steps = 1000

    # Perform DO_NOTHING actions until all trains get to READY_TO_DEPART
    for _ in range(max([agent.earliest_departure for agent in env.agents])):
        env.step({})  # DO_NOTHING for all agents

    set_penalties_for_replay(env)
    test_config = ReplayConfig(
        replay=[
            Replay(  # 0
                position=None,  # not entered grid yet
                direction=None,
                state=TrainState.READY_TO_DEPART,

                action=RailEnvActions.DO_NOTHING,
            ),
            Replay(  # 1
                position=None,  # not entered grid yet before step
                direction=None,
                state=TrainState.READY_TO_DEPART,

                action=RailEnvActions.MOVE_LEFT,
            ),
            Replay(  # 2
                position=(3, 9),
                direction=Grid4TransitionsEnum.EAST,
                # design: map entry sets distance to 0 exactly when position enters the map
                distance=0,
                state=TrainState.MOVING,

                action=RailEnvActions.MOVE_LEFT,
            ),
            Replay(  # 3
                position=(3, 9),
                direction=Grid4TransitionsEnum.EAST,
                distance=0.5,
                state=TrainState.MOVING,

                action=RailEnvActions.MOVE_FORWARD,
            ),
            Replay(  # 4
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                distance=0.0,
                state=TrainState.MOVING,

                action=RailEnvActions.MOVE_FORWARD,
            ),
            Replay(
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                state=TrainState.MOVING,
                action=None,
            ),
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD,
                state=TrainState.MOVING
            ),
            Replay(
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                state=TrainState.MOVING
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_RIGHT,
                state=TrainState.MOVING
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                state=TrainState.MOVING
            ),
            # Replay(
            #     position=(3, 5),
            #     direction=Grid4TransitionsEnum.WEST,
            #     action=None,
            #     reward=env.rewards.global_reward,  # already done
            #     status=RailAgentStatus.DONE
            # ),
            # Replay(
            #     position=(3, 5),
            #     direction=Grid4TransitionsEnum.WEST,
            #     action=None,
            #     reward=env.rewards.global_reward,  # already done
            #     status=RailAgentStatus.DONE
            # )

        ],
        initial_position=(3, 9),  # east dead-end
        initial_direction=Grid4TransitionsEnum.EAST,
        target=(3, 5),
        speed=0.5
    )

    run_replay_config(env, [test_config], activate_agents=False, skip_reward_check=True, skip_action_required_check=True,
                      set_ready_to_depart=True)

    assert env.agents[0].state == TrainState.DONE


def test_status_done_remove():
    """Test that agent lifecycle works correctly ready-to-depart -> active -> done."""
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  remove_agents_at_target=True)
    env = RailEnvStateMachineWrapper(env)
    env.reset()

    # Perform DO_NOTHING actions until all trains get to READY_TO_DEPART
    for _ in range(max([agent.earliest_departure for agent in env.agents])):
        env.step({})  # DO_NOTHING for all agents

    env._max_episode_steps = 1000

    set_penalties_for_replay(env)
    test_config = ReplayConfig(
        replay=[
            Replay(  # 0
                position=None,  # not entered grid yet
                direction=None,
                state=TrainState.READY_TO_DEPART,
                action=RailEnvActions.DO_NOTHING,

            ),
            Replay(  # 1
                position=None,  # not entered grid yet before step
                direction=None,
                state=TrainState.READY_TO_DEPART,
                action=RailEnvActions.MOVE_LEFT,
            ),
            Replay(  # 2
                position=(3, 9),
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.MOVING,
                action=RailEnvActions.MOVE_FORWARD,
            ),
            Replay(  # 3
                position=(3, 9),
                direction=Grid4TransitionsEnum.EAST,
                state=TrainState.MOVING,
                action=None,
            ),
            Replay(  # 4
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                state=TrainState.MOVING,
                action=RailEnvActions.MOVE_FORWARD,
            ),
            Replay(  # 5
                position=(3, 8),
                direction=Grid4TransitionsEnum.WEST,
                state=TrainState.MOVING,
                action=None,

            ),
            Replay(  # 6
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_RIGHT,
                state=TrainState.MOVING
            ),
            Replay(  # 7
                position=(3, 7),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                state=TrainState.MOVING
            ),
            Replay(  # 8
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=RailEnvActions.MOVE_FORWARD,
                state=TrainState.MOVING
            ),
            Replay(
                position=(3, 6),
                direction=Grid4TransitionsEnum.WEST,
                action=None,
                state=TrainState.MOVING
            ),
            # Replay(
            #     position=None,
            #     direction=Grid4TransitionsEnum.WEST,
            #     action=None,
            #     reward=env.rewards.global_reward,  # already done
            #     status=RailAgentStatus.DONE_REMOVED
            # ),
            # Replay(
            #     position=None,
            #     direction=Grid4TransitionsEnum.WEST,
            #     action=None,
            #     reward=env.rewards.global_reward,  # already done
            #     status=RailAgentStatus.DONE_REMOVED
            # )

        ],
        initial_position=(3, 9),  # east dead-end
        initial_direction=Grid4TransitionsEnum.EAST,
        target=(3, 5),
        speed=0.5
    )

    run_replay_config(env, [test_config], activate_agents=False, skip_reward_check=True, skip_action_required_check=True,
                      set_ready_to_depart=True)
    assert env.agents[0].state == TrainState.DONE


def _make_straight_rail(n_cells: int):
    """
    The smallest possible topology offering a target cell to cross into: a straight,
    switch-free corridor of `n_cells` cells, dead end - straight* - dead end (a plain 2-cell corridor
    for n_cells=2, no straight tiles needed in between at all).

    Used by the test_target_reached_not_removed_distance_is_leftover_* tests below, which exercise the
    "target reached, remove_agents_at_target=False" case via an actual RailEnv: the agent crosses into its
    target edge like any other crossing, so its distance is the leftover of that step
    (SpeedCounter.distance_after_crossing), and it stays there once the agent is DONE.
    """
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    dead_end = cells[7]
    straight = transitions.rotate_transition(cells[1], 90)
    opens_east = transitions.rotate_transition(dead_end, 270)  # connects only to its east neighbor
    opens_west = transitions.rotate_transition(dead_end, 90)  # connects only to its west neighbor
    row = [opens_east] + [straight] * (n_cells - 2) + [opens_west]
    rail_map = np.array([row], dtype=np.uint16)
    rail = RailGridTransitionMap(width=n_cells, height=1, transitions=transitions)
    rail.grid = rail_map
    agents_hints = {
        'city_positions': [(0, 0), (0, n_cells - 1)],
        'train_stations': [[((0, 0), 0)], [((0, n_cells - 1), 0)]],
        'city_orientations': [0, 0],
    }
    return rail, {'agents_hints': agents_hints}


def _place_agent_on_map(env, handle, position, direction, target, state, max_speed, speed, first_action):
    """ Bootstrap an agent directly onto the map at `position`/`direction`, in `state`, with `speed`
    (out of `max_speed`) - bypassing the READY_TO_DEPART/departure machinery so the test controls
    exactly when each agent starts moving (needed for the STOPPED-blocker below, which must not move
    until explicitly told to). """
    agent = env.agents[handle]
    agent.initial_entry_point = (position, direction)
    agent.current_entry_point = (position, direction)
    agent.targets = {(target, d) for d in Grid4TransitionsEnum}
    agent._set_state(state)
    agent.speed_counter = SpeedCounter(max_speed=max_speed, speed=speed)
    agent.speed_counter.set(speed=speed, distance=Fraction(0), edge_length=DEFAULT_EDGE_LENGTH)  # bootstrap distance to 0 on-map
    next_entry_point = env.rail.apply_action_independent(first_action, agent.current_entry_point)
    assert next_entry_point is not None
    agent.next_entry_point = _sanitize_entry_point(next_entry_point)


@pytest.mark.parametrize("max_speed,n_steps,expected_excess", [
    pytest.param(Fraction(1), 2, Fraction(0), id="exact_fit-default_speed"),
    pytest.param(Fraction(1, 2), 3, Fraction(0), id="exact_fit-fractional_speed"),
    pytest.param(Fraction(3, 4), 3, Fraction(1, 2), id="excess-non_dividing_max_speed"),
])
def test_target_reached_not_removed_distance_is_leftover_single_agent(max_speed, n_steps, expected_excess):
    """
    A single agent on the smallest possible corridor (2 cells: start, and target immediately next to
    it) reaching its target with `remove_agents_at_target=False` - it crosses into the target edge like any
    other crossing, so `distance` ends up at the leftover of that step, `expected_excess` (see
    `_make_straight_rail`'s docstring). The leftover is also checked against the last pre-step `configurations`
    entry.

    - exact_fit-default_speed: max_speed=1 (the common default). Step 0 is the agent's own departure
      settling step (distance deferred at 0, speed ramps 0 -> 1); step 1 already has pre_distance=0,
      pre_speed=1, so it crosses directly into the target with zero excess (sum == DEFAULT_EDGE_LENGTH
      exactly).
    - exact_fit-fractional_speed: max_speed=1/2 evenly divides DEFAULT_EDGE_LENGTH. Step 0 settles (distance
      0, speed ramps to 1/2); step 1 advances mid-cell to distance=1/2 (pre_distance=0, pre_speed=1/2,
      sum=1/2, no crossing yet); step 2 crosses into the target with pre_distance=1/2, pre_speed=1/2,
      sum=1 exactly - zero excess again, purely because 1/2 divides 1.
    - excess-non_dividing_max_speed: max_speed=3/4 does *not* evenly divide DEFAULT_EDGE_LENGTH. Step 0
      settles (speed ramps to 3/4); step 1 advances to distance=3/4 (sum=3/4, no crossing yet); step 2
      crosses into the target with pre_distance=3/4, pre_speed=3/4, sum=3/2 - the agent enters the
      target edge with a leftover of 1/2.
    """
    rail, optionals = _make_straight_rail(2)
    env = RailEnv(width=2, height=1, rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1,
                  obs_builder_object=GlobalObsForRailEnv(), remove_agents_at_target=False)
    env = RailEnvStateMachineWrapper(env)
    env.reset()
    env._max_episode_steps = 1000
    _place_agent_on_map(env, 0, (0, 0), Grid4TransitionsEnum.WEST, (0, 1), TrainState.MOVING,
                        max_speed, Fraction(0), RailEnvActions.MOVE_FORWARD)
    agent = env.agents[0]

    configurations = []
    for _ in range(n_steps):
        configurations.append((agent.current_entry_point, agent.speed_counter.distance, agent.speed_counter.speed))
        env.step({0: RailEnvActions.MOVE_FORWARD})
    print(f"configurations leading up to target: {configurations}")

    _, final_pre_distance, final_pre_speed = configurations[-1]
    assert agent.current_entry_point in agent.targets
    assert agent.speed_counter.distance == expected_excess
    assert agent.speed_counter.distance == SpeedCounter.distance_after_crossing(final_pre_distance, final_pre_speed, DEFAULT_EDGE_LENGTH)
    assert (final_pre_distance + final_pre_speed) == (DEFAULT_EDGE_LENGTH + expected_excess)


def test_target_reached_not_removed_distance_is_leftover_restart_after_denial():
    """
    A restart after a denied crossing reaches the target without excess at max_speed=1, like an agent that was
    never blocked (see exact_fit-default_speed above): a denied crossing doesn't move the agent, so its distance
    is still at its pre-step value (0) when it makes its successful attempt, not at DEFAULT_EDGE_LENGTH - it
    enters the target edge with a leftover of 0.

    Why 2 agents: the restart needs a denied crossing - either an invalid action or a
    resource_check denial. A switch-free corridor has no invalid action to give (a no-choice cell treats
    every movement action as the same single transition - verified empirically), and a switch
    would cost more cells than a second agent does. So agent 1 is a stationary blocker sitting exactly
    where agent 0 wants to go, forcing a real resource_check denial instead.

    Smallest possible topology for that: 3 cells, dead_end(A) - straight(B) - dead_end(C). Agent 0 (start
    A, target B), agent 1 (start B, target C) - agent 1 only leaves B once told to, so the test controls
    exactly when the block clears.

    Step by step (both agents max_speed=1, so speed jumps 0->1 in one step of ramping):
    - step 0: agent 0 (already MOVING, pre_speed == 1) attempts A->B for real; agent 1 (STOPPED, sitting at
      B, given DO_NOTHING) still holds B this step - denied. Agent 0 stays at distance == 0, becomes
      STOPPED. Agent 1 unchanged.
    - step 1: agent 0 given DO_NOTHING (stays STOPPED, distance still 0). Agent 1 given its first
      movement action - optimistically promoted STOPPED->MOVING (self-loop, distance stays 0, speed
      ramps to 1).
    - step 2: agent 0 still DO_NOTHING (unchanged). Agent 1's pre_speed is now 1 - its real
      crossing B->C succeeds uncontested (agent 0 isn't contesting B this step), reaching its own target
      C with pre_distance == 0, pre_speed == 1, sum == 1 exactly (zero excess) - agent 1 is now DONE, B is free.
    - step 3: agent 0 given MOVE_FORWARD - optimistically promoted STOPPED->MOVING (self-loop, distance
      stays 0, speed ramps to 1). Agent 1 stays DONE (DO_NOTHING).
    - step 4: agent 0's pre_speed is now 1 again, with pre_distance still 0 - its real
      crossing A->B succeeds (B is now completely free), reaching its target B with pre_distance == 0,
      pre_speed == 1, sum == 1 exactly (zero excess).
    """
    rail, optionals = _make_straight_rail(3)
    env = RailEnv(width=3, height=1, rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=2,
                  obs_builder_object=GlobalObsForRailEnv(), remove_agents_at_target=False)
    env = RailEnvStateMachineWrapper(env)
    env.reset()
    env._max_episode_steps = 1000
    _place_agent_on_map(env, 0, (0, 0), Grid4TransitionsEnum.WEST, (0, 1), TrainState.MOVING,
                        Fraction(1), Fraction(1), RailEnvActions.MOVE_FORWARD)
    _place_agent_on_map(env, 1, (0, 1), Grid4TransitionsEnum.EAST, (0, 2), TrainState.STOPPED,
                        Fraction(1), Fraction(0), RailEnvActions.MOVE_FORWARD)
    agent0, agent1 = env.agents[0], env.agents[1]

    actions0 = [RailEnvActions.MOVE_FORWARD, RailEnvActions.DO_NOTHING, RailEnvActions.DO_NOTHING,
                RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_FORWARD]
    actions1 = [RailEnvActions.DO_NOTHING, RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_FORWARD,
                RailEnvActions.DO_NOTHING, RailEnvActions.DO_NOTHING]
    expected_agent0 = [
        # (pre-step position, distance, speed, state)
        ((0, 0), Fraction(0), Fraction(1), TrainState.MOVING),  # about to attempt A->B for real - denied
        ((0, 0), Fraction(0), Fraction(0), TrainState.STOPPED),  # denied: stays where it was
        ((0, 0), Fraction(0), Fraction(0), TrainState.STOPPED),  # still waiting (agent 1 not gone yet)
        ((0, 0), Fraction(0), Fraction(0), TrainState.STOPPED),  # promoted only next step
        ((0, 0), Fraction(0), Fraction(1), TrainState.MOVING),  # about to attempt A->B for real again - granted
    ]
    pre_step_distance_speed = None
    for step, (position, distance, speed, state) in enumerate(expected_agent0):
        print(f"[{step}] agent 0: position={agent0.current_entry_point[0]} distance={agent0.speed_counter.distance} "
              f"speed={agent0.speed_counter.speed} state={agent0.state.name}")
        assert agent0.current_entry_point[0] == position
        assert agent0.speed_counter.distance == distance
        assert agent0.speed_counter.speed == speed
        assert agent0.state == state
        pre_step_distance_speed = (agent0.speed_counter.distance, agent0.speed_counter.speed)
        env.step({0: actions0[step], 1: actions1[step]})

    # the denied crossing left the distance at 0: the pre-step sum on the final (granted) crossing attempt is
    # pre_distance(0, not moved while denied) + pre_speed(1, ramped) == DEFAULT_EDGE_LENGTH exactly -
    # captured here from the last iteration's pre-step values.
    assert pre_step_distance_speed[0] + pre_step_distance_speed[1] == DEFAULT_EDGE_LENGTH

    assert agent0.current_entry_point in agent0.targets
    assert agent0.speed_counter.distance == 0
    assert agent0.state == TrainState.DONE
    # agent 1 completed its own (non-excess, exact-fit) crossing along the way
    assert agent1.current_entry_point in agent1.targets
    assert agent1.speed_counter.distance == 0
    assert agent1.state == TrainState.DONE


@pytest.mark.parametrize("restart_speed,expected_after_restart", [
    pytest.param(Fraction(1), [
        ((0, 1), Fraction(0), Fraction(1), TrainState.MOVING, True),  # just entered B, crosses B->C
    ], id="restart_full_speed"),
    pytest.param(Fraction(1, 2), [
        ((0, 0), Fraction(1, 2), Fraction(1), TrainState.MOVING, False),  # halfway across A
        ((0, 1), Fraction(1, 2), Fraction(1), TrainState.MOVING, True),  # just entered B with 1/2 carried, crosses B->C
    ], id="restart_half_speed"),
    pytest.param(Fraction(3, 4), [
        ((0, 0), Fraction(3, 4), Fraction(1), TrainState.MOVING, False),  # 3/4 across A
        ((0, 1), Fraction(3, 4), Fraction(1), TrainState.MOVING, True),  # just entered B with 3/4 carried, crosses B->C
    ], id="restart_non_dividing_speed"),
])
def test_distance_after_crossing_restart_after_denied_crossing_into_non_target_edge(restart_speed, expected_after_restart):
    """
    Same set-up as test_target_reached_not_removed_distance_is_leftover_restart_after_denial above, but on a
    4-cell corridor A - B - C - D, so the restart after the denied crossing crosses into B, which is not agent 0's
    target (C).

    Agent 0 (start A, target C) is denied A->B by agent 1 (STOPPED at B, target D): it doesn't move, it stays at the
    start of A (distance 0) with speed 0. Agent 1 leaves B (B->C, then C->D), then agent 0 restarts with speed
    `restart_speed`, set via acceleration_delta only after agent 1 is at full speed, so steps 0-3 are the same for
    every speed:
    - step 3: agent 0 promoted STOPPED->MOVING, speed ramps 0 -> restart_speed, distance stays 0 (the distance
      grows by the pre-step speed, which is still 0).
    - step 4 on: the new speed takes effect. At full speed, 0 + 1 reaches the exit of A, so agent 0 crosses into B
      with a leftover of 0 - nothing is dropped, as agent 0 didn't move during the denied step. At 1/2 and 3/4, agent
      0 first moves into A and crosses one step later (now at speed 1), carrying 1/2 and 3/4 into B.
    - agent 0 then crosses B->C into its target - one step later at 1/2 and 3/4 than at full speed.
    """
    rail, optionals = _make_straight_rail(4)
    env = RailEnv(width=4, height=1, rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=2,
                  obs_builder_object=GlobalObsForRailEnv(), remove_agents_at_target=False)
    env = RailEnvStateMachineWrapper(env)
    env.reset()
    env._max_episode_steps = 1000
    _place_agent_on_map(env, 0, (0, 0), Grid4TransitionsEnum.WEST, (0, 2), TrainState.MOVING,
                        Fraction(1), Fraction(1), RailEnvActions.MOVE_FORWARD)
    _place_agent_on_map(env, 1, (0, 1), Grid4TransitionsEnum.EAST, (0, 3), TrainState.STOPPED,
                        Fraction(1), Fraction(0), RailEnvActions.MOVE_FORWARD)
    agent0, agent1 = env.agents[0], env.agents[1]

    actions0 = [RailEnvActions.MOVE_FORWARD, RailEnvActions.DO_NOTHING, RailEnvActions.DO_NOTHING] + [RailEnvActions.MOVE_FORWARD] * 4
    actions1 = [RailEnvActions.DO_NOTHING] + [RailEnvActions.MOVE_FORWARD] * 3 + [RailEnvActions.DO_NOTHING] * 3
    expected_agent0 = [
        # (pre-step position, distance, speed, state, is_cell_entry)
        ((0, 0), Fraction(0), Fraction(1), TrainState.MOVING, True),  # about to attempt A->B for real - denied
        ((0, 0), Fraction(0), Fraction(0), TrainState.STOPPED, False),  # denied: stays where it was
        ((0, 0), Fraction(0), Fraction(0), TrainState.STOPPED, False),  # still waiting (agent 1 not gone yet)
        ((0, 0), Fraction(0), Fraction(0), TrainState.STOPPED, False),  # promoted only next step
        ((0, 0), Fraction(0), restart_speed, TrainState.MOVING, False),  # restarting with restart_speed
    ] + expected_after_restart
    for step, (position, distance, speed, state, is_cell_entry) in enumerate(expected_agent0):
        if step == 3:
            env.acceleration_delta = restart_speed
        print(f"[{step}] agent 0: position={agent0.current_entry_point[0]} distance={agent0.speed_counter.distance} "
              f"speed={agent0.speed_counter.speed} state={agent0.state.name} is_cell_entry={agent0.speed_counter.is_cell_entry}")
        assert agent0.current_entry_point[0] == position
        assert agent0.speed_counter.distance == distance
        assert agent0.speed_counter.speed == speed
        assert agent0.state == state
        assert agent0.speed_counter.is_cell_entry == is_cell_entry
        env.step({0: actions0[step], 1: actions1[step]})

    # reaches the target C right after crossing into B
    assert agent0.current_entry_point[0] == (0, 2)
    assert agent0.state == TrainState.DONE
    assert agent1.current_entry_point[0] == (0, 3)
    assert agent1.state == TrainState.DONE
