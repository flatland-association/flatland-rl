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
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.step_utils.speed_counter import SpeedCounter, SEGMENT_LENGTH
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
    The smallest possible topology offering a genuine target cell to cross into: a straight,
    switch-free corridor of `n_cells` cells, dead end - straight* - dead end (a plain 2-cell corridor
    for n_cells=2, no straight tiles needed in between at all).

    Used by the test_distance_without_crossing_reaches_segment_length_on_target_* tests below, which
    exercise SpeedCounter.distance_without_crossing's "target reached, remove_agents_at_target=False"
    case via an actual RailEnv rather than by calling the formula directly - this matters because the
    formula alone doesn't explain *why* it applies here: by the time RailEnv.step()'s (10b) runs,
    agent.state is already DONE (see (10a)'s update_if_reached(), called before (10b)), so
    _candidate_distance's own "done" branch is the one that applies (not its ordinary "genuine
    crossing" branch, which would wrap distance via distance_after_crossing) - it calls
    distance_without_crossing directly.
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
    agent.speed_counter.set(speed=speed, distance=Fraction(0))  # bootstrap distance to 0 on-map
    next_entry_point = env.rail.apply_action_independent(first_action, agent.current_entry_point)
    assert next_entry_point is not None
    agent.next_entry_point = _sanitize_entry_point(next_entry_point)


@pytest.mark.parametrize("max_speed,n_steps,expected_excess", [
    pytest.param(Fraction(1), 2, Fraction(0), id="exact_fit-default_speed"),
    pytest.param(Fraction(1, 2), 3, Fraction(0), id="exact_fit-fractional_speed"),
    pytest.param(Fraction(3, 4), 3, Fraction(1, 2), id="excess-non_dividing_max_speed"),
])
def test_distance_without_crossing_reaches_segment_length_on_target_single_agent(max_speed, n_steps, expected_excess):
    """
    A single agent on the smallest possible corridor (2 cells: start, and target immediately next to
    it) reaching its target with `remove_agents_at_target=False` - `distance` always ends up at exactly
    `SEGMENT_LENGTH`, discarding `expected_excess` momentum in the process (see
    `_make_straight_rail`/`SpeedCounter.distance_without_crossing`'s docstrings for why).

    Note `distance == SEGMENT_LENGTH` alone is true of *any* target arrival (exact-fit or excess) and
    stays true forever afterward (DONE's (10b) fallback keeps re-deriving it from a now-frozen speed=0)
    - it does not distinguish these cases from each other. What actually varies between them, and is
    what these parametrizations are for, is `expected_excess` - checked below from the last pre-step
    `configurations` entry, i.e. before the cap is applied.

    - exact_fit-default_speed: max_speed=1 (the common default). Step 0 is the agent's own departure
      settling step (distance deferred at 0, speed ramps 0 -> 1); step 1 already has pre_distance=0,
      pre_speed=1, so it crosses directly into the target with zero excess (sum == SEGMENT_LENGTH
      exactly).
    - exact_fit-fractional_speed: max_speed=1/2 evenly divides SEGMENT_LENGTH. Step 0 settles (distance
      0, speed ramps to 1/2); step 1 advances mid-cell to distance=1/2 (pre_distance=0, pre_speed=1/2,
      sum=1/2, no crossing yet); step 2 crosses into the target with pre_distance=1/2, pre_speed=1/2,
      sum=1 exactly - zero excess again, purely because 1/2 divides 1.
    - excess-non_dividing_max_speed: max_speed=3/4 does *not* evenly divide SEGMENT_LENGTH. Step 0
      settles (speed ramps to 3/4); step 1 advances to distance=3/4 (sum=3/4, no crossing yet); step 2
      crosses into the target with pre_distance=3/4, pre_speed=3/4, sum=3/2 - an excess of 1/2 gets
      silently discarded by the cap, with no prior blocking/banking involved at all.
    """
    rail, optionals = _make_straight_rail(2)
    env = RailEnv(width=2, height=1, rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1,
                  obs_builder_object=GlobalObsForRailEnv(), remove_agents_at_target=False)
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
    assert agent.speed_counter.distance == SEGMENT_LENGTH
    assert agent.speed_counter.distance == SpeedCounter.distance_without_crossing(final_pre_distance, final_pre_speed)
    assert (final_pre_distance + final_pre_speed) == (SEGMENT_LENGTH + expected_excess)


def test_distance_without_crossing_reaches_segment_length_on_target_banked_restart():
    """
    The one case that *does* need prior blocking: max_speed=1 would otherwise never produce excess (see
    exact_fit-default_speed above) - what makes this different is that the agent's distance is already
    banked at SEGMENT_LENGTH (from an earlier, real resource_check denial) by the time it makes its
    successful attempt, rather than arriving mid-cell.

    Why 2 agents: with max_speed=1, banking needs a denied crossing - either an invalid action or a
    resource_check denial. A switch-free corridor has no invalid action to give (a no-choice cell treats
    every movement action as the same single transition - verified empirically), and a genuine switch
    would cost more cells than a second agent does. So agent 1 is a stationary blocker sitting exactly
    where agent 0 wants to go, forcing a real resource_check denial instead.

    Smallest possible topology for that: 3 cells, dead_end(A) - straight(B) - dead_end(C). Agent 0 (start
    A, target B), agent 1 (start B, target C) - agent 1 only leaves B once told to, so the test controls
    exactly when the block clears.

    Step by step (both agents max_speed=1, so speed jumps 0->1 in one step of ramping):
    - step 0: agent 0 (already MOVING, pre_speed=1) attempts A->B for real; agent 1 (STOPPED, sitting at
      B, given DO_NOTHING) still holds B this step - denied. Agent 0 banks at distance=1, becomes
      STOPPED. Agent 1 unchanged.
    - step 1: agent 0 given DO_NOTHING (stays STOPPED, distance still 1). Agent 1 given its first
      movement action - optimistically promoted STOPPED->MOVING (self-loop, distance stays 0, speed
      ramps to 1).
    - step 2: agent 0 still DO_NOTHING (unchanged). Agent 1's pre_speed is now genuinely 1 - its real
      crossing B->C succeeds uncontested (agent 0 isn't contesting B this step), reaching its own target
      C with pre_distance=0, pre_speed=1, sum=1 exactly (zero excess) - agent 1 is now DONE, B is free.
    - step 3: agent 0 given MOVE_FORWARD - optimistically promoted STOPPED->MOVING (self-loop, distance
      stays at its banked value of 1, speed ramps to 1). Agent 1 stays DONE (DO_NOTHING).
    - step 4: agent 0's pre_speed is now genuinely 1 again, with pre_distance still 1 (banked) - its real
      crossing A->B succeeds (B is now completely free), reaching its target B with pre_distance=1,
      pre_speed=1, sum=2 - a full SEGMENT_LENGTH of momentum discarded by the cap.
    """
    rail, optionals = _make_straight_rail(3)
    env = RailEnv(width=3, height=1, rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=2,
                  obs_builder_object=GlobalObsForRailEnv(), remove_agents_at_target=False)
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
        ((0, 0), Fraction(1), Fraction(0), TrainState.STOPPED),  # banked at the boundary
        ((0, 0), Fraction(1), Fraction(0), TrainState.STOPPED),  # still waiting (agent 1 not gone yet)
        ((0, 0), Fraction(1), Fraction(0), TrainState.STOPPED),  # promoted only next step
        ((0, 0), Fraction(1), Fraction(1), TrainState.MOVING),  # about to attempt A->B for real again - granted
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

    # design: distance == SEGMENT_LENGTH alone (checked below) is true of *any* target arrival, exact-fit
    # or excess, and stays true forever afterward (DONE's (10b) fallback keeps re-deriving it from a
    # now-frozen speed=0) - it does not by itself distinguish this banked-restart case from an ordinary
    # exact-fit one. What is actually specific to this case is the pre-step sum on the final (granted)
    # crossing attempt: pre_distance(1, banked) + pre_speed(1, ramped) == 2, a full SEGMENT_LENGTH of excess
    # momentum silently discarded by the cap - captured here from the last iteration's pre-step values.
    assert pre_step_distance_speed[0] + pre_step_distance_speed[1] == SEGMENT_LENGTH + Fraction(1)

    assert agent0.current_entry_point in agent0.targets
    assert agent0.speed_counter.distance == SEGMENT_LENGTH
    assert agent0.state == TrainState.DONE
    # agent 1 completed its own (non-excess, exact-fit) crossing along the way
    assert agent1.current_entry_point in agent1.targets
    assert agent1.speed_counter.distance == SEGMENT_LENGTH
    assert agent1.state == TrainState.DONE
