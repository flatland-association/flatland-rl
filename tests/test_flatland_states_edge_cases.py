from fractions import Fraction

import numpy as np
import pytest

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rewards import DefaultRewards
from flatland.envs.step_utils.states import TrainState
from flatland.utils.simple_rail import make_simple_rail

pytestmark = pytest.mark.cython_ext


def test_return_to_ready_to_depart():
    """
    When going from ready to depart to malfunction off map, if do nothing is provided, should return to ready to depart
    """
    stochastic_data = MalfunctionParameters(malfunction_rate=0,  # Rate of malfunction occurence
                                            min_duration=0,  # Minimal duration of malfunction
                                            max_duration=0  # Max duration of malfunction
                                            )

    rail, _, optionals = make_simple_rail()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=10),
                  number_of_agents=1,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  )

    env.reset(False, False, random_seed=10)
    env._max_episode_steps = 100

    for _ in range(3):
        env.step({0: RailEnvActions.DO_NOTHING})

    env.agents[0].malfunction_handler._set_malfunction_down_counter(2)
    env.step({0: RailEnvActions.DO_NOTHING})

    assert env.agents[0].state == TrainState.MALFUNCTION_OFF_MAP

    for _ in range(2):
        env.step({0: RailEnvActions.DO_NOTHING})

    assert env.agents[0].state == TrainState.READY_TO_DEPART


def test_ready_to_depart_to_ready_to_depart_with_stop_action():
    """
    When going from ready to depart to malfunction off map, if stopped is provided, should stay ready to depart
    """
    stochastic_data = MalfunctionParameters(malfunction_rate=0,  # Rate of malfunction occurence
                                            min_duration=0,  # Minimal duration of malfunction
                                            max_duration=0  # Max duration of malfunction
                                            )

    rail, _, optionals = make_simple_rail()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=10),
                  number_of_agents=1,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  )

    env.reset(False, False, random_seed=10)
    env._max_episode_steps = 100

    for _ in range(3):
        env.step({0: RailEnvActions.STOP_MOVING})

    assert env.agents[0].state == TrainState.READY_TO_DEPART

    env.agents[0].malfunction_handler._set_malfunction_down_counter(2)
    env.step({0: RailEnvActions.STOP_MOVING})

    assert env.agents[0].state == TrainState.MALFUNCTION_OFF_MAP

    for _ in range(2):
        env.step({0: RailEnvActions.STOP_MOVING})

    # design: disallow entering the map stopped
    assert env.agents[0].state == TrainState.READY_TO_DEPART


def test_malfunction_no_phase_through():
    """
    A moving train shouldn't phase through a malfunctioning train
    """
    stochastic_data = MalfunctionParameters(malfunction_rate=0,  # Rate of malfunction occurence
                                            min_duration=0,  # Minimal duration of malfunction
                                            max_duration=0  # Max duration of malfunction
                                            )

    rail, _, optionals = make_simple_rail()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=10),
                  number_of_agents=2,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  )

    env.reset(False, False, random_seed=10)

    for _ in range(5):
        env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    env.agents[1].malfunction_handler._set_malfunction_down_counter(10)

    # design (issue #280): both agents depart one step earlier than before (default earliest_departure=0
    # for this seed), so agent 0 is one cell further along by the time it catches up to agent 1 - one more
    # step is needed here to land on the same STOPPED-behind-a-malfunctioning-train outcome.
    for _ in range(4):
        env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.DO_NOTHING})

    assert env.agents[0].state == TrainState.STOPPED
    assert env.agents[0].current_entry_point[0] == (3, 6)


def test_malfunction_off_map_not_on_map_with_stop_action_after_malfunction():
    """
    MALFUNCTION_OFF_MAP getting into map must respect without motion check.
    """
    stochastic_data = MalfunctionParameters(malfunction_rate=0,  # Rate of malfunction occurence
                                            min_duration=0,  # Minimal duration of malfunction
                                            max_duration=0  # Max duration of malfunction
                                            )

    rail, _, optionals = make_simple_rail()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=10),
                  number_of_agents=2,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  )

    env.reset(False, False, random_seed=10)

    env.agents[0].initial_entry_point = ((6, 6), Grid4TransitionsEnum.SOUTH)
    env.agents[0].targets = {((0, 3), d) for d in Grid4TransitionsEnum}
    # design (issue #280): earliest_departure=1, not 0 - an earliest_departure=0 agent now dispatches
    # directly on the very first movement action (see rail_env.py's step()), which isn't the point of
    # this test; =1 keeps the original two-real-steps-to-depart timing this test relies on.
    env.agents[0].earliest_departure = 1

    env.agents[1].initial_entry_point = ((6, 6), Grid4TransitionsEnum.SOUTH)
    env.agents[1].targets = {((0, 3), d) for d in Grid4TransitionsEnum}
    env.agents[1].earliest_departure = 1
    # design: malfunction counter decremented at start of step(), before new malfunctions are generated -
    # injected duration bumped by 1 so the state sequence below is unaffected (only the malfunction countdown
    # values shift by 1 while the malfunction is active)
    env.agents[1].malfunction_handler._set_malfunction_down_counter(3)

    # step 1
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})
    assert env.agents[0].current_entry_point is None
    assert env.agents[0].state == TrainState.READY_TO_DEPART

    assert env.agents[1].current_entry_point is None
    assert env.agents[1].state == TrainState.MALFUNCTION_OFF_MAP

    assert env.agents[1].malfunction_handler.malfunction_down_counter == 2

    # step 2
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].current_entry_point[0] == (6, 6)
    assert env.agents[0].state == TrainState.MOVING

    assert env.agents[1].current_entry_point is None
    assert env.agents[1].state == TrainState.MALFUNCTION_OFF_MAP
    assert env.agents[1].malfunction_handler.malfunction_down_counter == 1

    # step 3
    env.step({0: RailEnvActions.STOP_MOVING, 1: RailEnvActions.STOP_MOVING})

    # agent 0's pre-step momentum (speed 1, distance 0 at fresh departure) already reaches the cell
    # boundary this step, so STOP_MOVING completes the in-flight crossing before halting - same as
    # DO_NOTHING/MOVE_FORWARD would - landing one cell past (6, 6) rather than blocking in place.
    assert env.agents[0].current_entry_point[0] == (5, 6)
    assert env.agents[0].state == TrainState.STOPPED

    # design: disallow entering the map stopped
    assert env.agents[1].current_entry_point is None
    assert env.agents[1].state == TrainState.READY_TO_DEPART


def test_malfunction_motion_check_order_when_earliest_departure_is_not_reached():
    """
    Avoid adding agent to motion check as it can hinder other agents having earliest_departure_reached to start.

    Two agents share initial entry point (6, 6). Agent 0 has earliest_departure=55 (far off) and starts
    malfunctioning off map; agent 1 has earliest_departure=0. Design (issue #280): agent 0 is never
    registered in the motion check while off map and ineligible to depart, so it can't block agent 1 from
    dispatching on agent 1's very first step, regardless of which agent has the lower index.
    """
    stochastic_data = MalfunctionParameters(malfunction_rate=0,  # Rate of malfunction occurence
                                            min_duration=0,  # Minimal duration of malfunction
                                            max_duration=0  # Max duration of malfunction
                                            )

    rail, _, optionals = make_simple_rail()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=10),
                  number_of_agents=2,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  )

    env.reset(False, False, random_seed=10)

    env.agents[0].initial_entry_point = ((6, 6), Grid4TransitionsEnum.SOUTH)
    env.agents[0].targets = {((0, 3), d) for d in Grid4TransitionsEnum}
    env.agents[0].earliest_departure = 55
    # design: malfunction counter decremented at start of step(), before new malfunctions are generated -
    # injected duration bumped by 1 so the state sequence below is unaffected (only the malfunction countdown
    # values shift by 1 while the malfunction is active)
    env.agents[0].malfunction_handler._set_malfunction_down_counter(2)

    env.agents[1].initial_entry_point = ((6, 6), Grid4TransitionsEnum.SOUTH)
    env.agents[1].targets = {((0, 3), d) for d in Grid4TransitionsEnum}
    # design (issue #280): earliest_departure=0 - agent 1 now dispatches straight into MOVING on its own
    # very first step, collapsing what used to be a 3-step scenario into a single step.
    env.agents[1].earliest_departure = 0

    # step 1
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].current_entry_point is None
    assert env.agents[0].state == TrainState.MALFUNCTION_OFF_MAP
    assert env.agents[0].malfunction_handler.malfunction_down_counter == 1
    assert env.agents[1].current_entry_point[0] == (6, 6)
    assert env.agents[1].state == TrainState.MOVING


def test_malfunction_motion_check_order_when_earliest_departure_reached_but_not_moving_action():
    """
    Avoid adding agent to motion check as it can hinder other agents having earliest_departure_reached to start.

    Two agents share initial entry point (6, 6). Agent 0 has earliest_departure=3 and starts malfunctioning
    off map; agent 1 has earliest_departure=0. Design (issue #280): agent 1 dispatches into MOVING on the
    step it finally sends MOVE_FORWARD, unblocked by agent 0 sitting at the same entry point in
    READY_TO_DEPART/MALFUNCTION_OFF_MAP with a movement action of its own pending.
    """
    stochastic_data = MalfunctionParameters(malfunction_rate=0,  # Rate of malfunction occurence
                                            min_duration=0,  # Minimal duration of malfunction
                                            max_duration=0  # Max duration of malfunction
                                            )

    rail, _, optionals = make_simple_rail()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=10),
                  number_of_agents=2,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  )

    env.reset(False, False, random_seed=10)

    env.agents[0].initial_entry_point = ((6, 6), Grid4TransitionsEnum.SOUTH)
    env.agents[0].targets = {((0, 3), d) for d in Grid4TransitionsEnum}
    env.agents[0].earliest_departure = 3
    # design: malfunction counter decremented at start of step(), before new malfunctions are generated -
    # injected duration bumped by 1 so the state sequence below is unaffected (only the malfunction countdown
    # values shift by 1 while the malfunction is active)
    env.agents[0].malfunction_handler._set_malfunction_down_counter(2)

    env.agents[1].initial_entry_point = ((6, 6), Grid4TransitionsEnum.SOUTH)
    env.agents[1].targets = {((0, 3), d) for d in Grid4TransitionsEnum}
    # design (issue #280): earliest_departure=0 - agent 1 is READY_TO_DEPART already on its own first step
    # (DO_NOTHING here, so it doesn't yet dispatch), collapsing what used to be a 3-step scenario into 2.
    env.agents[1].earliest_departure = 0

    # step 1
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.DO_NOTHING})

    assert env.agents[0].current_entry_point is None
    assert env.agents[0].state == TrainState.MALFUNCTION_OFF_MAP
    assert env.agents[0].malfunction_handler.malfunction_down_counter == 1

    assert env.agents[1].current_entry_point is None
    assert env.agents[1].state == TrainState.READY_TO_DEPART

    # step 2
    env.step({0: RailEnvActions.DO_NOTHING, 1: RailEnvActions.MOVE_FORWARD})
    assert env.agents[0].current_entry_point is None
    assert env.agents[0].state == TrainState.READY_TO_DEPART
    assert env.agents[0].malfunction_handler.malfunction_down_counter == 0

    assert env.agents[1].current_entry_point[0] == (6, 6)
    assert env.agents[1].state == TrainState.MOVING


@pytest.mark.parametrize("malfunctioning", ["none", "agent_0", "agent_1", "both"])
def test_same_cell_same_earliest_departure_dispatch_conflict(malfunctioning):
    """
    Two agents share both initial entry point (6, 6) and earliest_departure=2. Design: when both are
    simultaneously eligible to depart into the same cell, the motion check resolves the conflict by
    agent index - agent 0 wins regardless of the tie being genuinely symmetric (unlike the
    motion-check-order tests above, where a lower-index agent winning is the bug being guarded against,
    here both agents are equally eligible, so index order is the actual, intended tie-break). The loser
    incurs no collision penalty: the collision penalty (see BaseDefaultRewards.step_reward) only fires
    on a MOVING -> STOPPED demotion, and a denied READY_TO_DEPART agent never was MOVING to begin with.

    Parametrized over which agent(s) malfunction right at the departure step: a malfunctioning agent is
    excluded from the motion check entirely (see rail_env.py's (3b.2), which takes priority over (3b.3)'s
    map-entry branch), so it can never block the other agent - whichever agent does NOT malfunction
    dispatches into (6, 6) unblocked, exactly as agent 0 would in the unparametrized "none" case.

    - Setup: malfunction_down_counter=2 injected (for the malfunctioning agent(s)) right before the
      departure step - observed in_malfunction=True for that one step.
    - Step 1 (both MOVE_FORWARD): both agents reach READY_TO_DEPART, still off map.
    - Step 2 (both MOVE_FORWARD, the departure step): outcome depends on malfunctioning:
      - "none": agent 0 dispatches into MOVING at (6, 6); agent 1 stays READY_TO_DEPART, denied.
      - "agent_0": agent 0 goes to MALFUNCTION_OFF_MAP instead of contesting the cell; agent 1
        dispatches into MOVING at (6, 6) unblocked.
      - "agent_1": symmetric - agent 1 goes to MALFUNCTION_OFF_MAP; agent 0 dispatches unblocked.
      - "both": neither contests the cell; both go to MALFUNCTION_OFF_MAP.
      In every variant, neither agent's reward carries a collision penalty this step.
    """
    stochastic_data = MalfunctionParameters(malfunction_rate=0,  # Rate of malfunction occurence
                                            min_duration=0,  # Minimal duration of malfunction
                                            max_duration=0  # Max duration of malfunction
                                            )

    rail, _, optionals = make_simple_rail()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=10),
                  number_of_agents=2,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  rewards=DefaultRewards(collision_factor=2.0),
                  )

    env.reset(False, False, random_seed=10)

    for a in range(2):
        env.agents[a].initial_entry_point = ((6, 6), Grid4TransitionsEnum.SOUTH)
        env.agents[a].targets = {((0, 3), d) for d in Grid4TransitionsEnum}
        env.agents[a].earliest_departure = 2

    # step 1
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})
    assert env.agents[0].state == TrainState.READY_TO_DEPART
    assert env.agents[1].state == TrainState.READY_TO_DEPART

    if malfunctioning in ("agent_0", "both"):
        env.agents[0].malfunction_handler._set_malfunction_down_counter(2)
    if malfunctioning in ("agent_1", "both"):
        env.agents[1].malfunction_handler._set_malfunction_down_counter(2)

    # step 2 (the departure step)
    _, rewards, _, _ = env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    # no collision penalty for either agent, regardless of malfunctioning: the penalty only fires on a
    # MOVING -> STOPPED demotion, never on a READY_TO_DEPART agent denied map entry.
    assert rewards[0] == 0
    assert rewards[1] == 0

    if malfunctioning == "none":
        assert env.agents[0].state == TrainState.MOVING
        assert env.agents[0].current_entry_point[0] == (6, 6)
        assert env.agents[1].state == TrainState.READY_TO_DEPART
        assert env.agents[1].current_entry_point is None
    elif malfunctioning == "agent_0":
        assert env.agents[0].state == TrainState.MALFUNCTION_OFF_MAP
        assert env.agents[0].current_entry_point is None
        assert env.agents[1].state == TrainState.MOVING
        assert env.agents[1].current_entry_point[0] == (6, 6)
    elif malfunctioning == "agent_1":
        assert env.agents[0].state == TrainState.MOVING
        assert env.agents[0].current_entry_point[0] == (6, 6)
        assert env.agents[1].state == TrainState.MALFUNCTION_OFF_MAP
        assert env.agents[1].current_entry_point is None
    else:  # both
        assert env.agents[0].state == TrainState.MALFUNCTION_OFF_MAP
        assert env.agents[0].current_entry_point is None
        assert env.agents[1].state == TrainState.MALFUNCTION_OFF_MAP
        assert env.agents[1].current_entry_point is None


@pytest.mark.parametrize("malfunctioning", ["none", "agent_0", "agent_1", "both"])
def test_same_cell_same_earliest_departure_dispatch_conflict_malfunction_ends_on_departure(malfunctioning):
    """
    Same setup as test_same_cell_same_earliest_departure_dispatch_conflict (two agents sharing initial
    entry point (6, 6) and earliest_departure=2), but here the parametrized agent(s) are already
    malfunctioning before ever departing, with the malfunction timed to end exactly on the departure
    step (earliest_departure=2) rather than starting on it. Design (issue #280): a malfunction ending
    with earliest_departure already reached goes straight from MALFUNCTION_OFF_MAP into MOVING on a
    movement action - or, if the cell is contested, into READY_TO_DEPART instead of back into
    MALFUNCTION_OFF_MAP, since the malfunction has genuinely already ended by then (see
    _handle_malfunction_off_map above). Same asserted properties as the sibling test: agent 0 (lower
    index) always wins the departure conflict regardless of which agent(s) were malfunctioning, and
    neither agent's reward carries a collision penalty.

    - Setup: malfunction_down_counter=2 injected (for the parametrized agent(s)) before step 1 -
      in_malfunction=True for step 1 only, already False again by step 2, the departure step.
    - Step 1 (both MOVE_FORWARD): a malfunctioning agent goes WAITING -> MALFUNCTION_OFF_MAP
      (malfunction_down_counter == 1 afterwards); a non-malfunctioning agent goes WAITING ->
      READY_TO_DEPART as usual.
    - Step 2 (both MOVE_FORWARD, the departure step - malfunction already ended for any parametrized
      agent): agent 0 dispatches into MOVING at (6, 6); agent 1 is denied and lands in READY_TO_DEPART -
      never back in MALFUNCTION_OFF_MAP, regardless of whether agent 1 itself was one of the
      malfunctioning agents. Neither agent's reward carries a collision penalty this step.
    """
    stochastic_data = MalfunctionParameters(malfunction_rate=0,  # Rate of malfunction occurence
                                            min_duration=0,  # Minimal duration of malfunction
                                            max_duration=0  # Max duration of malfunction
                                            )

    rail, _, optionals = make_simple_rail()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=10),
                  number_of_agents=2,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  rewards=DefaultRewards(collision_factor=2.0),
                  )

    env.reset(False, False, random_seed=10)

    for a in range(2):
        env.agents[a].initial_entry_point = ((6, 6), Grid4TransitionsEnum.SOUTH)
        env.agents[a].targets = {((0, 3), d) for d in Grid4TransitionsEnum}
        env.agents[a].earliest_departure = 2

    if malfunctioning in ("agent_0", "both"):
        env.agents[0].malfunction_handler._set_malfunction_down_counter(2)
    if malfunctioning in ("agent_1", "both"):
        env.agents[1].malfunction_handler._set_malfunction_down_counter(2)

    # step 1
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})
    if malfunctioning in ("agent_0", "both"):
        assert env.agents[0].state == TrainState.MALFUNCTION_OFF_MAP
        assert env.agents[0].malfunction_handler.malfunction_down_counter == 1
    else:
        assert env.agents[0].state == TrainState.READY_TO_DEPART
    if malfunctioning in ("agent_1", "both"):
        assert env.agents[1].state == TrainState.MALFUNCTION_OFF_MAP
        assert env.agents[1].malfunction_handler.malfunction_down_counter == 1
    else:
        assert env.agents[1].state == TrainState.READY_TO_DEPART

    # step 2 (the departure step - malfunction already ended for any parametrized agent)
    _, rewards, _, _ = env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    # no collision penalty for either agent, regardless of malfunctioning: the penalty only fires on a
    # MOVING -> STOPPED demotion, never on a denied off-map agent (READY_TO_DEPART or MALFUNCTION_OFF_MAP).
    assert rewards[0] == 0
    assert rewards[1] == 0

    # agent 0 always wins, regardless of which agent(s) were malfunctioning - agent 1 is denied and
    # lands in READY_TO_DEPART, never back in MALFUNCTION_OFF_MAP.
    assert env.agents[0].state == TrainState.MOVING
    assert env.agents[0].current_entry_point[0] == (6, 6)
    assert env.agents[1].state == TrainState.READY_TO_DEPART
    assert env.agents[1].current_entry_point is None


def test_malfunction_to_moving_instead_of_stopped():
    """
    MALFUNCTION to MOVING without going to STOPPED unnecessarily
    """
    stochastic_data = MalfunctionParameters(malfunction_rate=0,  # Rate of malfunction occurence
                                            min_duration=0,  # Minimal duration of malfunction
                                            max_duration=0  # Max duration of malfunction
                                            )

    rail, _, optionals = make_simple_rail()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=10),
                  number_of_agents=2,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  )

    env.reset(False, False, random_seed=10)

    env.agents[0].initial_entry_point = ((6, 6), Grid4TransitionsEnum.SOUTH)
    env.agents[0].targets = {((0, 3), d) for d in Grid4TransitionsEnum}
    # design (issue #280): earliest_departure=1, not 0 - an earliest_departure=0 agent now dispatches
    # directly on the very first movement action (see rail_env.py's step()), which isn't the point of
    # this test; =1 keeps the original two-real-steps-to-depart timing this test relies on.
    env.agents[0].earliest_departure = 1
    # design: speed is None while off map (agent hasn't departed yet) - only _max_speed applies here,
    # departure always (re-)accelerates from 0 regardless of any pre-set speed (see design D3).
    env.agents[0].speed_counter._max_speed = Fraction(1, 5)

    # step 1
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].current_entry_point is None
    assert env.agents[0].state == TrainState.READY_TO_DEPART

    # step 2
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].current_entry_point[0] == (6, 6)
    assert env.agents[0].state == TrainState.MOVING
    assert np.isclose(float(env.agents[0].speed_counter.speed), 0.2)
    # N.B. no movement in first time step after READY_TO_DEPART or MALFUNCTION_OFF_MAP!
    assert np.isclose(float(env.agents[0].speed_counter.distance), 0.0)

    # step 3
    # design: malfunction counter decremented at start of step(), before new malfunctions are generated -
    # injected duration bumped by 1 so the state sequence below is unaffected (only the malfunction countdown
    # values shift by 1 while the malfunction is active)
    env.agents[0].malfunction_handler._set_malfunction_down_counter(2)
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})
    assert env.agents[0].current_entry_point[0] == (6, 6)
    assert env.agents[0].state == TrainState.MALFUNCTION
    assert env.agents[0].malfunction_handler.malfunction_down_counter == 1
    assert np.isclose(float(env.agents[0].speed_counter.speed), 0.0)
    # design: distance update with pre-step speed.
    assert np.isclose(float(env.agents[0].speed_counter.distance), 0.0)

    # step 4
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})
    assert env.agents[0].current_entry_point[0] == (6, 6)
    assert env.agents[0].state == TrainState.MOVING
    assert np.isclose(float(env.agents[0].speed_counter.speed), 0.2)
    # design: distance update with pre-step speed.
    assert np.isclose(float(env.agents[0].speed_counter.distance), 0.0)


def test_stop_and_go():
    """
    Test stop and go.
    """
    stochastic_data = MalfunctionParameters(malfunction_rate=0,  # Rate of malfunction occurence
                                            min_duration=0,  # Minimal duration of malfunction
                                            max_duration=0  # Max duration of malfunction
                                            )

    rail, _, optionals = make_simple_rail()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(seed=10),
                  number_of_agents=2,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  )

    env.reset(False, False, random_seed=10)

    env.agents[0].initial_entry_point = ((6, 6), Grid4TransitionsEnum.SOUTH)
    env.agents[0].targets = {((0, 3), d) for d in Grid4TransitionsEnum}
    # design (issue #280): earliest_departure=1, not 0 - an earliest_departure=0 agent now dispatches
    # directly on the very first movement action (see rail_env.py's step()), which isn't the point of
    # this test; =1 keeps the original two-real-steps-to-depart timing this test relies on.
    env.agents[0].earliest_departure = 1
    # design: speed is None while off map (agent hasn't departed yet) - only _max_speed applies here,
    # departure always (re-)accelerates from 0 regardless of any pre-set speed (see design D3).
    env.agents[0].speed_counter._max_speed = Fraction(1, 5)

    # step 1
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].current_entry_point is None
    assert env.agents[0].state == TrainState.READY_TO_DEPART

    # step 2
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].current_entry_point[0] == (6, 6)
    assert env.agents[0].state == TrainState.MOVING
    assert np.isclose(float(env.agents[0].speed_counter.speed), 0.2)
    # N.B. no movement in first time step after READY_TO_DEPART or MALFUNCTION_OFF_MAP!
    assert np.isclose(float(env.agents[0].speed_counter.distance), 0.0)

    # step 3
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].current_entry_point[0] == (6, 6)
    assert env.agents[0].state == TrainState.MOVING
    assert np.isclose(float(env.agents[0].speed_counter.speed), 0.2)
    assert np.isclose(float(env.agents[0].speed_counter.distance), 0.2)

    # step 4
    env.step({0: RailEnvActions.STOP_MOVING, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].current_entry_point[0] == (6, 6)
    assert env.agents[0].state == TrainState.STOPPED
    assert np.isclose(float(env.agents[0].speed_counter.speed), 0.0)
    # design: distance update with pre-step speed.
    assert np.isclose(float(env.agents[0].speed_counter.distance), 0.4)

    # step 5
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})

    assert env.agents[0].current_entry_point[0] == (6, 6)
    assert env.agents[0].state == TrainState.MOVING
    assert np.isclose(float(env.agents[0].speed_counter.speed), 0.2)
    assert np.isclose(float(env.agents[0].speed_counter.distance), 0.4)
