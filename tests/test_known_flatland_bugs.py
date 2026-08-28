"""
File holding test to check if known flatland bugs are still present.
From: https://github.com/AI4REALNET/maze-flatland/blob/33048b1e2c36fc26d1543b158d823b2b1bfd2aa4/maze_flatland/test/env/test_known_flatland_bugs.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import flatland
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs import malfunction_generators
from flatland.envs.line_generators import SparseLineGen
from flatland.envs.malfunction_generators import MalfunctionParameters, NoMalfunctionGen, ParamMalfunctionGen
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.step_utils.states import TrainState


def init_test_rail_env(speed: float) -> RailEnv:
    """Initialize a small environment for testing."""
    if speed == 1:
        args = {}
    else:
        args = {'line_generator': SparseLineGen(speed_ratio_map={speed: 1})}
    rail_env = flatland.envs.rail_env.RailEnv(
        width=30,
        height=30,
        number_of_agents=2,
        malfunction_generator=NoMalfunctionGen(),
        rail_generator=sparse_rail_generator(backwards_compatibility_mode=True),
        random_seed=1234,
        **args,
    )
    _ = rail_env.reset(random_seed=1234)
    return rail_env


def test_min_distance_for_off_map_trains_speed_of_1_REVISEDESIGN() -> None:
    """
    TODO https://github.com/flatland-association/flatland-rl/issues/280 revise design: we could add +1 to "geometric" distance for off map states.

    The minimum distance for an off-map train is calculated from the initial position to the target. However, in
    order for the agent to spawn or be placed on map on the initial position one action is needed.
    As such the minimum distance can be viewed as being 1-off for off-map trains when it comes to the number of steps
    needed to reach the target and especially when reasoning on whether the train can reach its target in time.
    Although not strictly a bug, but something to be still aware of.
    """

    env = init_test_rail_env(1)
    env.step({0: RailEnvActions.DO_NOTHING, 1: RailEnvActions.DO_NOTHING})

    agent = env.agents[0]
    assert agent.state == TrainState.READY_TO_DEPART
    min_distance_off_map = env.distance_map.get()[
        agent.handle, agent.initial_entry_point[0][0], agent.initial_entry_point[0][1],
        agent.initial_entry_point[1]
    ]
    off_map_position = agent.initial_entry_point[0]

    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    min_distance_on_map = env.distance_map.get()[
        agent.handle, agent.initial_entry_point[0][0], agent.initial_entry_point[0][1],
        agent.initial_entry_point[1]
    ]
    on_map_position = agent.current_entry_point[0]
    assert np.all(on_map_position == off_map_position)
    assert min_distance_off_map == min_distance_on_map


def test_min_distance_for_off_map_trains_speed_of_half_REVISEDESIGN() -> None:
    """
    TODO https://github.com/flatland-association/flatland-rl/issues/280 revise design: we could add +1 to "geometric" distance for off map states.

    The minimum distance for an off-map train is calculated from the initial position to the target. However, in
    order for the agent to spawn or be placed on map on the initial position one action is needed.
    As such the minimum distance can be viewed as being 1-off for off-map trains when it comes to the number of steps
    needed to reach the target and especially when reasoning on whether the train can reach its target in time.
    Although not strictly a bug, but something to be still aware of.
    """
    rail_env = init_test_rail_env(0.5)

    rail_env.step({0: RailEnvActions.DO_NOTHING, 1: RailEnvActions.DO_NOTHING})
    rail_env.step({0: RailEnvActions.DO_NOTHING, 1: RailEnvActions.DO_NOTHING})

    agent = rail_env.agents[0]
    assert agent.state == TrainState.READY_TO_DEPART
    min_distance_off_map = rail_env.distance_map.get()[
        agent.handle, agent.initial_entry_point[0][0], agent.initial_entry_point[0][1],
        agent.initial_entry_point[1]
    ]
    off_map_position = agent.initial_entry_point[0]

    rail_env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    min_distance_on_map = rail_env.distance_map.get()[
        agent.handle, agent.initial_entry_point[0][0], agent.initial_entry_point[0][1],
        agent.initial_entry_point[1]
    ]
    on_map_position = agent.current_entry_point[0]
    assert np.all(on_map_position == off_map_position)
    assert min_distance_off_map == min_distance_on_map


# pylint: disable=protected-access
def test_earliest_departure_zero_bug_BYDESIGN() -> None:
    """
    TODO https://github.com/flatland-association/flatland-rl/issues/280 revise design: by design of
         https://flatland-association.github.io/flatland-book/environment/environment/agent.html#state-machine,
         an agent can go from WAITING to READY_TO_DEPART only after the first step transition. However, the design may be questioned:
         we could drop ready_to_depart by adding condition earliest_departure_reached to transition from WAITING to MOVING.

    Trains that have the earliest departure at ts 0 cannot be dispatched at ts 0 but only at ts 1. It seems like
    every train starts with train state Waiting no matter the earliest departure.
    """

    env = init_test_rail_env(1)
    assert env._elapsed_steps == 0

    agent_0, agent_1 = env.agents[0], env.agents[1]

    assert agent_1.earliest_departure == 0
    # Since agent 1s earliest departure is 0, we should be able to dispatch it, however
    assert agent_1.state == TrainState.WAITING
    # the train state is waiting

    # other train.
    assert agent_0.earliest_departure == 1
    assert agent_0.state == TrainState.WAITING

    # Now if we try to dispatch train 1 and do not dispatch train 0 ---> both end up being in ready to depart!
    env.step({0: RailEnvActions.DO_NOTHING, 1: RailEnvActions.MOVE_FORWARD})
    agent_0, agent_1 = env.agents[0], env.agents[1]
    assert agent_0.state == TrainState.READY_TO_DEPART
    assert agent_1.state == TrainState.READY_TO_DEPART
    assert agent_0.current_entry_point is None
    assert agent_1.current_entry_point is None

    # If we now try to dispatch both trains they will be dispatched.
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})
    agent_0, agent_1 = env.agents[0], env.agents[1]
    assert agent_0.state == TrainState.MOVING
    assert agent_1.state == TrainState.MOVING

    assert np.all(agent_0.current_entry_point[0] == agent_0.initial_entry_point[0])
    assert np.all(agent_1.current_entry_point[0] == agent_1.initial_entry_point[0])

    # Thus we showed that train 0 could be dispatched at it's earliest departure and train 1 could not


def test_train_can_move_when_malfunction_counter_is_0_off_map_FIXED():
    """
    When a train goes into a malfunction off-map then in the last ts of the malfunction the agent can actually
    take an action and move (in the next ts). The malfunction_handler specifies that the agent is not in a malfunction
    but the state is still saying the agent is in a malfunction."""
    rail_env = RailEnv(
        width=30,
        height=30,
        number_of_agents=1,
        obs_builder_object=DummyObservationBuilder(),
        malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(0.1, 5, 5)),
        rail_generator=sparse_rail_generator(backwards_compatibility_mode=True),
        random_seed=1234,
    )
    _ = rail_env.reset(random_seed=1234)

    for ii in range(7):
        rail_env.step({0: RailEnvActions.DO_NOTHING})

    agent = rail_env.agents[0]
    assert agent.state == TrainState.READY_TO_DEPART

    # After performing one action the agent should go into a malfunction.
    rail_env.step({0: RailEnvActions.DO_NOTHING})
    assert agent.state == TrainState.MALFUNCTION_OFF_MAP
    # design: malfunction counter decremented at start of step(), before new malfunctions are generated
    assert agent.malfunction_handler.malfunction_down_counter == 5 + 1

    for _ in range(5):
        rail_env.step({0: RailEnvActions.MOVE_FORWARD})

    # Here we can see the bug is fixed
    assert agent.state == TrainState.MALFUNCTION_OFF_MAP
    assert agent.malfunction_handler.in_malfunction
    # design: malfunction counter decremented at start of step(), before new malfunctions are generated
    assert agent.malfunction_handler.malfunction_down_counter == 0 + 1
    # Train is not dispatched during malfunction
    assert agent.current_entry_point is None

    rail_env.step({0: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert not agent.malfunction_handler.in_malfunction
    # Train is dispatched in the step the malfunction terminates
    assert agent.current_entry_point is not None


def test_train_can_move_when_malfunction_counter_is_0_on_map_FIXED():
    """
    When a train goes into a malfunction on-map then in the last ts of the malfunction the agent can actually
    take an action and move (in the next ts). The malfunction_handler specifies that the agent is not in a malfunction
    but the state is still saying the agent is in a malfunction."""
    rail_env = RailEnv(
        width=30,
        height=30,
        number_of_agents=1,
        obs_builder_object=DummyObservationBuilder(),
        malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(0.1, 5, 5)),
        rail_generator=sparse_rail_generator(backwards_compatibility_mode=True),
        random_seed=1234,
    )
    _ = rail_env.reset(random_seed=1234)

    for ii in range(7):
        rail_env.step({0: RailEnvActions.MOVE_FORWARD})

    agent = rail_env.agents[0]
    assert agent.state == TrainState.MOVING

    # After performing one action the agent should go into a malfunction.
    rail_env.step({0: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MALFUNCTION
    # design: malfunction counter decremented at start of step(), before new malfunctions are generated
    assert agent.malfunction_handler.malfunction_down_counter == 5 + 1
    old_pos = agent.current_entry_point[0]
    old_entry_point = agent.current_entry_point

    for _ in range(5):
        rail_env.step({0: RailEnvActions.DO_NOTHING})

    # Here we can see the bug is fixed
    assert agent.state == TrainState.MALFUNCTION
    assert agent.malfunction_handler.in_malfunction
    # design: malfunction counter decremented at start of step(), before new malfunctions are generated
    assert agent.malfunction_handler.malfunction_down_counter == 0 + 1
    # Train is not moved during malfunction
    assert agent.current_entry_point == old_entry_point

    assert agent.speed_counter.speed == 0
    rail_env.step({0: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point[0] == (old_pos[0], old_pos[1])
    assert agent.speed_counter.speed == 1
    # design: distance update with pre-step speed.
    assert agent.speed_counter.distance == 0

    rail_env.step({0: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    # Train is moved in the step after the step where the malfunction terminates
    assert agent.current_entry_point != old_entry_point
    assert agent.current_entry_point[0] == (old_pos[0] + 1, old_pos[1])
    assert agent.speed_counter.speed == 1
    assert agent.speed_counter.distance == 0


def test_spawning_cell_not_reserved_if_id_is_lower_SANITYCHECK():
    """Show that if two trains have the same spawning cell and the one with the higher ID goes into maintenance on the
    dispatch action. The spawning cell is NOT reserved, such that the train with the lower ID can dispatch."""
    rail_env = RailEnv(
        width=30,
        height=30,
        number_of_agents=4,
        obs_builder_object=DummyObservationBuilder(),
        malfunction_generator=malfunction_generators.ParamMalfunctionGen(MalfunctionParameters(0.1, 5, 5)),
        rail_generator=sparse_rail_generator(backwards_compatibility_mode=True),
        random_seed=321,
    )
    _ = rail_env.reset(random_seed=321)

    for agent in rail_env.agents:
        print(f'{agent.handle} - {agent.earliest_departure}, {agent.initial_entry_point[0]}')

    for ii in range(20):
        rail_env.step({0: RailEnvActions.DO_NOTHING})

    assert rail_env.agents[3].state == TrainState.READY_TO_DEPART
    rail_env.step({3: RailEnvActions.MOVE_FORWARD})
    assert rail_env.agents[3].state == TrainState.MALFUNCTION_OFF_MAP
    # design: malfunction counter decremented at start of step(), before new malfunctions are generated
    assert rail_env.agents[3].malfunction_handler.malfunction_down_counter == 5 + 1

    assert rail_env.agents[0].state == TrainState.READY_TO_DEPART
    rail_env.step({0: RailEnvActions.MOVE_FORWARD})
    assert rail_env.agents[0].state == TrainState.MOVING
    assert rail_env.agents[0].state.is_on_map_state()


def test_spawning_cell_reserved_if_id_is_higher_FIXED():
    """Show that if two trains have the same spawning cell and the one with the lower ID goes into maintenance on the
    dispatch action. The spawning cell IS reserved, such that the train with the higher ID cannot dispatch until the
    lower one dispatches!"""
    rail_env = RailEnv(
        width=30,
        height=30,
        number_of_agents=4,
        obs_builder_object=DummyObservationBuilder(),
        malfunction_generator=malfunction_generators.ParamMalfunctionGen(MalfunctionParameters(0.1, 5, 5)),
        rail_generator=sparse_rail_generator(backwards_compatibility_mode=True),
        random_seed=2334,
    )
    _ = rail_env.reset(random_seed=2334)

    for ii in range(18):
        rail_env.step({})

    assert rail_env.agents[1].state == TrainState.READY_TO_DEPART
    rail_env.step({1: RailEnvActions.MOVE_FORWARD})
    assert rail_env.agents[1].state == TrainState.MALFUNCTION_OFF_MAP
    # design: malfunction counter decremented at start of step(), before new malfunctions are generated
    assert rail_env.agents[1].malfunction_handler.malfunction_down_counter == 5 + 1

    assert rail_env.agents[3].state == TrainState.READY_TO_DEPART
    rail_env.step({3: RailEnvActions.MOVE_FORWARD})

    # FIXED: the train with higher ID can move:
    assert rail_env.agents[3].state == TrainState.MOVING
    assert rail_env.agents[3].state.is_on_map_state()


def test_two_trains_on_same_cell_bug_FIXED():
    """
    In case all the following are true:
    - the train is in a malfunction
    - the train is ready (end of malfunction)
    - the train has an action saved
    - the next cell is occupied by a train that cannot move.
    --> then using the 'normal' stop action would result in this train being dispatched although the cell is
        occupied. Using do nothing does does not!

    N.B. this scenario used to be reproduced by scripting 30 raw actions against a fresh env (seed 34086,
    20 agents, 10 of which share initial_entry_point (22, 7)) to organically build up the contested-cell
    situation, then applying one final, decisive action. That buildup proved fragile to unrelated timing
    changes elsewhere in the env: fixing `is_cell_exit()` to use an agent's pre-step speed rather than a
    hypothetical post-transition target speed (the F9 head-on-conflict livelock fix) delays every
    malfunction-recovering agent's first move by one step, and with ten agents cycling through malfunction
    while queued behind each other for that one shared cell, those one-step delays compound across the
    fixed 30-step script until the intended situation (agent 0 still occupying (22, 7); agents 4 and 13,
    also spawning there, still waiting) is never reached at all within the script - agent 13 (and 4) simply
    run out of scripted steps.

    To make the test robust against this class of change, it now loads a snapshot of the exact intended
    pre-final-action situation - captured once via `RailEnvPersister.save()` after running the original
    30-step buildup - and applies only that final action.
    """
    rail_env, _ = RailEnvPersister.load_new(
        str(Path(__file__).parent / "test_two_trains_on_same_cell_bug_FIXED_snapshot.pkl"))

    agent_0 = rail_env.agents[0]
    agent_4 = rail_env.agents[4]
    agent_13 = rail_env.agents[13]

    # Diagnostic: pin down the loaded snapshot's pre-final-action situation - the train with the
    # lowest ID (agent 0) is in MALFUNCTION and still occupies the contested spawn cell (22, 7);
    # trains 4 and 13 (which also spawn at (22, 7)) are still off-map / queued behind it.
    assert agent_0.initial_entry_point == ((22, 7), 0)
    assert agent_0.state == TrainState.MALFUNCTION
    assert agent_0.current_entry_point == ((22, 7), 0)
    assert agent_0.malfunction_handler.malfunction_down_counter == 4
    assert agent_4.initial_entry_point == ((22, 7), 0)
    assert agent_4.state == TrainState.MALFUNCTION_OFF_MAP
    assert agent_4.current_entry_point is None
    assert agent_4.malfunction_handler.malfunction_down_counter == 0
    assert agent_13.initial_entry_point == ((22, 7), 0)
    assert agent_13.state == TrainState.MALFUNCTION
    assert agent_13.current_entry_point == ((21, 7), 0)
    assert agent_13.malfunction_handler.malfunction_down_counter == 0

    # the final, decisive action from the original 30-step script
    rail_env.step({0: 0, 1: 2, 2: 1, 3: 0, 4: 2, 5: 0, 6: 2, 7: 4, 8: 4, 9: 0, 10: 2, 11: 2, 12: 2, 13: 4,
                   14: 4, 15: 4, 16: 4, 17: 2, 18: 4, 19: 2})

    # FIXED: Check that both train 4 and 13 are not on the same cell!
    assert agent_4.state.is_off_map_state()
    assert agent_13.state.is_on_map_state()
    agent_4_position = agent_4.current_entry_point[0] if agent_4.current_entry_point is not None else None
    agent_13_position = agent_13.current_entry_point[0] if agent_13.current_entry_point is not None else None
    assert agent_4_position != agent_13_position
