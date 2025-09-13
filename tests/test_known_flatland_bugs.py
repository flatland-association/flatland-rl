"""
File holding test to check if known flatland bugs are still present.
From: https://github.com/AI4REALNET/maze-flatland/blob/33048b1e2c36fc26d1543b158d823b2b1bfd2aa4/maze_flatland/test/env/test_known_flatland_bugs.py
"""
from __future__ import annotations

import numpy as np

import flatland
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs import malfunction_generators
from flatland.envs.line_generators import SparseLineGen
from flatland.envs.malfunction_generators import MalfunctionParameters, NoMalfunctionGen, ParamMalfunctionGen
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
    TODO revise design: we could add +1 to "geometric" distance for off map states.

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
        agent.handle, agent.initial_position[0], agent.initial_position[1], agent.direction
    ]
    off_map_position = agent.initial_position

    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    min_distance_on_map = env.distance_map.get()[
        agent.handle, agent.initial_position[0], agent.initial_position[1], agent.direction
    ]
    on_map_position = agent.position
    assert np.all(on_map_position == off_map_position)
    assert min_distance_off_map == min_distance_on_map


def test_min_distance_for_off_map_trains_speed_of_half_REVISEDESIGN() -> None:
    """
    TODO revise design: we could add +1 to "geometric" distance for off map states.

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
        agent.handle, agent.initial_position[0], agent.initial_position[1], agent.direction
    ]
    off_map_position = agent.initial_position

    rail_env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    min_distance_on_map = rail_env.distance_map.get()[
        agent.handle, agent.initial_position[0], agent.initial_position[1], agent.direction
    ]
    on_map_position = agent.position
    assert np.all(on_map_position == off_map_position)
    assert min_distance_off_map == min_distance_on_map


# pylint: disable=protected-access
def test_earliest_departure_zero_bug_BYDESIGN() -> None:
    """
    TODO revise design: by design of https://flatland-association.github.io/flatland-book/environment/environment/agent.html#state-machine,
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
    assert agent_0.position is None
    assert agent_1.position is None

    # If we now try to dispatch both trains they will be dispatched.
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})
    agent_0, agent_1 = env.agents[0], env.agents[1]
    assert agent_0.state == TrainState.MOVING
    assert agent_1.state == TrainState.MOVING

    assert np.all(agent_0.position == agent_0.initial_position)
    assert np.all(agent_1.position == agent_1.initial_position)

    # Thus we showed that train 0 could be dispatched at it's earliest departure and train 1 could not


def test_train_can_move_when_malfunction_counter_is_0_off_map_BYDESIGN():
    """
    TODO revise design: updating the malfunction counter after the state transition leaves ugly situation that malfunction_counter == 0 but state is in malfunction - move to begining of step function?

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
    assert agent.malfunction_handler.malfunction_down_counter == 5

    for _ in range(5):
        rail_env.step({0: RailEnvActions.DO_NOTHING})

    # Here we can see the contradiction
    assert agent.state == TrainState.MALFUNCTION_OFF_MAP
    assert not agent.malfunction_handler.in_malfunction
    assert agent.malfunction_handler.malfunction_down_counter == 0

    # Even though the train is in a malfunction state we can dispatch it.
    rail_env.step({0: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.position is not None


def test_train_can_move_when_malfunction_counter_is_0_on_map_BYDESIGN():
    """
    TODO revise design: updating the malfunction counter after the state transition leaves ugly situation that malfunction_counter == 0 but state is in malfunction - move to begining of step function?

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
    assert agent.malfunction_handler.malfunction_down_counter == 5
    org_pos = agent.position

    for _ in range(5):
        rail_env.step({0: RailEnvActions.DO_NOTHING})

    # Here we can see the contradiction
    assert agent.state == TrainState.MALFUNCTION
    assert not agent.malfunction_handler.in_malfunction
    assert agent.malfunction_handler.malfunction_down_counter == 0

    # Even though the train is in a malfunction state we can dispatch it.
    rail_env.step({0: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.position == (org_pos[0] + 1, org_pos[1])


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
        print(f'{agent.handle} - {agent.earliest_departure}, {agent.initial_position}')

    for ii in range(20):
        rail_env.step({0: RailEnvActions.DO_NOTHING})

    assert rail_env.agents[3].state == TrainState.READY_TO_DEPART
    rail_env.step({3: RailEnvActions.MOVE_FORWARD})
    assert rail_env.agents[3].state == TrainState.MALFUNCTION_OFF_MAP
    assert rail_env.agents[3].malfunction_handler.malfunction_down_counter == 5

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
    assert rail_env.agents[1].malfunction_handler.malfunction_down_counter == 5

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
    """
    seed = 34086
    rail_env = RailEnv(
        width=30,
        height=30,
        number_of_agents=20,
        obs_builder_object=DummyObservationBuilder(),
        malfunction_generator=ParamMalfunctionGen(MalfunctionParameters(0.1, 5, 5)),
        rail_generator=sparse_rail_generator(backwards_compatibility_mode=True),
        random_seed=seed,
    )
    _ = rail_env.reset(random_seed=seed)
    actions = [
        {
            0: 4,
            1: 4,
            2: 4,
            3: 4,
            4: 4,
            5: 4,
            6: 4,
            7: 4,
            8: 4,
            9: 4,
            10: 4,
            11: 4,
            12: 4,
            13: 4,
            14: 4,
            15: 4,
            16: 4,
            17: 4,
            18: 4,
            19: 4,
        },
        {
            0: 4,
            1: 4,
            2: 4,
            3: 2,
            4: 4,
            5: 4,
            6: 4,
            7: 4,
            8: 4,
            9: 4,
            10: 4,
            11: 4,
            12: 0,
            13: 4,
            14: 4,
            15: 4,
            16: 2,
            17: 4,
            18: 0,
            19: 4,
        },
        {
            0: 4,
            1: 2,
            2: 2,
            3: 2,
            4: 0,
            5: 4,
            6: 2,
            7: 4,
            8: 0,
            9: 4,
            10: 2,
            11: 4,
            12: 0,
            13: 4,
            14: 4,
            15: 4,
            16: 2,
            17: 4,
            18: 0,
            19: 4,
        },
        {
            0: 4,
            1: 2,
            2: 0,
            3: 2,
            4: 0,
            5: 4,
            6: 2,
            7: 2,
            8: 0,
            9: 4,
            10: 2,
            11: 0,
            12: 0,
            13: 4,
            14: 4,
            15: 4,
            16: 3,
            17: 4,
            18: 0,
            19: 4,
        },
        {
            0: 4,
            1: 2,
            2: 0,
            3: 2,
            4: 0,
            5: 4,
            6: 3,
            7: 2,
            8: 0,
            9: 4,
            10: 2,
            11: 0,
            12: 0,
            13: 4,
            14: 4,
            15: 4,
            16: 1,
            17: 4,
            18: 0,
            19: 2,
        },
        {
            0: 4,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 4,
            6: 1,
            7: 0,
            8: 0,
            9: 4,
            10: 3,
            11: 0,
            12: 0,
            13: 4,
            14: 4,
            15: 0,
            16: 2,
            17: 4,
            18: 0,
            19: 2,
        },
        {
            0: 4,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 4,
            6: 2,
            7: 0,
            8: 0,
            9: 4,
            10: 1,
            11: 0,
            12: 0,
            13: 4,
            14: 0,
            15: 0,
            16: 2,
            17: 4,
            18: 2,
            19: 2,
        },
        {
            0: 4,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 4,
            6: 2,
            7: 0,
            8: 2,
            9: 2,
            10: 2,
            11: 0,
            12: 4,
            13: 4,
            14: 0,
            15: 0,
            16: 2,
            17: 4,
            18: 2,
            19: 2,
        },
        {
            0: 4,
            1: 0,
            2: 2,
            3: 0,
            4: 4,
            5: 4,
            6: 2,
            7: 0,
            8: 2,
            9: 2,
            10: 2,
            11: 0,
            12: 4,
            13: 4,
            14: 0,
            15: 0,
            16: 1,
            17: 4,
            18: 1,
            19: 2,
        },
        {
            0: 4,
            1: 0,
            2: 2,
            3: 0,
            4: 0,
            5: 4,
            6: 1,
            7: 0,
            8: 1,
            9: 2,
            10: 2,
            11: 4,
            12: 0,
            13: 4,
            14: 0,
            15: 0,
            16: 2,
            17: 0,
            18: 3,
            19: 2,
        },
        {
            0: 4,
            1: 2,
            2: 2,
            3: 2,
            4: 0,
            5: 4,
            6: 0,
            7: 2,
            8: 3,
            9: 2,
            10: 2,
            11: 2,
            12: 0,
            13: 2,
            14: 0,
            15: 2,
            16: 2,
            17: 0,
            18: 2,
            19: 2,
        },
        {
            0: 4,
            1: 2,
            2: 2,
            3: 2,
            4: 0,
            5: 4,
            6: 0,
            7: 2,
            8: 2,
            9: 2,
            10: 2,
            11: 2,
            12: 0,
            13: 2,
            14: 0,
            15: 2,
            16: 2,
            17: 0,
            18: 2,
            19: 2,
        },
        {
            0: 2,
            1: 1,
            2: 2,
            3: 0,
            4: 0,
            5: 4,
            6: 0,
            7: 0,
            8: 2,
            9: 0,
            10: 2,
            11: 2,
            12: 0,
            13: 2,
            14: 4,
            15: 2,
            16: 2,
            17: 0,
            18: 2,
            19: 2,
        },
        {
            0: 0,
            1: 3,
            2: 1,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 2,
            9: 0,
            10: 2,
            11: 0,
            12: 0,
            13: 2,
            14: 4,
            15: 2,
            16: 1,
            17: 0,
            18: 3,
            19: 2,
        },
        {
            0: 0,
            1: 3,
            2: 1,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
            10: 2,
            11: 0,
            12: 2,
            13: 2,
            14: 2,
            15: 2,
            16: 2,
            17: 2,
            18: 2,
            19: 0,
        },
        {
            0: 0,
            1: 3,
            2: 1,
            3: 0,
            4: 4,
            5: 0,
            6: 1,
            7: 0,
            8: 0,
            9: 0,
            10: 2,
            11: 0,
            12: 2,
            13: 2,
            14: 0,
            15: 2,
            16: 2,
            17: 2,
            18: 2,
            19: 0,
        },
        {
            0: 0,
            1: 2,
            2: 3,
            3: 0,
            4: 4,
            5: 0,
            6: 2,
            7: 0,
            8: 0,
            9: 0,
            10: 1,
            11: 0,
            12: 1,
            13: 2,
            14: 0,
            15: 2,
            16: 2,
            17: 2,
            18: 1,
            19: 0,
        },
        {
            0: 0,
            1: 2,
            2: 3,
            3: 2,
            4: 4,
            5: 0,
            6: 2,
            7: 2,
            8: 0,
            9: 2,
            10: 0,
            11: 0,
            12: 3,
            13: 2,
            14: 0,
            15: 2,
            16: 2,
            17: 2,
            18: 3,
            19: 0,
        },
        {
            0: 2,
            1: 2,
            2: 3,
            3: 0,
            4: 0,
            5: 2,
            6: 2,
            7: 2,
            8: 0,
            9: 2,
            10: 0,
            11: 2,
            12: 2,
            13: 2,
            14: 0,
            15: 2,
            16: 2,
            17: 2,
            18: 1,
            19: 0,
        },
        {
            0: 2,
            1: 2,
            2: 3,
            3: 0,
            4: 0,
            5: 2,
            6: 2,
            7: 2,
            8: 2,
            9: 2,
            10: 0,
            11: 2,
            12: 2,
            13: 2,
            14: 0,
            15: 2,
            16: 2,
            17: 2,
            18: 0,
            19: 2,
        },
        {
            0: 2,
            1: 2,
            2: 3,
            3: 0,
            4: 0,
            5: 0,
            6: 1,
            7: 2,
            8: 3,
            9: 2,
            10: 0,
            11: 2,
            12: 2,
            13: 2,
            14: 2,
            15: 2,
            16: 2,
            17: 2,
            18: 0,
            19: 2,
        },
        {
            0: 2,
            1: 0,
            2: 3,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 2,
            8: 2,
            9: 2,
            10: 0,
            11: 2,
            12: 3,
            13: 2,
            14: 2,
            15: 2,
            16: 0,
            17: 2,
            18: 0,
            19: 2,
        },
        {
            0: 2,
            1: 0,
            2: 3,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 2,
            8: 2,
            9: 2,
            10: 1,
            11: 2,
            12: 2,
            13: 2,
            14: 2,
            15: 2,
            16: 0,
            17: 2,
            18: 0,
            19: 2,
        },
        {
            0: 2,
            1: 0,
            2: 0,
            3: 2,
            4: 2,
            5: 0,
            6: 0,
            7: 2,
            8: 1,
            9: 2,
            10: 2,
            11: 2,
            12: 2,
            13: 2,
            14: 2,
            15: 2,
            16: 0,
            17: 2,
            18: 0,
            19: 2,
        },
        {
            0: 2,
            1: 0,
            2: 0,
            3: 4,
            4: 2,
            5: 0,
            6: 0,
            7: 4,
            8: 0,
            9: 4,
            10: 2,
            11: 4,
            12: 2,
            13: 4,
            14: 4,
            15: 2,
            16: 0,
            17: 2,
            18: 4,
            19: 2,
        },
        {
            0: 2,
            1: 0,
            2: 0,
            3: 4,
            4: 0,
            5: 2,
            6: 0,
            7: 4,
            8: 0,
            9: 4,
            10: 2,
            11: 2,
            12: 0,
            13: 0,
            14: 2,
            15: 2,
            16: 0,
            17: 2,
            18: 4,
            19: 2,
        },
        {
            0: 2,
            1: 2,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 1,
            7: 4,
            8: 0,
            9: 0,
            10: 2,
            11: 2,
            12: 0,
            13: 0,
            14: 2,
            15: 2,
            16: 2,
            17: 4,
            18: 4,
            19: 2,
        },
        {
            0: 2,
            1: 1,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 2,
            7: 4,
            8: 0,
            9: 0,
            10: 1,
            11: 2,
            12: 0,
            13: 0,
            14: 2,
            15: 2,
            16: 2,
            17: 4,
            18: 4,
            19: 2,
        },
        {
            0: 2,
            1: 2,
            2: 3,
            3: 0,
            4: 0,
            5: 0,
            6: 2,
            7: 4,
            8: 0,
            9: 0,
            10: 2,
            11: 2,
            12: 0,
            13: 0,
            14: 4,
            15: 4,
            16: 2,
            17: 2,
            18: 4,
            19: 2,
        },
        {
            0: 0,
            1: 2,
            2: 2,
            3: 0,
            4: 0,
            5: 0,
            6: 2,
            7: 4,
            8: 1,
            9: 0,
            10: 2,
            11: 2,
            12: 0,
            13: 0,
            14: 2,
            15: 4,
            16: 1,
            17: 2,
            18: 4,
            19: 2,
        },
        {
            0: 0,
            1: 2,
            2: 1,
            3: 0,
            4: 2,
            5: 0,
            6: 2,
            7: 4,
            8: 4,
            9: 0,
            10: 2,
            11: 2,
            12: 2,
            13: 4,
            14: 4,
            15: 4,
            16: 4,
            17: 2,
            18: 4,
            19: 2,
        },
    ]

    for action in actions:
        rail_env.step(action)

    # FIXED: Check that both train 4 and 13 are not on the same cell!
    agent_4 = rail_env.agents[4]
    agent_13 = rail_env.agents[13]

    assert agent_4.state.is_off_map_state()
    assert agent_13.state.is_on_map_state()
    assert agent_4.position != agent_13.position
