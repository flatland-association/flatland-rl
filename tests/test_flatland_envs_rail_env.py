#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.transition_map import GridTransitionMap
from flatland.env_generation.env_generator import env_generator, env_generator_legacy
from flatland.envs.agent_utils import EnvAgent, with_direction
from flatland.envs.grid.rail_env_grid import RailEnvTransitions, RailEnvTransitionsEnum
from flatland.envs.line_generators import sparse_line_generator, line_from_file
from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_generators import sparse_rail_generator, rail_from_file
from flatland.envs.step_utils.speed_counter import SpeedCounter, _cap_speed
from flatland.envs.step_utils.states import TrainState
from flatland.trajectories.policy_runner import PolicyRunner
from flatland.utils.rendertools import RenderTool
from flatland.utils.simple_rail import make_simple_rail
from tests.trajectories.test_policy_runner import RandomPolicy

"""Tests for `flatland` package."""


def test_save_load():
    env = RailEnv(width=30, height=30,
                  rail_generator=sparse_rail_generator(seed=1),
                  line_generator=sparse_line_generator(), number_of_agents=2)
    env.reset()

    def _position(agent: EnvAgent) -> Optional[Tuple[int, int]]:
        return agent.current_entry_point[0] if agent.current_entry_point is not None else None

    def _direction(agent: EnvAgent) -> Optional[int]:
        return agent.current_entry_point[1] if agent.current_entry_point is not None else None

    agent_1_pos = _position(env.agents[0])
    agent_1_dir = _direction(env.agents[0])
    agent_1_tar = next(iter(env.agents[0].targets))[0]
    agent_2_pos = _position(env.agents[1])
    agent_2_dir = _direction(env.agents[1])
    agent_2_tar = next(iter(env.agents[1].targets))[0]

    os.makedirs("tmp", exist_ok=True)

    RailEnvPersister.save(env, "tmp/test_save.pkl")

    # env.load("test_save.dat")
    env, env_dict = RailEnvPersister.load_new("tmp/test_save.pkl")
    assert (env.width == 30)
    assert (env.height == 30)
    assert (len(env.agents) == 2)
    assert (agent_1_pos == _position(env.agents[0]))
    assert (agent_1_dir == _direction(env.agents[0]))
    assert (agent_1_tar == next(iter(env.agents[0].targets))[0])
    assert (agent_2_pos == _position(env.agents[1]))
    assert (agent_2_dir == _direction(env.agents[1]))
    assert (agent_2_tar == next(iter(env.agents[1].targets))[0])


@pytest.mark.skip("Msgpack serializing not supported")
def test_save_load_mpk():
    env = RailEnv(width=30, height=30,
                  rail_generator=sparse_rail_generator(seed=1),
                  line_generator=sparse_line_generator(), number_of_agents=2)
    env.reset()

    os.makedirs("tmp", exist_ok=True)

    RailEnvPersister.save(env, "tmp/test_save.mpk")

    # env.load("test_save.dat")
    env2, env_dict = RailEnvPersister.load_new("tmp/test_save.mpk")
    assert (env.width == env2.width)
    assert (env.height == env2.height)
    assert (len(env2.agents) == len(env.agents))

    for agent1, agent2 in zip(env.agents, env2.agents):
        pos1 = agent1.current_entry_point[0] if agent1.current_entry_point is not None else None
        pos2 = agent2.current_entry_point[0] if agent2.current_entry_point is not None else None
        dir1 = agent1.current_entry_point[1] if agent1.current_entry_point is not None else None
        dir2 = agent2.current_entry_point[1] if agent2.current_entry_point is not None else None
        assert (pos1 == pos2)
        assert (dir1 == dir2)
        assert (next(iter(agent1.targets))[0] == next(iter(agent2.targets))[0])


@pytest.mark.skip(reason="Old file used to create env, not sure how to regenerate")
def test_rail_environment_single_agent(show=False):
    # We instantiate the following map on a 3x3 grid
    #  _  _
    # / \/ \
    # | |  |
    # \_/\_/

    transitions = RailEnvTransitions()

    if False:
        # This env creation doesn't quite work right.
        cells = transitions.transition_list
        vertical_line = cells[1]
        south_symmetrical_switch = cells[6]
        north_symmetrical_switch = transitions.rotate_transition(south_symmetrical_switch, 180)
        south_east_turn = int('0100000000000010', 2)
        south_west_turn = transitions.rotate_transition(south_east_turn, 90)
        north_east_turn = transitions.rotate_transition(south_east_turn, 270)
        north_west_turn = transitions.rotate_transition(south_east_turn, 180)

        rail_map = np.array([[south_east_turn, south_symmetrical_switch,
                              south_west_turn],
                             [vertical_line, vertical_line, vertical_line],
                             [north_east_turn, north_symmetrical_switch,
                              north_west_turn]],
                            dtype=np.uint16)

        rail = GridTransitionMap(width=3, height=3, transitions=transitions)
        rail.grid = rail_map
        rail_env = RailEnv(width=3, height=3, rail_generator=rail_from_grid_transition_map(rail),
                           line_generator=sparse_line_generator(), number_of_agents=1,
                           obs_builder_object=GlobalObsForRailEnv())
    else:
        rail_env, env_dict = RailEnvPersister.load_new("test_env_loop.pkl", "env_data.tests")
        rail_map = rail_env.rail.grid

    rail_env._max_episode_steps = 1000

    _ = rail_env.reset(False, False, True)

    liActions = [int(a) for a in RailEnvActions]

    env_renderer = RenderTool(rail_env)

    # RailEnvPersister.save(rail_env, "test_env_figure8.pkl")

    for _ in range(5):

        # rail_env.agents[0].initial_position = (1,2)
        _ = rail_env.reset(False, False, True)

        # We do not care about target for the moment
        agent = rail_env.agents[0]
        agent.targets = {([-1, -1], d) for d in Grid4TransitionsEnum}

        # Check that trains are always initialized at a consistent position
        # or direction.
        # They should always be able to go somewhere.
        agent_direction = agent.current_entry_point[1] if agent.current_entry_point is not None else None
        if show:
            print("After reset - agent pos:", agent.current_entry_point[0], "dir: ", agent_direction)
            print(transitions.get_transitions(rail_map[agent.current_entry_point[0]], agent_direction))

        # assert (transitions.get_transitions(
        #    rail_map[agent.position],
        #    agent.direction) != (0, 0, 0, 0))

        # HACK - force the direction to one we know is good.
        # agent.initial_position = agent.position = (2,3)
        agent.initial_entry_point = with_direction(agent.initial_entry_point, 0)
        agent.current_entry_point = with_direction(agent.current_entry_point, 0)

        if show:
            print("handle:", agent.handle)
        # agent.initial_position = initial_pos = agent.position

        valid_active_actions_done = 0
        pos = agent.current_entry_point[0]

        if show:
            env_renderer.render_env(show=show, show_agents=True)
            time.sleep(0.01)

        iStep = 0
        while valid_active_actions_done < 6:
            # We randomly select an action
            action = np.random.choice(liActions)
            # action = RailEnvActions.MOVE_FORWARD

            _, _, dict_done, _ = rail_env.step({0: action})

            prev_pos = pos
            pos = agent.current_entry_point[0]  # rail_env.agents_position[0]

            print("action:", action, "pos:", agent.current_entry_point[0], "prev:", prev_pos,
                  agent.current_entry_point[1] if agent.current_entry_point is not None else None)
            print(dict_done)
            if prev_pos != pos:
                valid_active_actions_done += 1
            iStep += 1

            if show:
                env_renderer.render_env(show=show, show_agents=True, step=iStep)
                time.sleep(0.01)
            assert iStep < 100, "valid actions should have been performed by now - hung agent"

        # After 6 movements on this railway network, the train should be back
        # to its original height on the map.
        # assert (initial_pos[0] == agent.position[0])

        # We check that the train always attains its target after some time
        for _ in range(10):
            _ = rail_env.reset()

            rail_env.agents[0].current_entry_point = with_direction(rail_env.agents[0].current_entry_point, 0)

            # JW - to avoid problem with sparse_line_generator.
            # rail_env.agents[0].position = (1,2)

            iStep = 0
            while iStep < 100:
                # We randomly select an action
                action = np.random.choice(liActions)

                _, _, dones, _ = rail_env.step({0: action})
                done = dones['__all__']
                if done:
                    break
                iStep += 1
                assert iStep < 100, "agent should have finished by now"
                env_renderer.render_env(show=show)


def test_dead_end():
    transitions = RailEnvTransitions()

    straight_vertical = int('1000000000100000', 2)  # Case 1 - straight
    straight_horizontal = transitions.rotate_transition(straight_vertical,
                                                        90)

    dead_end_from_south = int('0010000000000000', 2)  # Case 7 - dead end

    # We instantiate the following railway
    # O->-- where > is the train and O the target. After 6 steps,
    # the train should be done.

    rail_map = np.array(
        [[transitions.rotate_transition(dead_end_from_south, 270)] +
         [straight_horizontal] * 3 +
         [transitions.rotate_transition(dead_end_from_south, 90)]],
        dtype=np.uint16)

    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0],
                             transitions=transitions)

    rail.grid = rail_map

    city_positions = [(0, 0), (0, 3)]
    train_stations = [
        [((0, 0), 0)],
        [((0, 0), 0)],
    ]
    city_orientations = [0, 2]
    agents_hints = {'num_agents': 2,
                    'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                    }
    optionals = {'agents_hints': agents_hints}

    rail_env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                       rail_generator=rail_from_grid_transition_map(rail, optionals),
                       line_generator=sparse_line_generator(), number_of_agents=1,
                       obs_builder_object=GlobalObsForRailEnv())

    # We try the entry point in the 4 directions:
    rail_env.reset()
    rail_env.agents = [
        EnvAgent(initial_entry_point=((0, 2), 1), current_entry_point=(None, 1), targets={((0, 0), d) for d in Grid4TransitionsEnum}, moving=False)]

    rail_env.reset()
    rail_env.agents = [
        EnvAgent(initial_entry_point=((0, 2), 3), current_entry_point=(None, 3), targets={((0, 4), d) for d in Grid4TransitionsEnum}, moving=False)]

    # In the vertical entry point:
    rail_map = np.array(
        [[dead_end_from_south]] + [[straight_vertical]] * 3 +
        [[transitions.rotate_transition(dead_end_from_south, 180)]],
        dtype=np.uint16)

    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0],
                             transitions=transitions)

    city_positions = [(0, 0), (0, 3)]
    train_stations = [
        [((0, 0), 0)],
        [((0, 0), 0)],
    ]
    city_orientations = [0, 2]
    agents_hints = {'num_agents': 2,
                    'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                    }
    optionals = {'agents_hints': agents_hints}

    rail.grid = rail_map
    rail_env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                       rail_generator=rail_from_grid_transition_map(rail, optionals),
                       line_generator=sparse_line_generator(), number_of_agents=1,
                       obs_builder_object=GlobalObsForRailEnv())

    rail_env.reset()
    rail_env.agents = [
        EnvAgent(initial_entry_point=((2, 0), 2), current_entry_point=(None, 2), targets={((0, 0), d) for d in Grid4TransitionsEnum}, moving=False)]

    rail_env.reset()
    rail_env.agents = [
        EnvAgent(initial_entry_point=((2, 0), 0), current_entry_point=(None, 0), targets={((4, 0), d) for d in Grid4TransitionsEnum}, moving=False)]

    # TODO make assertions


def test_get_entry_directions():
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    env.reset()

    def _assert(position, expected):
        actual = env.rail.get_valid_directions_on_grid(*position)
        assert actual == expected, "[{},{}] actual={}, expected={}".format(*position, actual, expected)

    # north dead end
    assert env.rail.get_full_transitions(0, 3) == RailEnvTransitionsEnum.dead_end_from_south
    _assert((0, 3), [True, False, False, False])

    # west dead end
    assert env.rail.get_full_transitions(3, 0) == RailEnvTransitionsEnum.dead_end_from_east
    _assert((3, 0), [False, False, False, True])

    # switch
    assert env.rail.get_full_transitions(3, 3) == RailEnvTransitionsEnum.simple_switch_west_right
    _assert((3, 3), [False, True, True, True])

    # horizontal
    assert env.rail.get_full_transitions(3, 2) == RailEnvTransitionsEnum.horizontal_straight
    _assert((3, 2), [False, True, False, True])

    # vertical
    assert env.rail.get_full_transitions(2, 3) == RailEnvTransitionsEnum.vertical_straight
    _assert((2, 3), [True, False, True, False])

    # nowhere
    assert env.rail.get_full_transitions(0, 0) == RailEnvTransitionsEnum.empty
    _assert((0, 0), [False, False, False, False])


def test_rail_env_reset():
    file_name = "test_rail_env_reset.pkl"

    # Test to save and load file.

    rail, rail_map, optionals = make_simple_rail()

    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=3,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    env.reset()

    # env.save(file_name)
    RailEnvPersister.save(env, file_name)

    dist_map_shape = np.shape(env.distance_map.get())
    rails_initial = env.rail.grid
    agents_initial = env.agents

    # env2 = RailEnv(width=1, height=1, rail_generator=rail_from_file(file_name),
    #               line_generator=line_from_file(file_name), number_of_agents=1,
    #               obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    # env2.reset(False, False, False)
    env2, env2_dict = RailEnvPersister.load_new(file_name)

    rails_loaded = env2.rail.grid
    agents_loaded = env2.agents

    assert np.all(np.array_equal(rails_initial, rails_loaded))
    assert agents_initial == agents_loaded

    env3 = RailEnv(width=1, height=1, rail_generator=rail_from_file(file_name),
                   line_generator=line_from_file(file_name), number_of_agents=1,
                   obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    env3.reset(False, True)
    rails_loaded = env3.rail.grid
    agents_loaded = env3.agents
    # override `earliest_departure` & `latest_arrival` since they aren't expected to be the same
    for agent_initial, agent_loaded in zip(agents_initial, agents_loaded):
        agent_loaded.earliest_departure = agent_initial.earliest_departure
        agent_loaded.latest_arrival = agent_initial.latest_arrival
        agent_loaded.waypoints_earliest_departure = [agent_initial.earliest_departure, None]
        agent_loaded.waypoints_latest_arrival = [None, agent_initial.latest_arrival]

    assert np.all(np.array_equal(rails_initial, rails_loaded))
    assert agents_initial == agents_loaded

    env4 = RailEnv(width=1, height=1, rail_generator=rail_from_file(file_name),
                   line_generator=line_from_file(file_name), number_of_agents=1,
                   obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    env4.reset(True, False)
    rails_loaded = env4.rail.grid
    agents_loaded = env4.agents
    # override `earliest_departure` & `latest_arrival` since they aren't expected to be the same
    for agent_initial, agent_loaded in zip(agents_initial, agents_loaded):
        agent_loaded.earliest_departure = agent_initial.earliest_departure
        agent_loaded.latest_arrival = agent_initial.latest_arrival
        agent_loaded.waypoints_earliest_departure = [agent_initial.earliest_departure, None]
        agent_loaded.waypoints_latest_arrival = [None, agent_initial.latest_arrival]

    assert np.all(np.array_equal(rails_initial, rails_loaded))
    assert agents_initial == agents_loaded


def test_load_new_random_states():
    env, _, _ = env_generator(seed=42, )

    # env loaded has random state of env AFTER reset since generator use the same
    # TODO https://github.com/flatland-association/flatland-rl/issues/242 revise design - keep random state in generators separate (malfunction, rail etc. have their own)
    RailEnvPersister.save(env, "blup.pkl")
    loaded, _ = RailEnvPersister.load_new("blup.pkl", obs_builder=TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)))
    assert all(env.np_random.get_state()[1] == loaded.np_random.get_state()[1])

    # a reset on the original and the loaded env is different as the loaded env's rail generator only stores the rail of the saved env.
    env.reset(True, True, random_seed=42)
    loaded.reset(True, True, random_seed=42)
    assert not all(env.np_random.get_state()[1] == loaded.np_random.get_state()[1])


def test_clone_from_random_states():
    env, _, _ = env_generator(seed=42, )

    # env loaded has random state of env AFTER reset since generator use the same
    # TODO https://github.com/flatland-association/flatland-rl/issues/242 revise design - keep random state in generators separate (malfunction, rail etc. have their own)
    clone = RailEnv(30, 30)
    clone.clone_from(env, obs_builder=TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)))
    assert all(env.np_random.get_state()[1] == clone.np_random.get_state()[1])

    # a reset on the original and the cloned env is different as the clone's rail generator only stores the rail of the saved env.
    env.reset(True, True, random_seed=55)
    clone.reset(True, True, random_seed=53)
    assert not all(env.np_random.get_state()[1] == clone.np_random.get_state()[1])


def test_clone_from_with_random_policy():
    env, _, _ = env_generator(seed=42, )

    clone = RailEnv(30, 30)
    clone.clone_from(env)

    # use Trajectory API for comparison
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(env=env, policy=RandomPolicy(), data_dir=data_dir / "one")
        other = PolicyRunner.create_from_policy(env=clone, policy=RandomPolicy(), data_dir=data_dir / "two")

        assert len(trajectory.compare_arrived(other)) == 0
        assert len(trajectory.compare_actions(other)) == 0
        assert len(trajectory.compare_positions(other)) == 0
        assert len(trajectory.compare_rewards_dones_infos(other)) == 0


def test_speed_after_malfunction():
    env, _, _ = env_generator_legacy(seed=42, n_agents=1, malfunction_interval=1)
    env.acceleration_delta = Fraction(1, 10)
    agent = env.agents[0]

    initial_speed = agent.speed_counter.max_speed
    assert initial_speed == Fraction(1, 2)
    # design: speed is None until the agent enters the map
    assert agent.speed_counter.speed is None

    while not agent.state.is_on_map_state():
        env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
        if not agent.state.is_on_map_state():
            assert agent.speed_counter.speed is None
            assert agent.speed_counter.distance is None
        else:
            # departure step: distance resets to 0, speed reaches the (possibly partial)
            # acceleration delta rather than jumping straight to max_speed.
            assert agent.speed_counter.speed == _cap_speed(initial_speed, env.acceleration_delta)
            assert agent.speed_counter.distance == Fraction(0)
    while not agent.malfunction_handler.in_malfunction:
        env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    speed = agent.speed_counter.speed

    distance = agent.speed_counter.distance
    assert speed == Fraction(0)
    while agent.state.is_malfunction_state():
        # TODO revise design: set speed to 0 during malfunction?
        assert agent.speed_counter.speed == speed
        assert agent.speed_counter.distance == distance
        env.step({agent.handle: RailEnvActions.MOVE_FORWARD})

    # takes up old speed plus increment.
    assert agent.state == TrainState.MOVING
    assert agent.speed_counter.speed == speed + env.acceleration_delta
    assert agent.speed_counter.speed <= agent.speed_counter.max_speed
    # design: distance update with pre-step speed.
    assert agent.speed_counter.distance == distance


def test_speed_after_malfunction_full_acceleration_braking():
    env, _, _ = env_generator_legacy(seed=42, n_agents=1, malfunction_interval=1)
    agent = env.agents[0]

    assert agent.speed_counter.max_speed == Fraction(1, 2)
    initial_speed = agent.speed_counter.max_speed
    assert initial_speed == Fraction(1, 2)
    # design: speed is None until the agent enters the map
    assert agent.speed_counter.speed is None

    while not agent.state.is_on_map_state():
        env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
        if not agent.state.is_on_map_state():
            assert agent.speed_counter.speed is None
            assert agent.speed_counter.distance is None
        else:
            # departure step: distance resets to 0, speed reaches the (possibly partial)
            # acceleration delta rather than jumping straight to max_speed. Default (unset)
            # acceleration_delta is 1, capped at max_speed here.
            assert agent.speed_counter.speed == _cap_speed(initial_speed, env.acceleration_delta)
            assert agent.speed_counter.distance == Fraction(0)
    while not agent.malfunction_handler.in_malfunction:
        env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    speed = agent.speed_counter.speed

    distance = agent.speed_counter.distance
    assert speed == Fraction(0)
    assert distance == Fraction(1, 2)

    while agent.state.is_malfunction_state():
        assert agent.speed_counter.speed == speed
        assert agent.speed_counter.distance == distance
        previous_distance = agent.speed_counter.distance
        env.step({agent.handle: RailEnvActions.MOVE_FORWARD})

    # takes up old speed plus increment.
    assert agent.state == TrainState.MOVING
    assert agent.speed_counter.speed == Fraction(1, 2)
    # design: distance update with pre-step speed.
    assert agent.speed_counter.distance == previous_distance


def test_symmetric_switch_stop_action():
    """
    Document agent behaviour when choosing an invalid action upon entering a symmetric switch:
    entry point and next entry point must always advance together (see the invariant documented
    in RailEnv.step()), so the crossing into the switch is denied and the agent is forced to a
    stop at the cell boundary, until a genuinely valid action (MOVE_LEFT/MOVE_RIGHT) is given.
    """
    env, _, _ = env_generator_legacy(seed=43, n_agents=1)

    assert (np.count_nonzero(env.rail.grid == RailEnvTransitionsEnum.symmetric_switch_from_west) > 0)
    print(np.argwhere(env.rail.grid == RailEnvTransitionsEnum.symmetric_switch_from_west))
    assert env.rail.get_full_transitions(15, 15) == RailEnvTransitionsEnum.symmetric_switch_from_west
    assert not env.rail.is_valid_entry_point(((15, 16), 1))  # cannot enter (15,16) heading EAST == 1:

    env.braking_delta = - Fraction(1, 10)

    agent = env.agents[0]
    # design: speed is None until the agent enters the map
    assert agent.speed_counter.speed is None
    assert agent.speed_counter.max_speed == Fraction(1, 2)
    agent.initial_entry_point = ((15, 14), 1)
    assert agent.speed_counter.distance is None
    while not agent.state == TrainState.READY_TO_DEPART:
        env.step({})
        assert agent.speed_counter.distance is None
        assert agent.speed_counter.speed is None
        assert agent.current_entry_point is None

    # enter grid
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.current_entry_point[0] == (15, 14)
    assert agent.current_entry_point[1] == 1
    assert agent.state == TrainState.MOVING
    assert agent.speed_counter.speed == Fraction(1, 2)
    # TODO revise design: no distance travelled upon entering the grid despite state MOVING!
    assert agent.speed_counter.distance == Fraction(0)

    env.step({})
    assert agent.current_entry_point[0] == (15, 14)
    assert agent.current_entry_point[1] == 1
    assert agent.state == TrainState.MOVING
    assert agent.speed_counter.speed == Fraction(1, 2)
    assert agent.speed_counter.distance == Fraction(1, 2)

    # design: entry point and next entry point always advance together (see the invariant in
    # RailEnv.step()) -- STOP_MOVING has no valid transition from the pending target (15,15),1 (a
    # symmetric switch has no straight-through option), so the crossing itself is denied: the agent
    # stays parked at (15,14),1, still pending (15,15),1, forced to a stop (crossing_denied).
    env.step({agent.handle: RailEnvActions.STOP_MOVING})
    assert agent.current_entry_point == ((15, 14), 1)
    assert agent.next_entry_point == ((15, 15), 1)
    assert agent.state == TrainState.STOPPED
    assert agent.speed_counter.speed == Fraction(0)
    # design: distance update with pre-step speed
    assert agent.speed_counter.distance == Fraction(1, 1)

    # retrying the same invalid action leaves the agent parked in exactly the same state.
    env.step({agent.handle: RailEnvActions.STOP_MOVING})
    assert agent.current_entry_point == ((15, 14), 1)
    assert agent.next_entry_point == ((15, 15), 1)
    assert agent.state == TrainState.STOPPED
    assert agent.speed_counter.speed == Fraction(0)
    assert agent.speed_counter.distance == Fraction(1, 1)

    # only a genuinely valid action (MOVE_LEFT/MOVE_RIGHT) lets the agent actually enter the switch.
    env.step({agent.handle: RailEnvActions.MOVE_RIGHT})
    assert agent.current_entry_point == ((15, 15), 1)
    assert agent.next_entry_point == ((16, 15), 2)
    assert agent.state == TrainState.MOVING
    assert agent.speed_counter.speed == Fraction(1, 2)
    assert agent.speed_counter.distance == Fraction(0)


def test_symmetric_switch_move_forward_action():
    """
    Document agent behaviour when choosing an invalid action upon entering a symmetric switch:
    entry point and next entry point must always advance together (see the invariant documented
    in RailEnv.step()), so the crossing into the switch is denied and the agent is forced to a
    stop at the cell boundary, until a genuinely valid action (MOVE_LEFT/MOVE_RIGHT) is given.
    """
    env, _, _ = env_generator_legacy(seed=43, n_agents=1)

    assert (np.count_nonzero(env.rail.grid == RailEnvTransitionsEnum.symmetric_switch_from_west) > 0)
    print(np.argwhere(env.rail.grid == RailEnvTransitionsEnum.symmetric_switch_from_west))
    assert env.rail.get_full_transitions(15, 15) == RailEnvTransitionsEnum.symmetric_switch_from_west
    assert not env.rail.is_valid_entry_point(((15, 16), 1))  # cannot enter (15,16) heading EAST == 1:

    # FORWARD east (to enter the cell) on cell left of symmetric switch from west (aka. from left) is allowed:
    assert env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, ((15, 14), 1)) == ((15, 15), 1)
    # FORWARD entering symmetric switch facing is not allowed:
    assert env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, ((15, 15), 1)) is None

    agent = env.agents[0]
    # design: speed is None until the agent enters the map
    assert agent.speed_counter.speed is None
    assert agent.speed_counter.max_speed == Fraction(1, 2)
    agent.initial_entry_point = ((15, 14), 1)
    assert agent.speed_counter.distance is None
    while not agent.state == TrainState.READY_TO_DEPART:
        env.step({})
        assert agent.speed_counter.distance is None
        assert agent.speed_counter.speed is None
        assert agent.current_entry_point is None

    # enter grid
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.current_entry_point[0] == (15, 14)
    assert agent.current_entry_point[1] == 1
    assert agent.state == TrainState.MOVING
    assert agent.speed_counter.speed == Fraction(1, 2)
    assert agent.speed_counter.distance == Fraction(0)

    env.step({})
    assert agent.current_entry_point[0] == (15, 14)
    assert agent.current_entry_point[1] == 1
    assert agent.state == TrainState.MOVING
    assert agent.speed_counter.speed == Fraction(1, 2)
    assert agent.speed_counter.distance == Fraction(1, 2)

    env.step({agent.handle: RailEnvActions.STOP_MOVING})
    assert agent.current_entry_point[0] == (15, 14)
    assert agent.current_entry_point[1] == 1
    assert agent.state == TrainState.STOPPED
    assert agent.speed_counter.speed == Fraction(0)
    # design: distance update with pre-step speed.
    assert agent.speed_counter.distance == Fraction(1, 1)
    assert agent.current_entry_point == ((15, 14), 1)
    assert agent.next_entry_point == ((15, 15), 1)

    # design: entry point and next entry point always advance together (see the invariant in
    # RailEnv.step()) -- MOVE_FORWARD has no valid transition from the pending target (15,15),1 (a
    # symmetric switch has no straight-through option), so the crossing itself is denied: the agent
    # stays parked at (15,14),1, still pending (15,15),1, forced to a stop exactly like an ordinary
    # invalid action would (crossing_denied), even though a movement action was given.
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.current_entry_point == ((15, 14), 1)
    assert agent.next_entry_point == ((15, 15), 1)
    assert agent.state == TrainState.STOPPED
    assert agent.speed_counter.speed == Fraction(0)
    assert agent.speed_counter.distance == Fraction(1, 1)

    # retrying the same invalid action leaves the agent parked in exactly the same state.
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.current_entry_point == ((15, 14), 1)
    assert agent.next_entry_point == ((15, 15), 1)
    assert agent.state == TrainState.STOPPED
    assert agent.speed_counter.speed == Fraction(0)
    assert agent.speed_counter.distance == Fraction(1, 1)

    # only a genuinely valid action (MOVE_LEFT/MOVE_RIGHT) lets the agent actually enter the switch.
    env.step({agent.handle: RailEnvActions.MOVE_LEFT})
    assert agent.current_entry_point == ((15, 15), 1)
    assert agent.next_entry_point == ((14, 15), 0)
    assert agent.state == TrainState.MOVING
    assert agent.speed_counter.speed == Fraction(1, 2)
    assert agent.speed_counter.distance == Fraction(0)


def test_blocked_agent_cannot_redirect_via_later_action():
    """
    Document a real behavioural consequence of "actions applied at cell entry": once an agent's
    next_entry_point (pending target B) is decided - one cell before it is even reached - it is
    held fixed across however many retries it takes to actually be granted, no matter what action
    is given on those retries. Before this design, B was recomputed fresh from the agent's current
    cell on every retry, so giving a turn action while blocked could redirect the agent onto a
    different cell and break e.g. a symmetric head-on deadlock (see the two_trains_on_same_cell-
    style scenarios). Now it can't: a later action only ever affects the *next* look-ahead beyond
    B, which is discarded unless/until B is actually entered.

    Rail: `make_simple_rail()`'s row 3 corridor, agent 0 heading west from (3, 8) through the
    switch at (3, 6) (which also has a valid southward branch) towards (3, 5) - blocked there for
    several steps by agent 1 parked at (3, 5). While blocked, agent 0 is repeatedly given
    MOVE_LEFT, which *would* redirect it onto the southward branch at (4, 6) if evaluated fresh
    from its current cell (3, 6) - but does not, since it is evaluated from the already-pending
    target (3, 5) instead (a plain straight cell, so MOVE_LEFT there just corrects to (3, 4)).
    """
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=2,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  random_seed=1)
    env.reset()
    agent0, agent1 = env.agents[0], env.agents[1]
    agent0.initial_entry_point = ((3, 8), 3)
    agent1.initial_entry_point = ((3, 5), 3)
    agent0.earliest_departure = 0
    agent1.earliest_departure = 0
    # agent 1 must genuinely stay parked at (3, 5) to block agent 0: at max_speed 1, its pre-step
    # momentum on departure (distance 0, speed 1) would already reach the cell boundary on the very
    # next step, so STOP_MOVING would complete an in-flight crossing out of (3, 5) instead of
    # holding it there (see the STOP_MOVING boundary-crossing fix, issue #178 design D2a). A slower
    # max_speed keeps it mid-cell (is_cell_exit false) when STOP_MOVING brakes it to 0. Only
    # _max_speed needs overriding here - agent1 hasn't departed yet, so _speed must stay None (off
    # map); departure (see (3a.3)) computes candidate_speed from acceleration_delta/_max_speed
    # regardless of any pre-existing _speed, so it was never read anyway.
    agent1.speed_counter._max_speed = Fraction(1, 2)

    # MOVE_LEFT from the switch at (3, 6) is a genuine, valid redirect onto the southward branch -
    # confirms the escape route agent 0 will be denied further down is real, not just invalid input.
    assert env.rail.apply_action_independent(RailEnvActions.MOVE_LEFT, ((3, 6), 3)) == ((4, 6), 2)

    while agent0.state != TrainState.READY_TO_DEPART:
        env.step({})

    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})  # depart both
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.STOP_MOVING})  # agent 0 -> (3, 7), pending (3, 6)
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.STOP_MOVING})  # agent 0 -> (3, 6), pending (3, 5)

    # agent 0 attempts to enter (3, 5), denied - agent 1 is parked there.
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.STOP_MOVING})
    assert agent0.current_entry_point == ((3, 6), 3)
    assert agent0.next_entry_point == ((3, 5), 3)
    assert agent0.state == TrainState.STOPPED

    # giving MOVE_LEFT while blocked does NOT redirect the pending target onto the southward
    # branch at (4, 6) - it stays locked onto (3, 5), retried for as long as agent 1 blocks it.
    for _ in range(2):
        env.step({0: RailEnvActions.MOVE_LEFT, 1: RailEnvActions.STOP_MOVING})
        assert agent0.current_entry_point == ((3, 6), 3)
        assert agent0.next_entry_point == ((3, 5), 3)
        assert agent0.state == TrainState.STOPPED

    # once agent 1 vacates (3, 5), agent 0 enters it and continues straight to (3, 4) - even though
    # it is still being given MOVE_LEFT - confirming the earlier MOVE_LEFTs were never consulted
    # for the (3, 6) -> (3, 5) crossing itself, only (uselessly) for the look-ahead beyond it. Agent
    # 1 was braked to a stop at its reduced max_speed, so it needs a few steps of MOVE_FORWARD to
    # regain enough momentum before it actually crosses out of (3, 5).
    while agent0.current_entry_point != ((3, 5), 3):
        env.step({0: RailEnvActions.MOVE_LEFT, 1: RailEnvActions.MOVE_FORWARD})
    assert agent0.next_entry_point == ((3, 4), 3)
    assert agent0.state == TrainState.MOVING


def test_earliest_departure_state_transitions_initial_speed_zero():
    """
    Document state transitions WAITING -> READY_TO_DEPART -> MOVING for a single agent with max
    speed 1 and acceleration delta 0.5, issuing MOVE_FORWARD from the very first step. Tracks
    entry point, speed and state as the agent accelerates and crosses several entry points.
    """
    env, _, _ = env_generator(seed=42, n_agents=1)
    env.acceleration_delta = Fraction(1, 2)
    agent = env.agents[0]
    agent.speed_counter = SpeedCounter(max_speed=Fraction(1))
    agent.earliest_departure = 3

    # WAITING: off map, earliest departure not yet reached - no speed/distance progress.
    while agent.state == TrainState.WAITING:
        env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
        assert agent.current_entry_point is None
        assert agent.speed_counter.speed is None
        assert agent.speed_counter.distance is None

    # READY_TO_DEPART: still off map, waiting for a valid MOVE_FORWARD to actually depart.
    assert agent.state == TrainState.READY_TO_DEPART
    assert agent.current_entry_point is None

    # READY_TO_DEPART -> MOVING: agent appears at its initial entry point this very step, distance
    # resets to 0 and speed reaches the acceleration delta immediately (rather than staying at 0
    # for one more step).
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == agent.initial_entry_point
    assert agent.speed_counter.speed == Fraction(1, 2)
    assert agent.speed_counter.distance == Fraction(0)

    first_entry_point = agent.current_entry_point
    second_entry_point = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, first_entry_point)
    third_entry_point = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, second_entry_point)
    fourth_entry_point = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, third_entry_point)

    # design: distance update with pre-step speed - still in the first cell (pre-step speed was
    # only 1/2), but already accelerated to max speed for the next step.
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == first_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(1, 2)

    # Now at max speed with a full cell's worth of distance already banked: the agent overshoots into
    # the entry point after the first one.
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == second_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(1, 2)

    # At max speed, the agent advances exactly one entry point per step from here on.
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == third_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(1, 2)

    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == fourth_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(1, 2)


# N.B. test_earliest_departure_state_transitions_initial_speed_equals_max_speed used to live here,
# comparing initial_speed=0 vs. initial_speed=max_speed at construction for an off-map agent, to prove
# departure always (re-)accelerates from 0 regardless of whatever speed a SpeedCounter was constructed
# with beforehand. Design D3 (branch 178-agents-living-on-the-edge-11) made that comparison impossible
# to even construct: a SpeedCounter with a non-None speed for an off-map agent (current_entry_point is
# None) now violates the off/on-map invariant (see "Speed/distance are Fractions, and None while off map"
# in CLAUDE.md) and is rejected by _check_off_on_map_invariant on the very next step(), rather than being
# silently discarded on departure. The analogous on-map scenario (agent constructed directly on the map
# with speed == max_speed, then stepped) is already covered by test_multi_speed.py's
# test_multi_speed_init and test_multispeed_actions_no_malfunction_no_blocking, so this test was
# removed rather than repurposed to avoid duplicating that coverage.


def test_earliest_departure_state_transitions_full_acceleration():
    """
    Same as test_earliest_departure_state_transitions_initial_speed_zero, but with acceleration
    delta equal to max speed (1): the agent reaches max speed in a single accelerating step, so
    distance resets to 0 cleanly on every crossing from then on (no fractional cruise offset).
    """
    env, _, _ = env_generator(seed=42, n_agents=1)
    env.acceleration_delta = Fraction(1)
    agent = env.agents[0]
    agent.speed_counter = SpeedCounter(max_speed=Fraction(1))
    agent.earliest_departure = 3

    # WAITING: off map, earliest departure not yet reached - no speed/distance progress.
    while agent.state == TrainState.WAITING:
        env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
        assert agent.current_entry_point is None
        assert agent.speed_counter.speed is None
        assert agent.speed_counter.distance is None
    assert env._elapsed_steps == 3  # _elapsed_steps +3 (3 WAITING steps)

    # READY_TO_DEPART: still off map, waiting for a valid MOVE_FORWARD to actually depart.
    assert agent.state == TrainState.READY_TO_DEPART
    assert agent.current_entry_point is None

    # READY_TO_DEPART -> MOVING: agent appears at its initial entry point this very step. Full
    # acceleration delta reaches max speed immediately (acceleration_delta equals max_speed here).
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == 4  # _elapsed_steps +1
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == agent.initial_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(0)

    first_entry_point = agent.current_entry_point
    second_entry_point = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, first_entry_point)
    third_entry_point = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, second_entry_point)
    fourth_entry_point = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, third_entry_point)

    # Already at max speed since departure: the agent advances exactly one entry point every step,
    # distance resetting to 0 cleanly (unlike a fractional acceleration delta, see
    # test_earliest_departure_state_transitions_initial_speed_zero's 1/2 cruise offset).
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == 5  # _elapsed_steps +1
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == second_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(0)

    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == 6  # _elapsed_steps +1
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == third_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(0)

    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == 7  # _elapsed_steps +1
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == fourth_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(0)


def test_earliest_departure_state_transitions_partial_acceleration():
    """
    Same as test_earliest_departure_state_transitions_initial_speed_zero, but with acceleration
    delta 0.3 (max speed 1): ramps up over more steps, and settles into a 0.8 cruise offset once
    at max speed (instead of 0.5 for delta 0.5, or 0 for a full delta) - the cruise offset is
    1 - (max_speed % acceleration_delta)-driven and differs per delta.
    """
    env, _, _ = env_generator(seed=42, n_agents=1)
    env.acceleration_delta = Fraction(3, 10)
    agent = env.agents[0]
    agent.speed_counter = SpeedCounter(max_speed=Fraction(1))
    agent.earliest_departure = 3

    # WAITING: off map, earliest departure not yet reached - no speed/distance progress.
    while agent.state == TrainState.WAITING:
        env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
        assert agent.current_entry_point is None
        assert agent.speed_counter.speed is None
        assert agent.speed_counter.distance is None
    assert env._elapsed_steps == 3  # _elapsed_steps +3 (3 WAITING steps)

    # READY_TO_DEPART: still off map, waiting for a valid MOVE_FORWARD to actually depart.
    assert agent.state == TrainState.READY_TO_DEPART
    assert agent.current_entry_point is None

    # READY_TO_DEPART -> MOVING: agent appears at its initial entry point this very step, distance
    # resets to 0 and speed reaches the acceleration delta immediately.
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == 4  # _elapsed_steps +1
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == agent.initial_entry_point
    assert agent.speed_counter.speed == Fraction(3, 10)
    assert agent.speed_counter.distance == Fraction(0)

    first_entry_point = agent.current_entry_point
    second_entry_point = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, first_entry_point)
    third_entry_point = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, second_entry_point)
    fourth_entry_point = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, third_entry_point)

    # design: distance update with pre-step speed. Ramping 0.3 -> 0.6 -> 0.9, distance
    # accumulating the pre-step speed each time, staying in the initial entry point throughout.
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == 5  # _elapsed_steps +1
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == first_entry_point
    assert agent.speed_counter.speed == Fraction(6, 10)
    assert agent.speed_counter.distance == Fraction(3, 10)

    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == 6  # _elapsed_steps +1
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == first_entry_point
    assert agent.speed_counter.speed == Fraction(9, 10)
    assert agent.speed_counter.distance == Fraction(9, 10)

    # distance(0.9) + speed(0.9) = 1.8 >= 1: crosses into the second entry point, wraps to 0.8, and
    # speed finally saturates at max speed (0.9 + 0.3 capped at 1).
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == 7  # _elapsed_steps +1
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == second_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(8, 10)

    # At max speed, the agent advances exactly one entry point per step from here on, cruising at
    # a steady 0.8 offset (0.8 + 1 = 1.8, wraps to 0.8 again).
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == 8  # _elapsed_steps +1
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == third_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(8, 10)

    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == 9  # _elapsed_steps +1
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == fourth_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(8, 10)


def test_malfunction_off_map_state_transitions_to_moving():
    """
    Document state transitions WAITING -> MALFUNCTION_OFF_MAP -> MOVING for a single agent that malfunctions
    before ever departing, issuing MOVE_FORWARD throughout. Once the malfunction clears (earliest
    departure is already reached), the agent enters the grid directly at its initial
    entry point, skipping READY_TO_DEPART.
    """
    env, _, _ = env_generator(seed=42, n_agents=1)
    env.acceleration_delta = Fraction(1, 2)
    agent = env.agents[0]
    agent.earliest_departure = 2
    agent.malfunction_handler.malfunction_down_counter = 2

    # WAITING -> MALFUNCTION_OFF_MAP: malfunction takes priority over WAITING, regardless of
    # earliest departure - step through until reached.
    while agent.state == TrainState.WAITING:
        env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
        assert agent.current_entry_point is None
        assert agent.speed_counter.speed is None
        assert agent.speed_counter.distance is None
    assert agent.state == TrainState.MALFUNCTION_OFF_MAP

    # MALFUNCTION_OFF_MAP: off map, still malfunctioning - no speed/distance progress.
    # design: malfunction counter decremented at start of step(), before new malfunctions are generated,
    # so the MALFUNCTION_OFF_MAP -> MOVING transition merges into the last loop iteration below (no
    # separate transition step needed) - guard the malfunction-only assertions accordingly.
    while agent.malfunction_handler.in_malfunction:
        env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
        if agent.state == TrainState.MALFUNCTION_OFF_MAP:
            assert agent.current_entry_point is None
            assert agent.speed_counter.speed is None
            assert agent.speed_counter.distance is None

    # MALFUNCTION_OFF_MAP -> MOVING: agent appears at its initial entry point this very step,
    # distance resets to 0 and speed reaches the acceleration delta immediately.
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == agent.initial_entry_point
    assert agent.speed_counter.speed == Fraction(1, 2)
    assert agent.speed_counter.distance == Fraction(0)

    first_entry_point = agent.current_entry_point
    second_entry_point = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, first_entry_point)
    third_entry_point = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, second_entry_point)
    fourth_entry_point = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, third_entry_point)

    # design: distance update with pre-step speed - still in the first cell (pre-step speed was
    # only 1/2), but already accelerated to max speed for the next step.
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == first_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(1, 2)

    # Now at max speed with a full cell's worth of distance already banked: the agent overshoots
    # into the entry point after the first one, and cruises with a steady 1/2 offset from here on.
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == second_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(1, 2)

    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == third_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(1, 2)

    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == fourth_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(1, 2)


def test_malfunction_off_map_state_transitions_to_ready_to_depart():
    """
    Document state transition WAITING -> MALFUNCTION_OFF_MAP -> READY_TO_DEPART for a single agent that
    malfunctions before ever departing, issuing DO_NOTHING throughout. Once the malfunction clears
    (earliest departure is already reached) but no movement/stop action is given, the agent
    becomes READY_TO_DEPART, staying off map.
    """
    env, _, _ = env_generator(seed=42, n_agents=1)
    env.acceleration_delta = Fraction(1, 2)
    agent = env.agents[0]
    agent.earliest_departure = 2
    agent.malfunction_handler.malfunction_down_counter = 2

    # WAITING -> MALFUNCTION_OFF_MAP: malfunction takes priority over WAITING, regardless of
    # earliest departure - step through until reached.
    while agent.state == TrainState.WAITING:
        env.step({agent.handle: RailEnvActions.DO_NOTHING})
        assert agent.current_entry_point is None
        assert agent.speed_counter.speed is None
        assert agent.speed_counter.distance is None
    assert agent.state == TrainState.MALFUNCTION_OFF_MAP

    # MALFUNCTION_OFF_MAP: off map, still malfunctioning - no speed/distance progress.
    # design: malfunction counter decremented at start of step(), before new malfunctions are generated,
    # so the MALFUNCTION_OFF_MAP -> READY_TO_DEPART transition merges into the last loop iteration below
    # (no separate transition step needed) - guard the malfunction-only assertions accordingly.
    while agent.malfunction_handler.in_malfunction:
        env.step({agent.handle: RailEnvActions.DO_NOTHING})
        if agent.state == TrainState.MALFUNCTION_OFF_MAP:
            assert agent.current_entry_point is None
            assert agent.speed_counter.speed is None
            assert agent.speed_counter.distance is None
    assert agent.state == TrainState.READY_TO_DEPART

    # MALFUNCTION_OFF_MAP -> READY_TO_DEPART: malfunction cleared and earliest departure already
    # reached, but no movement/stop action given - the agent stays off map, ready to depart.
    for _ in range(3):
        env.step({agent.handle: RailEnvActions.DO_NOTHING})
        assert agent.state == TrainState.READY_TO_DEPART
        assert agent.current_entry_point is None
        assert agent.speed_counter.speed is None
        assert agent.speed_counter.distance is None


def test_malfunction_off_map_state_transitions_to_ready_to_depart_with_stop_action():
    """
    Document state transition WAITING -> MALFUNCTION_OFF_MAP -> READY_TO_DEPART for a single agent that
    malfunctions before ever departing, issuing STOP_MOVING throughout. Once the malfunction clears
    (earliest departure is already reached) with only a stop action given, the agent stays off map,
    ready to depart.
    """
    env, _, _ = env_generator(seed=42, n_agents=1)
    env.acceleration_delta = Fraction(1, 2)
    agent = env.agents[0]
    agent.earliest_departure = 2
    agent.malfunction_handler.malfunction_down_counter = 2

    # WAITING -> MALFUNCTION_OFF_MAP: malfunction takes priority over WAITING, regardless of
    # earliest departure - step through until reached.
    while agent.state == TrainState.WAITING:
        env.step({agent.handle: RailEnvActions.STOP_MOVING})
        assert agent.current_entry_point is None
        assert agent.speed_counter.speed is None
        assert agent.speed_counter.distance is None
    assert agent.state == TrainState.MALFUNCTION_OFF_MAP

    # MALFUNCTION_OFF_MAP: off map, still malfunctioning - no speed/distance progress.
    # design: malfunction counter decremented at start of step(), before new malfunctions are generated,
    # so the MALFUNCTION_OFF_MAP -> READY_TO_DEPART transition merges into the last loop iteration below
    # (no separate transition step needed) - guard the malfunction-only assertions accordingly.
    while agent.malfunction_handler.in_malfunction:
        env.step({agent.handle: RailEnvActions.STOP_MOVING})
        if agent.state == TrainState.MALFUNCTION_OFF_MAP:
            assert agent.current_entry_point is None
            assert agent.speed_counter.speed is None
            assert agent.speed_counter.distance is None
    assert agent.state == TrainState.READY_TO_DEPART

    # design: disallow entering the map stopped
    for _ in range(3):
        env.step({agent.handle: RailEnvActions.STOP_MOVING})
        assert agent.state == TrainState.READY_TO_DEPART
        assert agent.current_entry_point is None
        assert agent.speed_counter.speed is None
        assert agent.speed_counter.distance is None


def test_malfunction_state_transitions_to_moving():
    """
    Document state transitions MOVING -> MALFUNCTION -> MOVING for a single agent that malfunctions
    while on the map and moving, issuing MOVE_FORWARD throughout. Once the malfunction clears, the
    agent resumes moving, re-accelerating from 0 exactly like a fresh departure.
    """
    env, _, _ = env_generator(seed=42, n_agents=1)
    env.acceleration_delta = Fraction(1, 2)
    agent = env.agents[0]

    # Get the agent on the map and moving first: off map (WAITING/READY_TO_DEPART), no speed/distance
    # progress while waiting to depart.
    while agent.state != TrainState.MOVING:
        assert agent.state in (TrainState.WAITING, TrainState.READY_TO_DEPART)
        assert agent.current_entry_point is None
        assert agent.speed_counter.speed is None
        assert agent.speed_counter.distance is None
        env.step({agent.handle: RailEnvActions.MOVE_FORWARD})

    # MOVING: distance resets to 0 and speed reaches the acceleration delta immediately.
    assert agent.speed_counter.speed == Fraction(1, 2)
    assert agent.speed_counter.distance == Fraction(0)

    agent.malfunction_handler.malfunction_down_counter = 2
    malfunction_entry_point = agent.current_entry_point
    distance = agent.speed_counter.distance

    # MOVING -> MALFUNCTION: speed is reset to 0 and distance freezes at whatever it was when the malfunction hit.
    # design: malfunction counter decremented at start of step(), before new malfunctions are generated,
    # so the MALFUNCTION -> MOVING transition (re-acceleration with pre-step speed 0) merges into the last
    # loop iteration below - guard the malfunction-only assertions accordingly.
    while agent.malfunction_handler.in_malfunction:
        env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
        if agent.state == TrainState.MALFUNCTION:
            assert agent.current_entry_point == malfunction_entry_point
            assert agent.speed_counter.speed == Fraction(0)
            assert agent.speed_counter.distance == distance
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == malfunction_entry_point
    assert agent.speed_counter.speed == Fraction(1, 2)
    assert agent.speed_counter.distance == Fraction(0)

    second_entry_point = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, malfunction_entry_point)
    third_entry_point = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, second_entry_point)

    # design: distance update with pre-step speed.
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == malfunction_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(1, 2)

    # Now at max speed with a full cell's worth of distance already banked: the agent overshoots into
    # the next entry point.
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == second_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(1, 2)

    # At max speed, the agent advances exactly one entry point per step from here on.
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == third_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(1, 2)


def test_malfunction_state_transitions_to_stopped():
    """
    Document state transitions MOVING -> MALFUNCTION -> STOPPED for a single agent that malfunctions
    while on the map and moving, issuing STOP_MOVING throughout. Once the malfunction clears, no
    movement action was given, so the agent stops in place rather than resuming.
    """
    env, _, _ = env_generator(seed=42, n_agents=1)
    env.acceleration_delta = Fraction(1, 2)
    agent = env.agents[0]

    # Get the agent on the map and moving first: off map (WAITING/READY_TO_DEPART), no speed/distance
    # progress while waiting to depart.
    while agent.state != TrainState.MOVING:
        assert agent.state in (TrainState.WAITING, TrainState.READY_TO_DEPART)
        assert agent.current_entry_point is None
        assert agent.speed_counter.speed is None
        assert agent.speed_counter.distance is None
        env.step({agent.handle: RailEnvActions.MOVE_FORWARD})

    # MOVING: distance resets to 0 and speed reaches the acceleration delta immediately.
    assert agent.speed_counter.speed == Fraction(1, 2)
    assert agent.speed_counter.distance == Fraction(0)

    agent.malfunction_handler.malfunction_down_counter = 2
    malfunction_entry_point = agent.current_entry_point
    distance = agent.speed_counter.distance

    # MOVING -> MALFUNCTION: speed is reset to 0 and distance freezes at whatever it was when the malfunction hit.
    # design: malfunction counter decremented at start of step(), before new malfunctions are generated,
    # so the MALFUNCTION -> STOPPED transition merges into the last loop iteration below (no separate
    # transition step needed) - guard the malfunction-only assertions accordingly.
    while agent.malfunction_handler.in_malfunction:
        env.step({agent.handle: RailEnvActions.STOP_MOVING})
        if agent.state == TrainState.MALFUNCTION:
            assert agent.current_entry_point == malfunction_entry_point
            assert agent.speed_counter.speed == Fraction(0)
            assert agent.speed_counter.distance == distance
    assert agent.state == TrainState.STOPPED

    # MALFUNCTION -> STOPPED: malfunction cleared but no movement action given - the agent stops in
    # place rather than resuming.
    for _ in range(3):
        env.step({agent.handle: RailEnvActions.STOP_MOVING})
        assert agent.state == TrainState.STOPPED
        assert agent.current_entry_point == malfunction_entry_point
        assert agent.speed_counter.speed == Fraction(0)
        assert agent.speed_counter.distance == distance


def test_malfunction_state_transitions_to_stopped_do_nothing():
    """
    Document state transitions MOVING -> MALFUNCTION -> STOPPED for a single agent that malfunctions
    while on the map and moving, issuing DO_NOTHING throughout. Behaves identically to issuing
    STOP_MOVING: once the malfunction clears, DO_NOTHING is not treated as a movement action either,
    so the agent stops in place rather than resuming.
    """
    env, _, _ = env_generator(seed=42, n_agents=1)
    env.acceleration_delta = Fraction(1, 2)
    agent = env.agents[0]

    # Get the agent on the map and moving first: off map (WAITING/READY_TO_DEPART), no speed/distance
    # progress while waiting to depart.
    while agent.state != TrainState.MOVING:
        assert agent.state in (TrainState.WAITING, TrainState.READY_TO_DEPART)
        assert agent.current_entry_point is None
        assert agent.speed_counter.speed is None
        assert agent.speed_counter.distance is None
        env.step({agent.handle: RailEnvActions.MOVE_FORWARD})

    # MOVING: distance resets to 0 and speed reaches the acceleration delta immediately.
    assert agent.speed_counter.speed == Fraction(1, 2)
    assert agent.speed_counter.distance == Fraction(0)

    agent.malfunction_handler.malfunction_down_counter = 2
    malfunction_entry_point = agent.current_entry_point
    distance = agent.speed_counter.distance

    # MOVING -> MALFUNCTION: speed is reset to 0 and distance freezes at whatever it was when the malfunction hit.
    # design: malfunction counter decremented at start of step(), before new malfunctions are generated,
    # so the MALFUNCTION -> STOPPED transition merges into the last loop iteration below (no separate
    # transition step needed) - guard the malfunction-only assertions accordingly.
    while agent.malfunction_handler.in_malfunction:
        env.step({agent.handle: RailEnvActions.DO_NOTHING})
        if agent.state == TrainState.MALFUNCTION:
            assert agent.current_entry_point == malfunction_entry_point
            assert agent.speed_counter.speed == Fraction(0)
            assert agent.speed_counter.distance == distance
    assert agent.state == TrainState.STOPPED

    # MALFUNCTION -> STOPPED: malfunction cleared but DO_NOTHING is not a movement action either -
    # the agent stops in place rather than resuming.
    for _ in range(3):
        env.step({agent.handle: RailEnvActions.DO_NOTHING})
        assert agent.state == TrainState.STOPPED
        assert agent.current_entry_point == malfunction_entry_point
        assert agent.speed_counter.speed == Fraction(0)
        assert agent.speed_counter.distance == distance
