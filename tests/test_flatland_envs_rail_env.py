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
from flatland.envs.agent_utils import EnvAgent, with_direction, _sanitize_entry_point
from flatland.envs.grid.rail_env_grid import RailEnvTransitions, RailEnvTransitionsEnum
from flatland.envs.line_generators import sparse_line_generator, line_from_file
from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_generators import sparse_rail_generator, rail_from_file
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rewards import BaseDefaultRewards, DefaultPenalties
from flatland.envs.step_utils.speed_counter import SpeedCounter, _cap_speed
from flatland.envs.step_utils.states import TrainState
from flatland.trajectories.policy_runner import PolicyRunner
from flatland.utils.rendertools import RenderTool
from flatland.utils.simple_rail import make_simple_rail
from tests.test_flatland_rail_agent_status import _make_straight_rail, _place_agent_on_map
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
    stop at the cell boundary, until a genuinely valid action (MOVE_LEFT/MOVE_RIGHT) is given. The
    forced MOVING->STOPPED transition that denies the crossing draws an INVALID_ACTION penalty
    (pre-step speed times collision_factor, see BaseDefaultRewards.step_reward) - never a COLLISION
    one, since there is no other agent to conflict with - and only on that one transition step, not
    on every retry of the same denied action.
    """
    env, _, _ = env_generator_legacy(seed=43, n_agents=1, rewards=BaseDefaultRewards(collision_factor=COLLISION_FACTOR))

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
    _, rewards, _, _ = env.step({agent.handle: RailEnvActions.STOP_MOVING})
    assert agent.current_entry_point == ((15, 14), 1)
    assert agent.next_entry_point == ((15, 15), 1)
    assert agent.state == TrainState.STOPPED
    assert agent.speed_counter.speed == Fraction(0)
    # design: an invalid action denies the crossing at the cell boundary, same consequence as a
    # resource_check denial - distance still banks up to the boundary (real physical momentum, not
    # credited with the crossing), it just doesn't coast past it.
    assert agent.speed_counter.distance == Fraction(1, 1)
    # this MOVING->STOPPED transition is env-forced by an invalid action (not a motion-check
    # conflict, there is no other agent) - charged the full collision penalty (pre-step speed 1/2
    # times collision_factor) under INVALID_ACTION, not COLLISION.
    assert rewards[agent.handle][DefaultPenalties.INVALID_ACTION.value] == -1 * Fraction(1, 2) * COLLISION_FACTOR
    assert rewards[agent.handle][DefaultPenalties.COLLISION.value] == 0

    # retrying the same invalid action leaves the agent parked in exactly the same state - already
    # STOPPED going in, so no MOVING->STOPPED transition occurs this step and no penalty is charged.
    _, rewards, _, _ = env.step({agent.handle: RailEnvActions.STOP_MOVING})
    assert agent.current_entry_point == ((15, 14), 1)
    assert agent.next_entry_point == ((15, 15), 1)
    assert agent.state == TrainState.STOPPED
    assert agent.speed_counter.speed == Fraction(0)
    assert agent.speed_counter.distance == Fraction(1, 1)
    assert rewards[agent.handle][DefaultPenalties.INVALID_ACTION.value] == 0
    assert rewards[agent.handle][DefaultPenalties.COLLISION.value] == 0

    # design: stopped->moving with pre-speed 0, travelling no distance. An optimistic STOPPED->MOVING
    # resumption, not a forced stop - no penalty.
    _, rewards, _, _ = env.step({agent.handle: RailEnvActions.MOVE_RIGHT})
    assert agent.current_entry_point == ((15, 14), 1)
    assert agent.next_entry_point == ((15, 15), 1)
    assert agent.state == TrainState.MOVING
    assert agent.speed_counter.speed == Fraction(1, 2)
    assert agent.speed_counter.distance == Fraction(1, 1)
    assert rewards[agent.handle][DefaultPenalties.INVALID_ACTION.value] == 0
    assert rewards[agent.handle][DefaultPenalties.COLLISION.value] == 0

    # now genuinely moving (pre-step speed > 0) - this step actually completes the crossing, no denial.
    _, rewards, _, _ = env.step({agent.handle: RailEnvActions.MOVE_RIGHT})
    assert agent.current_entry_point == ((15, 15), 1)
    assert agent.next_entry_point == ((16, 15), 2)
    assert agent.state == TrainState.MOVING
    assert agent.speed_counter.speed == Fraction(1, 2)
    assert agent.speed_counter.distance == Fraction(1, 2)
    assert rewards[agent.handle][DefaultPenalties.INVALID_ACTION.value] == 0
    assert rewards[agent.handle][DefaultPenalties.COLLISION.value] == 0


def test_symmetric_switch_move_forward_action():
    """
    Document agent behaviour when choosing an invalid action upon entering a symmetric switch:
    entry point and next entry point must always advance together (see the invariant documented
    in RailEnv.step()), so the crossing into the switch is denied and the agent is forced to a
    stop at the cell boundary. A moving-action retry (MOVE_FORWARD, still invalid at this switch)
    is optimistically promoted back to MOVING regardless (SpeedCounter.is_cell_exit() requires
    speed > 0, so a STOPPED/banked agent never blocks its own promotion - see design_by_contract.md),
    but its genuine next re-attempt at the boundary is denied and penalized again; a non-moving
    STOP_MOVING retry never promotes at all. Only a genuinely valid action (MOVE_LEFT/MOVE_RIGHT)
    lets the agent actually enter the switch. Each MOVING->STOPPED transition that denies a crossing
    draws an INVALID_ACTION penalty (pre-step speed times collision_factor, see
    BaseDefaultRewards.step_reward) - never a COLLISION one, since there is no other agent to
    conflict with - once per genuine entering attempt, not once per retry.
    """
    env, _, _ = env_generator_legacy(seed=43, n_agents=1, rewards=BaseDefaultRewards(collision_factor=COLLISION_FACTOR))

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

    # STOP_MOVING resolves (via apply_action_independent, no direction preference) to the same
    # straight-through look-ahead as MOVE_FORWARD from the pending target (15,15),1 - invalid at
    # this symmetric switch, so this is an invalid action, not an intentional stop: distance banks
    # up to the boundary (same consequence as a resource_check denial), it doesn't stay at its
    # pre-step value.
    _, rewards, _, _ = env.step({agent.handle: RailEnvActions.STOP_MOVING})
    assert agent.current_entry_point[0] == (15, 14)
    assert agent.current_entry_point[1] == 1
    assert agent.state == TrainState.STOPPED
    assert agent.speed_counter.speed == Fraction(0)
    assert agent.speed_counter.distance == Fraction(1, 1)
    assert agent.current_entry_point == ((15, 14), 1)
    assert agent.next_entry_point == ((15, 15), 1)
    # design: this is the entering attempt itself - going into this step the agent was still MOVING,
    # not yet banked (distance 1/2), and is_cell_exit becomes true only this step as it reaches the
    # boundary. That is exactly "the agent would enter the symmetric switch with an action other than
    # L/R" - charged the full collision penalty (pre-step speed 1/2 times collision_factor) under
    # INVALID_ACTION, not COLLISION (no other agent involved).
    assert rewards[agent.handle][DefaultPenalties.INVALID_ACTION.value] == -1 * Fraction(1, 2) * COLLISION_FACTOR
    assert rewards[agent.handle][DefaultPenalties.COLLISION.value] == 0

    # design: distance is already banked at the boundary (SEGMENT_LENGTH), so is_cell_exit() reads
    # False at speed 0 (see design_by_contract.md) - a STOPPED agent given a moving action is always
    # optimistically promoted back to MOVING, regardless of whether that action (MOVE_FORWARD's
    # straight-through look-ahead) is itself structurally valid here. Distance stays banked at the
    # boundary, unchanged (pre_speed was still 0 going into this step). This promotion step is not a
    # MOVING->STOPPED transition, so no penalty is charged for it.
    _, rewards, _, _ = env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.current_entry_point == ((15, 14), 1)
    assert agent.next_entry_point == ((15, 15), 1)
    assert agent.state == TrainState.MOVING
    assert agent.speed_counter.speed == Fraction(1, 2)
    assert agent.speed_counter.distance == Fraction(1, 1)
    assert rewards[agent.handle][DefaultPenalties.INVALID_ACTION.value] == 0
    assert rewards[agent.handle][DefaultPenalties.COLLISION.value] == 0

    # retrying the same invalid action now genuinely re-attempts the crossing (pre_speed > 0 going
    # into this step, so is_cell_exit is true again) - denied again, forced back to STOPPED, and
    # charged a fresh INVALID_ACTION penalty: this is a new entering attempt, not a retry of the
    # previous one.
    _, rewards, _, _ = env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.current_entry_point == ((15, 14), 1)
    assert agent.next_entry_point == ((15, 15), 1)
    assert agent.state == TrainState.STOPPED
    assert agent.speed_counter.speed == Fraction(0)
    assert agent.speed_counter.distance == Fraction(1, 1)
    assert rewards[agent.handle][DefaultPenalties.INVALID_ACTION.value] == -1 * Fraction(1, 2) * COLLISION_FACTOR
    assert rewards[agent.handle][DefaultPenalties.COLLISION.value] == 0

    # only a genuinely valid action (MOVE_LEFT/MOVE_RIGHT) lets the agent actually enter the switch.
    # design: stopped->moving with pre-speed 0, travelling no distance. An optimistic STOPPED->MOVING
    # resumption, not a forced stop - no penalty.
    _, rewards, _, _ = env.step({agent.handle: RailEnvActions.MOVE_LEFT})
    assert agent.current_entry_point == ((15, 14), 1)
    assert agent.next_entry_point == ((15, 15), 1)
    assert agent.state == TrainState.MOVING
    assert agent.speed_counter.speed == Fraction(1, 2)
    assert agent.speed_counter.distance == Fraction(1, 1)
    assert rewards[agent.handle][DefaultPenalties.INVALID_ACTION.value] == 0
    assert rewards[agent.handle][DefaultPenalties.COLLISION.value] == 0

    # now genuinely moving (pre-step speed > 0) - this step actually completes the crossing, no denial.
    _, rewards, _, _ = env.step({agent.handle: RailEnvActions.MOVE_LEFT})
    assert agent.current_entry_point == ((15, 15), 1)
    assert agent.next_entry_point == ((14, 15), 0)
    assert agent.state == TrainState.MOVING
    assert agent.speed_counter.speed == Fraction(1, 2)
    assert agent.speed_counter.distance == Fraction(1, 2)
    assert rewards[agent.handle][DefaultPenalties.INVALID_ACTION.value] == 0
    assert rewards[agent.handle][DefaultPenalties.COLLISION.value] == 0


def _assert_speed_distance_match_candidates(env, agent, action_dict):
    """Capture `agent`'s pre-step values, step `env` with `action_dict`, then assert the real post-step
    agent.speed_counter.speed/.distance exactly match env._candidate_speed()/env._candidate_distance()
    (or, when resource_check denies the candidate, the discarded-candidate fallback formulas) computed
    from those captured pre-step values - the same cross-check RailEnv.step()'s own
    _check_post_speed_distance_speedup_invariants performs internally after every step, done here
    explicitly against the candidate_ methods themselves. Returns the real resource_check outcome and
    this step's rewards dict for `agent`, for the caller's own assertions on top.
    """
    action = RailEnvActions.from_value(action_dict.get(agent.handle, RailEnvActions.DO_NOTHING))
    pre_speed = agent.speed_counter.speed
    pre_offset = agent.speed_counter.distance
    pre_done = agent.target_entry_point is not None
    in_malfunction = agent.malfunction_handler.in_malfunction
    pre_current_entry_point = agent.current_entry_point
    pre_next_entry_point = agent.next_entry_point

    candidate_entry_point_independent = env.rail.apply_action_independent(
        action, pre_next_entry_point if pre_next_entry_point is not None else agent.initial_entry_point)
    candidate_entry_point, candidate_next_entry_point = env._candidate_entry_points(
        action=action, initial_entry_point=agent.initial_entry_point, pre_current_entry_point=pre_current_entry_point,
        pre_next_entry_point=pre_next_entry_point, pre_speed=pre_speed, pre_offset=pre_offset,
        pre_done=pre_done, in_malfunction=in_malfunction, elapsed_steps=env._elapsed_steps + 1,
        candidate_entry_point_independent=candidate_entry_point_independent,
        earliest_departure=agent.earliest_departure, agent_targets=frozenset(agent.targets),
    )

    _, rewards, _, _ = env.step(action_dict)

    resource_check = env.temp_transition_data[agent.handle].resource_check
    if not resource_check:
        expected_speed = None if pre_current_entry_point is None else Fraction(0)
        expected_distance = SpeedCounter.distance_without_crossing(pre_offset, pre_speed)
    else:
        expected_distance = env._candidate_distance(
            pre_speed=pre_speed,
            pre_offset=pre_offset,
            pre_current_entry_point=pre_current_entry_point,
            pre_next_entry_point=pre_next_entry_point,
            pre_done=pre_done,
            candidate_entry_point=candidate_entry_point,
            in_malfunction=in_malfunction,
            candidate_entry_point_independent=candidate_entry_point_independent,
            agent_targets=frozenset(agent.targets),
            remove_agents_at_target=env.remove_agents_at_target,
        )
        if env.remove_agents_at_target and (pre_done or candidate_entry_point in agent.targets):
            expected_speed = None
        elif candidate_entry_point is None:
            expected_speed = None
        else:
            expected_speed = env._candidate_speed(
                pre_speed=pre_speed, pre_offset=pre_offset, action=action,
                pre_current_entry_point=pre_current_entry_point, pre_next_entry_point=pre_next_entry_point,
                pre_done=pre_done,
                candidate_entry_point=candidate_entry_point, in_malfunction=in_malfunction,
                candidate_entry_point_independent=candidate_entry_point_independent,
                agent_targets=frozenset(agent.targets),
                agent_max_speed=agent.speed_counter.max_speed,
                acceleration_delta=env.acceleration_delta,
                braking_delta=env.braking_delta,
            )
    assert agent.speed_counter.speed == expected_speed, (agent.speed_counter.speed, expected_speed)
    assert agent.speed_counter.distance == expected_distance, (agent.speed_counter.distance, expected_distance)
    return resource_check, rewards[agent.handle]


def test_candidate_speed_and_distance_match_genuine_crossing():
    """Single MOVING agent on L=(3,8) of make_simple_rail's row-3 corridor, at max_speed=1, halfway
    across L (distance 0.5) - MOVE_FORWARD completes the crossing into R this step (no other agent to
    contest it, resource_check trivially granted): the real post-step speed/distance exactly match
    env._candidate_speed()/env._candidate_distance() computed from the pre-step values captured just
    before the step.
    """
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1, random_seed=1,
                  rewards=BaseDefaultRewards(collision_factor=COLLISION_FACTOR))
    env.reset()
    env.acceleration_delta = Fraction(1)
    agent = env.agents[0]
    L = ((3, 8), Grid4TransitionsEnum.WEST)
    agent.current_entry_point = L
    agent._set_state(TrainState.MOVING)
    agent.next_entry_point = _sanitize_entry_point(env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, L))
    agent.speed_counter = SpeedCounter(max_speed=Fraction(1), speed=Fraction(1))
    agent.speed_counter.set(speed=Fraction(1), distance=Fraction(1, 2))

    resource_check, rewards = _assert_speed_distance_match_candidates(env, agent, {0: RailEnvActions.MOVE_FORWARD})
    assert resource_check
    assert agent.current_entry_point != L  # the crossing genuinely completed
    # a genuine, granted crossing is not a forced stop - no collision/invalid-action penalty
    assert rewards[DefaultPenalties.COLLISION.value] == 0
    assert rewards[DefaultPenalties.INVALID_ACTION.value] == 0


def test_candidate_speed_and_distance_match_invalid_action_denial_at_cell_exit():
    """An invalid STOP_MOVING at a symmetric switch (no straight-through option, see
    test_symmetric_switch_stop_action) denies the crossing at the cell boundary - the real post-step
    speed/distance exactly match env._candidate_speed()/env._candidate_distance() computed from the
    pre-step values captured just before the step: speed forced to 0, distance banked at the boundary,
    not credited with the (denied) crossing. The forced MOVING->STOPPED transition draws an
    INVALID_ACTION penalty (pre-step speed 1/2 times collision_factor), not COLLISION - there is no
    other agent, the denial is purely the invalid action.
    """
    env, _, _ = env_generator_legacy(seed=43, n_agents=1, rewards=BaseDefaultRewards(collision_factor=COLLISION_FACTOR))
    assert env.rail.get_full_transitions(15, 15) == RailEnvTransitionsEnum.symmetric_switch_from_west

    agent = env.agents[0]
    agent.current_entry_point = ((15, 14), 1)
    agent._set_state(TrainState.MOVING)
    agent.next_entry_point = _sanitize_entry_point(
        env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, agent.current_entry_point))
    agent.speed_counter = SpeedCounter(max_speed=Fraction(1, 2), speed=Fraction(1, 2))
    agent.speed_counter.set(speed=Fraction(1, 2), distance=Fraction(1))  # banked exactly at the boundary

    resource_check, rewards = _assert_speed_distance_match_candidates(
        env, agent, {agent.handle: RailEnvActions.STOP_MOVING})
    # denial here is via the invalid action, not a resource conflict - self-loop, trivially granted
    assert resource_check
    assert agent.current_entry_point == ((15, 14), 1)  # crossing denied - position unchanged
    assert agent.speed_counter.speed == Fraction(0)
    assert agent.speed_counter.distance == Fraction(1)
    assert rewards[DefaultPenalties.INVALID_ACTION.value] == -1 * Fraction(1, 2) * COLLISION_FACTOR
    assert rewards[DefaultPenalties.COLLISION.value] == 0


def test_candidate_speed_and_distance_match_resource_check_denial():
    """Agent A on L=(3,8) of make_simple_rail's row-3 corridor, agent B parked at rest on the
    neighboring cell R=(3,7) directly ahead of A - A tries to cross into R this step and is denied
    (resource_check False, B still there): the real post-step speed/distance exactly match the
    discarded-candidate fallback formulas (speed forced to 0, distance banked at the boundary) computed
    from the pre-step values captured just before the step. The forced MOVING->STOPPED transition
    draws a COLLISION penalty (pre-step speed 1 times collision_factor), not INVALID_ACTION - A's
    action itself is valid, the denial is purely B's motion-check conflict.
    """
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=2, random_seed=1,
                  rewards=BaseDefaultRewards(collision_factor=COLLISION_FACTOR))
    env.reset()
    env.acceleration_delta = Fraction(1)
    agent_a, agent_b = env.agents[0], env.agents[1]
    L = ((3, 8), Grid4TransitionsEnum.WEST)
    R = ((3, 7), Grid4TransitionsEnum.WEST)

    agent_a.current_entry_point = L
    agent_a._set_state(TrainState.MOVING)
    agent_a.next_entry_point = _sanitize_entry_point(env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, L))
    agent_a.speed_counter = SpeedCounter(max_speed=Fraction(1), speed=Fraction(1))
    agent_a.speed_counter.set(speed=Fraction(1), distance=Fraction(1, 2))

    agent_b.current_entry_point = R
    agent_b._set_state(TrainState.STOPPED)
    agent_b.next_entry_point = _sanitize_entry_point(env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, R))
    agent_b.speed_counter = SpeedCounter(max_speed=Fraction(1), speed=Fraction(0))
    agent_b.speed_counter.set(speed=Fraction(0), distance=Fraction(1, 2))

    resource_check, rewards = _assert_speed_distance_match_candidates(
        env, agent_a, {0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.DO_NOTHING})
    assert not resource_check
    assert agent_a.current_entry_point == L  # denied - position unchanged
    assert agent_a.speed_counter.speed == Fraction(0)
    assert agent_a.speed_counter.distance == Fraction(1)
    assert rewards[DefaultPenalties.COLLISION.value] == -1 * Fraction(1) * COLLISION_FACTOR
    assert rewards[DefaultPenalties.INVALID_ACTION.value] == 0


def test_pre_done_candidate_entry_point_independent_is_stale_not_a_live_reservation():
    """
    Documents that a removed agent never actually holds its initial cell as a reservation another agent
    can be blocked by.

    Rail: `_make_straight_rail(3)`'s 3-cell corridor A=(0,0) - B=(0,1) - C=(0,2). Agent 0's target is B;
    B is also agent 1's own initial cell (agent 1's line runs B -> C).

    - Setup: agent 0 bootstrapped directly onto A, MOVING at max_speed=1 with distance already at the
      cell boundary (crosses into B in one step). Agent 1 stays off map, READY_TO_DEPART with
      earliest_departure=0 and initial_entry_point B.
    - Step 0: both agents contest resource B in the same step - agent 0 crossing A->B, agent 1 departing
      directly onto B - so agent 0 genuinely occupies B (agent 1's own initial cell) as this step is
      resolved. MotionCheck's same-target tie-break favors the lower handle: agent 0 wins, reaches its
      target B, and (remove_agents_at_target) is immediately DONE with current_entry_point already None
      by the end of this step. Agent 1 loses the conflict - motion_check.stopped names it as blocked,
      it stays off map, still READY_TO_DEPART, and (collision_factor set > 0 here specifically to make
      this checkable) draws no collision/invalid-action penalty for the denial.
    - Step 1: agent 1 retries the identical departure action - granted this time with no denial, even
      though the pre-step snapshot still computes a transition from agent 0's (now DONE) initial cell A:
      that lookup is never consulted for a DONE agent's own resource, so it leaves B genuinely free -
      agent 1 ends up on exactly its own initial cell.
    """
    rail, optionals = _make_straight_rail(3)
    env = RailEnv(width=3, height=1, rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=2,
                  obs_builder_object=GlobalObsForRailEnv(),
                  rewards=BaseDefaultRewards(collision_factor=1.0))
    env.reset()
    env._max_episode_steps = 1000
    _place_agent_on_map(env, 0, (0, 0), Grid4TransitionsEnum.WEST, (0, 1), TrainState.MOVING,
                        Fraction(1), Fraction(1), RailEnvActions.MOVE_FORWARD)
    agent0, agent1 = env.agents[0], env.agents[1]

    agent1.initial_entry_point = ((0, 1), Grid4TransitionsEnum.EAST)
    agent1.targets = {((0, 2), d) for d in Grid4TransitionsEnum}
    agent1.earliest_departure = 0
    agent1._set_state(TrainState.READY_TO_DEPART)

    # agent 0 is about to cross into B - agent 1's own initial cell - this very step
    assert agent0.next_entry_point[0] == agent1.initial_entry_point[0]

    _, rewards_dict, _, _ = env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})
    # agent 0 genuinely occupied B (agent 1's initial cell) while this conflicting step was resolved,
    # even though it is DONE and removed (current_entry_point back to None) by the time step() returns
    assert env.temp_transition_data[agent0.handle].candidate_entry_point[0] == agent1.initial_entry_point[0]
    assert agent0.state == TrainState.DONE
    assert agent0.current_entry_point is None  # removed - B genuinely free now
    assert agent1.state == TrainState.READY_TO_DEPART  # denied - lower-handle agent 0 won the conflict
    assert agent1.current_entry_point is None
    assert agent1.handle in env.resource_check.stopped  # motion_check itself names agent 1 as blocked
    assert agent0.handle not in env.resource_check.stopped  # agent 0 was not blocked - it won the conflict
    # denial by a resource conflict is not penalized as a collision/invalid action for agent 1, despite
    # collision_factor > 0 - BaseDefaultRewards.step_reward only penalizes a MOVING->STOPPED transition,
    # and agent 1 never left READY_TO_DEPART
    assert rewards_dict[agent1.handle][DefaultPenalties.COLLISION.value] == 0
    assert rewards_dict[agent1.handle][DefaultPenalties.INVALID_ACTION.value] == 0

    env.step({1: RailEnvActions.MOVE_FORWARD})
    assert agent1.current_entry_point == agent1.initial_entry_point  # entered unhindered, on its own initial cell
    assert agent1.state == TrainState.MOVING
    assert agent0.state == TrainState.DONE  # still done - stays terminal
    assert agent0.current_entry_point is None
    assert agent0.next_entry_point is None


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
    target (3, 5) instead (a plain straight cell, so MOVE_LEFT there just corrects to (3, 4)). Each
    of agent 0's env-forced MOVING->STOPPED transitions while blocked draws a COLLISION penalty
    (pre-step speed 1 times collision_factor, agent 0's max_speed here) - never INVALID_ACTION, since
    MOVE_FORWARD/MOVE_LEFT are both structurally valid actions, the denial is purely agent 1's
    motion-check conflict - and only on the transition steps, not on the optimistic STOPPED->MOVING
    resumption steps in between.
    """
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=2,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  random_seed=1, rewards=BaseDefaultRewards(collision_factor=COLLISION_FACTOR))
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
    _, rewards, _, _ = env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.STOP_MOVING})
    assert agent0.current_entry_point == ((3, 6), 3)
    assert agent0.next_entry_point == ((3, 5), 3)
    assert agent0.state == TrainState.STOPPED
    assert rewards[0][DefaultPenalties.COLLISION.value] == -1 * Fraction(1) * COLLISION_FACTOR
    assert rewards[0][DefaultPenalties.INVALID_ACTION.value] == 0

    # giving MOVE_LEFT while blocked does NOT redirect the pending target onto the southward
    # branch at (4, 6) - it stays locked onto (3, 5), retried for as long as agent 1 blocks it.
    # design (D1/D2): a STOPPED agent given a movement action is optimistically promoted to MOVING
    # on the operator's request (see rail_env.py's (3b.2bis)/movement_allowed design note) even
    # though its target is still occupied - position/distance stay unchanged either way (D1). The
    # *following* step then genuinely attempts the crossing for real, is denied again by MotionCheck
    # (agent 1 still parked at (3, 5)), and the state machine demotes back to STOPPED - so the state
    # alternates MOVING/STOPPED every retry for as long as agent 1 blocks it, never actually moving.
    # The optimistic STOPPED->MOVING resumption is free (nothing re-contested yet); each genuine
    # re-attempt (MOVING->STOPPED) is charged the same full collision penalty again.
    for expected_state, expected_collision in [
        (TrainState.MOVING, 0),
        (TrainState.STOPPED, -1 * Fraction(1) * COLLISION_FACTOR),
        (TrainState.MOVING, 0),
        (TrainState.STOPPED, -1 * Fraction(1) * COLLISION_FACTOR),
    ]:
        _, rewards, _, _ = env.step({0: RailEnvActions.MOVE_LEFT, 1: RailEnvActions.STOP_MOVING})
        assert agent0.current_entry_point == ((3, 6), 3)
        assert agent0.next_entry_point == ((3, 5), 3)
        assert agent0.state == expected_state
        assert rewards[0][DefaultPenalties.COLLISION.value] == expected_collision
        assert rewards[0][DefaultPenalties.INVALID_ACTION.value] == 0

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
    # design (issue #280): earliest_departure_reached is signalled one step earlier than
    # elapsed_steps itself reaches earliest_departure,
    # agent goes to READY_TO_DEPART at the end of the step it reaches earliest_departure
    # (e.g. when earliest_departure==0, agent is READY_TO_DEPART before the first step (where _elapsed_steps 0->1)
    assert env._elapsed_steps == 2  # _elapsed_steps +2 (2 WAITING steps)

    # READY_TO_DEPART: still off map, waiting for a valid MOVE_FORWARD to actually depart.
    assert agent.state == TrainState.READY_TO_DEPART
    assert agent.current_entry_point is None

    # READY_TO_DEPART -> MOVING: agent appears at its initial entry point this very step. Full
    # acceleration delta reaches max speed immediately (acceleration_delta equals max_speed here).
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == 3  # _elapsed_steps +1
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
    assert env._elapsed_steps == 4  # _elapsed_steps +1
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == second_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(0)

    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == 5  # _elapsed_steps +1
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == third_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(0)

    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == 6  # _elapsed_steps +1
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == fourth_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(0)


@pytest.mark.parametrize("earliest_departure", [0, 1, 2, 5])
@pytest.mark.parametrize("with_malfunction", [False, True], ids=["no_malfunction", "malfunction_off_map"])
def test_map_entry(with_malfunction, earliest_departure):
    """
    Single agent, max speed 1, acceleration delta equal to max speed, earliest_departure parametrized
    over 0/1/2/5. Parametrized over whether a malfunction is injected right as earliest_departure is
    reached. Design (issue #280): both WAITING and MALFUNCTION_OFF_MAP go straight to MOVING on a
    movement action once earliest_departure is reached, never observably visiting READY_TO_DEPART
    first - see _handle_malfunction_off_map's docstring and rail_env.py's step() first-step
    earliest_departure=0 tweak. Apart from the malfunction variant's extra delay, the two variants'
    assertions are identical from the departure step onward.

    - Setup: agent off map, WAITING, speed and distance both None. Stepped earliest_departure times
      with DO_NOTHING first - this withholds the movement action so the agent cannot depart yet, but
      still lets the earliest_departure-driven WAITING -> READY_TO_DEPART transition happen on
      schedule, landing the agent exactly at the point earliest_departure is considered reached: for
      earliest_departure == 0 this is zero steps (still WAITING, _elapsed_steps == 0); for
      earliest_departure >= 1 the agent is already READY_TO_DEPART after those steps, at
      _elapsed_steps == earliest_departure (the earliest_departure_reached signal fires one step ahead
      of when it becomes externally visible, so earliest_departure 1 and 2 both resolve to
      READY_TO_DEPART already by _elapsed_steps == 1 - see state_machine.py's step() docstring).
    - With malfunction: malfunction_down_counter is set to 3 right at this point (malfunction_interval=0
      disables env_generator's own random malfunction generation, so only this injected malfunction is
      in play) - i.e. triggered exactly as earliest_departure is reached, not before. The state machine
      decrements the counter on the same step it reads in_malfunction, so it is observed True for 2
      steps and reaches 0 on the 3rd, from whichever state (WAITING or READY_TO_DEPART) the agent was
      in at injection time.
    - [malfunction only] First MOVE_FORWARD step after injection (in_malfunction after this step's
      decrement): WAITING or READY_TO_DEPART -> MALFUNCTION_OFF_MAP, still off map,
      malfunction_down_counter == 2.
    - [malfunction only] Second MOVE_FORWARD step (still in_malfunction): stays MALFUNCTION_OFF_MAP,
      still off map, malfunction_down_counter == 1.
    - Departure step (MOVE_FORWARD; the 1st post-injection step without malfunction, the 3rd with): map
      entry directly into MOVING - position becomes the agent's initial entry point, distance resets to
      0, speed reaches max speed immediately (with malfunction: malfunction_down_counter == 0). This
      lands at _elapsed_steps == earliest_departure + 1 without malfunction, or
      earliest_departure + 3 with malfunction (the 3-step malfunction length contributing 2 extra steps
      beyond the no-malfunction departure step).
    - Final step (MOVE_FORWARD): already at max speed, so the agent crosses into the next entry
      point, distance wraps back to 0 completing the crossing, speed stays at max speed.
    """
    env, _, _ = env_generator(seed=42, n_agents=1, malfunction_interval=0)
    env.acceleration_delta = Fraction(1)
    agent = env.agents[0]
    agent.speed_counter = SpeedCounter(max_speed=Fraction(1))
    agent.earliest_departure = earliest_departure

    assert agent.state == TrainState.WAITING
    assert agent.current_entry_point is None
    assert agent.speed_counter.speed is None
    assert agent.speed_counter.distance is None

    for _ in range(earliest_departure):
        env.step({agent.handle: RailEnvActions.DO_NOTHING})
    assert env._elapsed_steps == earliest_departure  # N
    assert agent.state == (TrainState.WAITING if earliest_departure == 0 else TrainState.READY_TO_DEPART)
    assert agent.current_entry_point is None
    assert agent.speed_counter.speed is None
    assert agent.speed_counter.distance is None

    MALFUNCTION_DURATION = 0
    if with_malfunction:
        MALFUNCTION_DURATION = 2  # M
        agent.malfunction_handler._set_malfunction_down_counter(MALFUNCTION_DURATION + 1)  # as decrement in step after malfunction generator runs!

        env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
        assert env._elapsed_steps == earliest_departure + 1
        assert agent.state == TrainState.MALFUNCTION_OFF_MAP
        assert agent.current_entry_point is None
        assert agent.speed_counter.speed is None
        assert agent.speed_counter.distance is None
        assert agent.malfunction_handler.malfunction_down_counter == 2

        env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
        assert env._elapsed_steps == earliest_departure + 2
        assert agent.state == TrainState.MALFUNCTION_OFF_MAP
        assert agent.current_entry_point is None
        assert agent.speed_counter.speed is None
        assert agent.speed_counter.distance is None
        assert agent.malfunction_handler.malfunction_down_counter == 1

    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == earliest_departure + 1 + MALFUNCTION_DURATION  # w/o malfunction: N+1 / w malfunction (N+1) + M
    assert agent.state == TrainState.MOVING
    first_entry_point = agent.current_entry_point
    assert first_entry_point == agent.initial_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(0)
    if with_malfunction:
        assert agent.malfunction_handler.malfunction_down_counter == 0

    second_entry_point = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, first_entry_point)
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == earliest_departure + (4 if with_malfunction else 2)
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == second_entry_point
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
    # design (issue #280): earliest_departure_reached is signalled one step earlier than
    # elapsed_steps itself reaches earliest_departure,
    # agent goes to READY_TO_DEPART at the end of the step it reaches earliest_departure
    # (e.g. when earliest_departure==0, agent is READY_TO_DEPART before the first step (where _elapsed_steps 0->1)
    assert env._elapsed_steps == 2  # _elapsed_steps +2 (2 WAITING steps)

    # READY_TO_DEPART: still off map, waiting for a valid MOVE_FORWARD to actually depart.
    assert agent.state == TrainState.READY_TO_DEPART
    assert agent.current_entry_point is None

    # READY_TO_DEPART -> MOVING: agent appears at its initial entry point this very step, distance
    # resets to 0 and speed reaches the acceleration delta immediately.
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == 3  # _elapsed_steps +1
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
    assert env._elapsed_steps == 4  # _elapsed_steps +1
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == first_entry_point
    assert agent.speed_counter.speed == Fraction(6, 10)
    assert agent.speed_counter.distance == Fraction(3, 10)

    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == 5  # _elapsed_steps +1
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == first_entry_point
    assert agent.speed_counter.speed == Fraction(9, 10)
    assert agent.speed_counter.distance == Fraction(9, 10)

    # distance(0.9) + speed(0.9) = 1.8 >= 1: crosses into the second entry point, wraps to 0.8, and
    # speed finally saturates at max speed (0.9 + 0.3 capped at 1).
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == 6  # _elapsed_steps +1
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == second_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(8, 10)

    # At max speed, the agent advances exactly one entry point per step from here on, cruising at
    # a steady 0.8 offset (0.8 + 1 = 1.8, wraps to 0.8 again).
    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == 7  # _elapsed_steps +1
    assert agent.state == TrainState.MOVING
    assert agent.current_entry_point == third_entry_point
    assert agent.speed_counter.speed == Fraction(1)
    assert agent.speed_counter.distance == Fraction(8, 10)

    env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert env._elapsed_steps == 8  # _elapsed_steps +1
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


def test_action_required_false_during_malfunction():
    """
    Single agent, max speed 1/2, acceleration delta equal to max speed, earliest_departure=0. Design:
    info['action_required'] (RailEnv.action_required()) is False for a malfunctioning agent, both off
    map (MALFUNCTION_OFF_MAP) and on map (MALFUNCTION) - contrasted here against the same agent's
    action_required alternating True/False while genuinely MOVING, to show the malfunction periods
    aren't just incidentally False for some unrelated reason (e.g. always being mid-cell).

    - Setup: malfunction_down_counter=3 injected before the very first step - the agent malfunctions
      off map before it ever gets a chance to depart.
    - Steps 1-2 (MOVE_FORWARD, in_malfunction): MALFUNCTION_OFF_MAP, still off map, action_required False.
    - Step 3 (malfunction clears): dispatches straight into MOVING at distance 0 - at half max speed,
      distance 0 does not yet reach the cell boundary, so action_required is False here too (not yet
      due to malfunction - genuinely not at a cell exit).
    - Step 4 (MOVE_FORWARD): distance reaches 1/2, the cell boundary - action_required True, contrasting
      with the malfunction periods above and below.
    - A second malfunction (down_counter=3) is injected right at this boundary (distance 1/2).
    - Steps 5-6 (MOVE_FORWARD, in_malfunction): MALFUNCTION, distance frozen at 1/2 with speed forced to
      0 - is_cell_exit() no longer holds (1/2 + 0 < 1), so action_required is False throughout, even
      though the agent was sitting exactly at the boundary the instant the malfunction hit.
    - Step 7 (malfunction clears): resumes MOVING from the frozen distance 1/2 at half speed again -
      distance + speed reaches the boundary again this same step, action_required True.
    - Step 8 (MOVE_FORWARD): crosses into the next cell, distance wraps to 0, action_required False again.
    """
    env, _, _ = env_generator(seed=42, n_agents=1, malfunction_interval=0)
    env.acceleration_delta = Fraction(1)
    agent = env.agents[0]
    agent.speed_counter = SpeedCounter(max_speed=Fraction(1, 2))
    agent.earliest_departure = 0
    agent.malfunction_handler._set_malfunction_down_counter(3)

    _, _, _, info = env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MALFUNCTION_OFF_MAP
    assert info['action_required'][agent.handle] is False

    _, _, _, info = env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MALFUNCTION_OFF_MAP
    assert info['action_required'][agent.handle] is False

    _, _, _, info = env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.speed_counter.distance == Fraction(0)
    assert agent.speed_counter.is_cell_exit() is False
    assert info['action_required'][agent.handle] is False

    _, _, _, info = env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.speed_counter.distance == Fraction(1, 2)
    assert agent.speed_counter.is_cell_exit() is True
    assert info['action_required'][agent.handle] is True

    agent.malfunction_handler._set_malfunction_down_counter(3)

    _, _, _, info = env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MALFUNCTION
    assert agent.speed_counter.distance == Fraction(1, 2)
    assert agent.speed_counter.speed == 0
    assert agent.speed_counter.is_cell_exit() is False
    assert info['action_required'][agent.handle] is False

    _, _, _, info = env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MALFUNCTION
    assert agent.speed_counter.is_cell_exit() is False
    assert info['action_required'][agent.handle] is False

    _, _, _, info = env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.speed_counter.distance == Fraction(1, 2)
    assert agent.speed_counter.is_cell_exit() is True
    assert info['action_required'][agent.handle] is True

    _, _, _, info = env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.speed_counter.distance == Fraction(0)
    assert info['action_required'][agent.handle] is False


@pytest.mark.parametrize("with_malfunction", [False, True], ids=["stopped_only", "malfunction_while_stopped"])
def test_action_required_at_full_segment_length(with_malfunction):
    """
    Single agent departs at (2,3) heading WEST towards a symmetric switch at (2,1) whose only valid
    exits are north/south - continuing MOVE_FORWARD (straight through) is invalid there, so the agent
    is force-stopped one cell short, banked exactly at the (2,2) cell boundary: distance ==
    SEGMENT_LENGTH (1), speed == 0. Design: for an on-map state, info['action_required'] depends purely
    on SpeedCounter.is_cell_exit() (speed > 0 and distance + speed >= SEGMENT_LENGTH - see
    design_by_contract.md), not on which on-map state (MOVING/STOPPED/MALFUNCTION) the agent is
    actually in - is_cell_exit() (and so action_required) reads False for any agent parked at speed 0,
    even one banked exactly at the boundary by distance alone, whether or not it also happens to be
    malfunctioning at that same banked position.

    Layout (train departs at (2,3) heading WEST, rolls towards the symmetric switch at (2,1) whose only
    valid exits are north and south -- continuing west is invalid):

        (0,1) dead-end
              |
        (2,1) switch <- (2,2) <- (2,3) start
              |
        (4,1) dead-end

    - Steps 1-2 (MOVE_FORWARD): departs and crosses one full cell at max speed - MOVING, is_cell_exit
      True every step (a fresh cell's full speed always reaches the boundary the same step),
      action_required True.
    - Step 3 (MOVE_FORWARD, invalid at the symmetric switch): force-stopped at the (2,2) boundary -
      STOPPED, distance == 1, speed == 0, is_cell_exit False (speed == 0 overrides distance alone),
      action_required False.
    - [malfunction_while_stopped only] malfunction_down_counter=3 injected right at this banked
      position; steps 4-5 (still MOVE_FORWARD, in_malfunction): MALFUNCTION, distance unchanged,
      speed still 0, is_cell_exit and action_required both remain False.
    - Final step (MOVE_FORWARD): a moving action given to a STOPPED (or recovering-MALFUNCTION) agent
      is always optimistically promoted back to MOVING - is_cell_exit() reading False at speed 0 is
      what lets this promotion happen even though MOVE_FORWARD is still structurally invalid at this
      switch (see test_symmetric_switch_move_forward_action). Both variants end up MOVING here, speed
      > 0 again, distance still pinned at the boundary (the promotion itself travels no distance), so
      is_cell_exit and action_required both read True again - a further genuine re-attempt on the next
      step would be denied and force-stopped once more.
    """
    transitions = RailEnvTransitions()
    grid = np.zeros((5, 6), dtype=np.uint16)
    grid[2, 1] = RailEnvTransitionsEnum.symmetric_switch_from_east  # heading west forks N/S
    grid[2, 2] = RailEnvTransitionsEnum.horizontal_straight
    grid[2, 3] = RailEnvTransitionsEnum.horizontal_straight
    grid[2, 4] = RailEnvTransitionsEnum.dead_end_from_east
    grid[1, 1] = RailEnvTransitionsEnum.vertical_straight
    grid[0, 1] = RailEnvTransitionsEnum.dead_end_from_north
    grid[3, 1] = RailEnvTransitionsEnum.vertical_straight
    grid[4, 1] = RailEnvTransitionsEnum.dead_end_from_south

    rail = RailGridTransitionMap(width=6, height=5, transitions=transitions)
    rail.grid = grid
    optionals = {'agents_hints': {'city_positions': [(2, 4), (0, 1)],
                                  'train_stations': [[((2, 4), 0)], [((0, 1), 0)]],
                                  'city_orientations': [3, 0]}}
    env = RailEnv(width=6, height=5,
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1)
    env.reset(random_seed=42)
    env._max_episode_steps = 100

    agent = env.agents[0]
    agent.initial_entry_point = ((2, 3), Grid4TransitionsEnum.WEST)
    agent.current_entry_point = None
    agent.earliest_departure = 0
    agent.latest_arrival = 50
    agent.targets = {((0, 1), d) for d in Grid4TransitionsEnum}

    for _ in range(2):
        _, _, _, info = env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
        assert agent.state == TrainState.MOVING
        assert agent.speed_counter.is_cell_exit() is True
        assert info['action_required'][agent.handle] is True

    _, _, _, info = env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.STOPPED
    assert agent.speed_counter.distance == Fraction(1)
    assert agent.speed_counter.speed == 0
    assert agent.speed_counter.is_cell_exit() is False
    assert info['action_required'][agent.handle] is False

    if with_malfunction:
        agent.malfunction_handler._set_malfunction_down_counter(3)
        for _ in range(2):
            _, _, _, info = env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
            assert agent.state == TrainState.MALFUNCTION
            assert agent.speed_counter.distance == Fraction(1)
            assert agent.speed_counter.is_cell_exit() is False
            assert info['action_required'][agent.handle] is False

    _, _, _, info = env.step({agent.handle: RailEnvActions.MOVE_FORWARD})
    assert agent.state == TrainState.MOVING
    assert agent.speed_counter.distance == Fraction(1)
    assert agent.speed_counter.is_cell_exit() is True
    assert info['action_required'][agent.handle] is True


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


COLLISION_FACTOR = 250.0


def test_agent_blocked_at_boundary_cannot_accelerate_nor_advance_into_stopped_neighbor():
    """
    Agent A on L=(3,8), agent B on R=(3,7) - adjacent cells of make_simple_rail's row-3 corridor, both
    heading west, max speed 1.

    - Setup: A departs onto L, B departs onto R, both at distance 0, speed 1. B is then braked to a
      stop before it can leave R, parking it there at distance 1/2, speed 0.
    - A reaches the L->R boundary and tries to cross into R: denied, since B still occupies it. A's
      position stays on L, its distance pins at the boundary (1.0), its speed is forced back to 0, and
      the attempt is charged the full collision penalty (pre-attempt speed 1 times the collision
      factor).
    - Retrying MOVE_FORWARD from there: position never leaves L, distance stays pinned at 1.0. A retry
      that only optimistically resumes MOVING (not yet re-contesting the boundary) costs nothing; a
      retry that genuinely re-contests the boundary is denied again and charged the full collision
      penalty again.
    - B's position, speed and distance on R stay untouched throughout.
    """
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=2, random_seed=1,
                  rewards=BaseDefaultRewards(collision_factor=COLLISION_FACTOR))
    env.reset()
    env.acceleration_delta = Fraction(1)
    env.braking_delta = Fraction(-1)
    agent_a, agent_b = env.agents[0], env.agents[1]
    L = ((3, 8), Grid4TransitionsEnum.WEST)
    R = ((3, 7), Grid4TransitionsEnum.WEST)
    agent_a.initial_entry_point = L
    agent_b.initial_entry_point = R
    # design (issue #280): earliest_departure=1, not 0 - an earliest_departure=0 agent now dispatches
    # directly on the very first movement action (see rail_env.py's step()), which would make the "two
    # steps of MOVE_FORWARD to get onto the map" below only one; =1 keeps the original timing.
    agent_a.earliest_departure = 1
    agent_b.earliest_departure = 1
    # B's own max speed is lower than A's (still off map, so only _max_speed needs overriding - see
    # test_blocked_agent_cannot_redirect_via_later_action for why _speed must stay None here): this
    # keeps it mid-cell (distance < 1) rather than already at the cell boundary when it is braked
    # below, so STOP_MOVING genuinely parks it on R instead of letting an already-banked crossing
    # complete first.
    agent_b.speed_counter._max_speed = Fraction(1, 2)

    # WAITING -> READY_TO_DEPART -> MOVING: two steps of MOVE_FORWARD to get both agents onto the map.
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})
    env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD})
    assert agent_a.current_entry_point == L
    assert agent_a.speed_counter.speed == Fraction(1)
    assert agent_a.speed_counter.distance == Fraction(0)
    assert agent_b.current_entry_point == R
    assert agent_b.speed_counter.speed == Fraction(1, 2)
    assert agent_b.speed_counter.distance == Fraction(0)

    # B brakes to a stop on R (still mid-cell, so it doesn't cross first); A, at speed 1, reaches the
    # L->R boundary and tries to cross - denied, since B is still occupying R.
    _, rewards, _, _ = env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.STOP_MOVING})
    assert agent_b.current_entry_point == R
    assert agent_b.speed_counter.speed == Fraction(0)
    assert agent_b.speed_counter.distance == Fraction(1, 2)
    assert agent_b.state == TrainState.STOPPED

    assert agent_a.current_entry_point == L  # denied - did not enter R
    assert agent_a.speed_counter.speed == Fraction(0)  # cannot accelerate to speed > 0 while blocked
    assert agent_a.speed_counter.distance == Fraction(1)  # pinned at the L->R cell boundary
    assert agent_a.state == TrainState.STOPPED
    assert rewards[0][DefaultPenalties.COLLISION.value] == -1 * 1 * COLLISION_FACTOR  # full collision penalty
    # a motion-check conflict is a collision, not an invalid action
    assert rewards[0][DefaultPenalties.INVALID_ACTION.value] == 0

    # Retrying while B stays parked on R never gets A any further: its position never leaves L. Each
    # retry first optimistically resumes MOVING (nothing re-contested yet -> no penalty), then genuinely
    # re-attempts the boundary as STOPPED again (denied again -> full collision penalty again).
    for expected_state, expected_collision in [
        (TrainState.MOVING, 0),
        (TrainState.STOPPED, -1 * 1 * COLLISION_FACTOR),
        (TrainState.MOVING, 0),
        (TrainState.STOPPED, -1 * 1 * COLLISION_FACTOR),
    ]:
        _, rewards, _, _ = env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.DO_NOTHING})
        assert agent_a.state == expected_state
        assert agent_a.current_entry_point == L
        assert agent_a.speed_counter.distance == Fraction(1)
        assert rewards[0][DefaultPenalties.COLLISION.value] == expected_collision
        assert rewards[0][DefaultPenalties.INVALID_ACTION.value] == 0  # always a collision, never an invalid action
        assert agent_b.current_entry_point == R
        assert agent_b.speed_counter.speed == Fraction(0)
        assert agent_b.speed_counter.distance == Fraction(1, 2)


@pytest.mark.parametrize("speed,steps_to_boundary", [
    (Fraction(1, 5), 3),
    (Fraction(1, 2), 1),
    (Fraction(1), 1),
], ids=["speed-0.2-three-steps", "speed-0.5-one-step", "speed-1.0-one-step"])
def test_agent_cruising_at_constant_speed_banks_distance_to_boundary_then_stops(speed, steps_to_boundary):
    """
    Agent A on L=(3,8) of make_simple_rail's row-3 corridor, agent B parked at rest on the neighboring
    cell R=(3,7), directly ahead of A. Parametrized over A's cruising speed: 0.2, 0.5 or 1.0 (its max
    speed equals that speed, so MOVE_FORWARD can never push it faster).

    - Setup: A already halfway across L (distance 0.5), cruising at the given constant speed.
    - Approach: given MOVE_FORWARD every step, A's distance increases by exactly `speed` each step,
      still short of the L->R boundary (1.0) and still at speed `speed` - 2 such steps at speed 0.2,
      none at speed 0.5 or 1.0 (already at or past the boundary on the very first step at those
      speeds). Reaching the boundary takes 3, 1 and 1 steps respectively.
    - Boundary step: A tries to cross into R - denied, since B is still there. Distance clamps at
      exactly 1.0 rather than overshooting into R, position stays on L, speed is forced back to 0, and
      the attempt is charged a collision penalty of speed * collision_factor - the full collision
      penalty only for the 1.0-speed variant, proportionally less for the slower ones (0.2x/0.5x
      collision_factor).
    - Two more retries: MOVE_FORWARD first optimistically resumes MOVING at the same constant speed
      without yet re-contesting the boundary (no penalty), then genuinely re-attempts it as STOPPED
      again (denied again, same speed * collision_factor penalty) - position and distance stay exactly
      as pinned throughout.
    """
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=2, random_seed=1,
                  rewards=BaseDefaultRewards(collision_factor=COLLISION_FACTOR))
    env.reset()
    env.acceleration_delta = Fraction(1)
    agent_a, agent_b = env.agents[0], env.agents[1]
    L = ((3, 8), Grid4TransitionsEnum.WEST)
    R = ((3, 7), Grid4TransitionsEnum.WEST)

    # place A directly on L, already MOVING at a constant `speed` (max_speed == speed caps it there)
    # with distance already banked halfway to R - bypassing the WAITING/READY_TO_DEPART/acceleration
    # ramp-up covered by test_earliest_departure_state_transitions_* to start from this state directly.
    agent_a.current_entry_point = L
    agent_a._set_state(TrainState.MOVING)
    agent_a.next_entry_point = _sanitize_entry_point(env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, L))
    agent_a.speed_counter = SpeedCounter(max_speed=speed, speed=speed)
    agent_a.speed_counter.set(speed=speed, distance=Fraction(0))  # bootstrap onto the map at distance 0
    agent_a.speed_counter._distance = Fraction(1, 2)

    # place B directly on R, at rest - a stopped, blocking neighbor.
    agent_b.current_entry_point = R
    agent_b._set_state(TrainState.STOPPED)
    agent_b.next_entry_point = _sanitize_entry_point(env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, R))
    agent_b.speed_counter = SpeedCounter(max_speed=Fraction(1), speed=Fraction(0))
    agent_b.speed_counter.set(speed=Fraction(0), distance=Fraction(0))
    agent_b.speed_counter._distance = Fraction(1, 2)

    for step in range(1, steps_to_boundary):
        _, rewards, _, _ = env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.DO_NOTHING})
        assert agent_a.state == TrainState.MOVING
        assert agent_a.current_entry_point == L
        assert agent_a.speed_counter.speed == speed
        assert agent_a.speed_counter.distance == Fraction(1, 2) + step * speed
        assert rewards[0][DefaultPenalties.COLLISION.value] == 0  # not at the boundary yet - no attempt
        assert rewards[0][DefaultPenalties.INVALID_ACTION.value] == 0

    # the step that reaches (or would overshoot past) the boundary: denied entry into R since B is
    # still there - distance clamps at the boundary instead of overshooting, position stays on L, and
    # speed is forced back to 0. Charged a collision penalty of speed * collision_factor - full only
    # for the speed-1.0 variant, proportionally less for the slower ones. MOVE_FORWARD is a valid
    # action (B is a motion-check conflict, not an invalid transition), so this is always a collision,
    # never an invalid action.
    _, rewards, _, _ = env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.DO_NOTHING})
    assert agent_a.state == TrainState.STOPPED
    assert agent_a.current_entry_point == L
    assert agent_a.speed_counter.distance == Fraction(1)
    assert agent_a.speed_counter.speed == Fraction(0)
    assert rewards[0][DefaultPenalties.COLLISION.value] == -1 * speed * COLLISION_FACTOR
    assert rewards[0][DefaultPenalties.INVALID_ACTION.value] == 0

    # two more retries at the boundary: optimistic MOVING resumption (no penalty, nothing re-contested
    # yet) alternating with a genuine re-attempt, denied again at the same speed - position and distance
    # never move from where they were pinned.
    for expected_state, expected_collision in [
        (TrainState.MOVING, 0),
        (TrainState.STOPPED, -1 * speed * COLLISION_FACTOR),
    ]:
        _, rewards, _, _ = env.step({0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.DO_NOTHING})
        assert agent_a.state == expected_state
        assert agent_a.current_entry_point == L
        assert agent_a.speed_counter.distance == Fraction(1)
        assert rewards[0][DefaultPenalties.COLLISION.value] == expected_collision
        assert rewards[0][DefaultPenalties.INVALID_ACTION.value] == 0


def test_platoon_of_four_agents_starts_and_advances_together_without_force_stops():
    """
    Four agents A, B, C, D nose-to-tail on four consecutive cells C1, C2, C3, C4 of make_simple_rail's
    row-3 corridor, all heading the same direction, shared max speed 0.5.

    - Setup: each already stopped right at its own cell's exit boundary (distance 1.0, speed 0) - as if
      each had just been denied entry into the cell ahead of it, occupied by the next agent in line (C4,
      at the front, has open track ahead of it).
    - Start: given MOVE_FORWARD every step, all four start moving together on the same step - each
      reaches speed 0.5 while its position and distance stay exactly where they were.
    - Advance: every agent reaches its own cell's boundary on the same step as every other agent (same
      speed, same starting distance), and each time they do, they all advance into the next cell
      together in that same step - A into C2, B into C3, C into C4, D into open track beyond C4 - since
      the cell each of A, B and C requests is being vacated by the agent ahead of it in that very same
      step, and D's own target is open track throughout. Nobody is ever forced back to a stop.
    - Outcome: after three such advances, A has reached C4, B has reached C5, C has reached C6, and D
      has reached C7 - still nose-to-tail, still all at speed 0.5.
    """
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=4, random_seed=1,
                  rewards=BaseDefaultRewards(collision_factor=COLLISION_FACTOR))
    env.reset()
    env.acceleration_delta = Fraction(1)
    C1, C2, C3, C4, C5, C6, C7 = [((3, c), Grid4TransitionsEnum.WEST) for c in (8, 7, 6, 5, 4, 3, 2)]

    # place all four directly nose-to-tail on C1..C4, each already stopped right at its cell's boundary
    # (distance 1.0, speed 0) - see
    # test_agent_blocked_at_boundary_cannot_accelerate_nor_advance_into_stopped_neighbor for how a single
    # agent reaches exactly this state when denied entry into an occupied neighbor.
    for agent, cell in zip(env.agents, [C1, C2, C3, C4]):
        agent.current_entry_point = cell
        agent._set_state(TrainState.STOPPED)
        next_transition = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, cell)
        agent.next_entry_point = _sanitize_entry_point(next_transition)
        agent.speed_counter = SpeedCounter(max_speed=Fraction(1, 2), speed=Fraction(0))
        agent.speed_counter.set(speed=Fraction(0), distance=Fraction(0))
        agent.speed_counter._distance = Fraction(1)

    forward = {i: RailEnvActions.MOVE_FORWARD for i in range(4)}

    # all four start moving together: an optimistic STOPPED -> MOVING resumption, nothing re-contested
    # yet (each is still self-looping on its own cell) - free, no collision penalty anywhere.
    _, rewards, _, _ = env.step(forward)
    for agent, cell in zip(env.agents, [C1, C2, C3, C4]):
        assert agent.state == TrainState.MOVING
        assert agent.current_entry_point == cell
        assert agent.speed_counter.speed == Fraction(1, 2)
        assert agent.speed_counter.distance == Fraction(1)
    for h in range(4):
        assert rewards[h][DefaultPenalties.COLLISION.value] == 0
        assert rewards[h][DefaultPenalties.INVALID_ACTION.value] == 0

    # from here, every agent is genuinely at (or, every other step, still short of) its own boundary in
    # lockstep with the others; whenever they cross, they all cross together - nobody is ever force-
    # stopped, and the whole platoon ends up three cells further along, on C4, C5, C6, C7.
    for expected_distance, expected_cells in [
        (Fraction(1, 2), [C2, C3, C4, C5]),
        (Fraction(0), [C3, C4, C5, C6]),
        (Fraction(1, 2), [C3, C4, C5, C6]),
        (Fraction(0), [C4, C5, C6, C7]),
    ]:
        _, rewards, _, _ = env.step(forward)
        for agent, cell in zip(env.agents, expected_cells):
            assert agent.state == TrainState.MOVING  # never forced back to STOPPED
            assert agent.speed_counter.speed == Fraction(1, 2)
            assert agent.speed_counter.distance == expected_distance
            assert agent.current_entry_point == cell
        for h in range(4):
            assert rewards[h][DefaultPenalties.COLLISION.value] == 0
            assert rewards[h][DefaultPenalties.INVALID_ACTION.value] == 0

    assert [a.current_entry_point for a in env.agents] == [C4, C5, C6, C7]


@pytest.mark.parametrize("distance,expected_steps", [
    (Fraction(1, 5), [
        (Fraction(7, 10), 0),
        (Fraction(1, 5), 1),
        (Fraction(7, 10), 1),
        (Fraction(1, 5), 2),
    ]),
    (Fraction(1, 2), [
        (Fraction(0), 1),
        (Fraction(1, 2), 1),
        (Fraction(0), 2),
        (Fraction(1, 2), 2),
    ]),
    (Fraction(4, 5), [
        (Fraction(3, 10), 1),
        (Fraction(4, 5), 1),
        (Fraction(3, 10), 2),
        (Fraction(4, 5), 2),
    ]),
], ids=["distance-0.2", "distance-0.5", "distance-0.8"])
def test_platoon_of_four_agents_starting_mid_cell_moves_in_lockstep_without_force_stops(distance, expected_steps):
    """
    Four agents A, B, C, D nose-to-tail on four consecutive cells C1, C2, C3, C4 of make_simple_rail's
    row-3 corridor, all heading the same direction, shared max speed 0.5. Parametrized over starting
    distance: 0.2, 0.5 or 0.8 (all four at the same distance, in a given variant) - mid-cell this time,
    rather than right at the boundary like test_platoon_of_four_agents_starts_and_advances_together_without_force_stops.

    - Setup: each already stopped mid-cell at the given distance, speed 0.
    - Start: given MOVE_FORWARD every step, all four start moving together on the same step - each
      reaches speed 0.5 while its position and distance stay exactly where they were.
    - Advance: since they all started at the same distance and share the same speed, every agent keeps
      reaching its own cell's boundary on the same step as every other agent, and each time they do, they
      all cross into the next cell together in that same step - the cell each of A, B and C requests is
      being vacated by the agent ahead of it in that very same step, and D's own target is open track
      throughout. Nobody is ever forced back to a stop, mid-cell start or not.
    """
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=4, random_seed=1,
                  rewards=BaseDefaultRewards(collision_factor=COLLISION_FACTOR))
    env.reset()
    env.acceleration_delta = Fraction(1)
    cells = [((3, c), Grid4TransitionsEnum.WEST) for c in (8, 7, 6, 5, 4, 3, 2)]

    # place all four directly nose-to-tail on the first four cells, each already stopped mid-cell
    # (distance `distance`, short of its own exit boundary) with speed 0.
    for agent, cell in zip(env.agents, cells[:4]):
        agent.current_entry_point = cell
        agent._set_state(TrainState.STOPPED)
        next_transition = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, cell)
        agent.next_entry_point = _sanitize_entry_point(next_transition)
        agent.speed_counter = SpeedCounter(max_speed=Fraction(1, 2), speed=Fraction(0))
        agent.speed_counter.set(speed=Fraction(0), distance=Fraction(0))
        agent.speed_counter._distance = distance

    forward = {i: RailEnvActions.MOVE_FORWARD for i in range(4)}

    # all four start moving together: an optimistic STOPPED -> MOVING resumption, nothing re-contested
    # yet (each is still self-looping on its own cell) - free, no collision penalty anywhere.
    _, rewards, _, _ = env.step(forward)
    for agent, cell in zip(env.agents, cells[:4]):
        assert agent.state == TrainState.MOVING
        assert agent.current_entry_point == cell
        assert agent.speed_counter.speed == Fraction(1, 2)
        assert agent.speed_counter.distance == distance
    for h in range(4):
        assert rewards[h][DefaultPenalties.COLLISION.value] == 0
        assert rewards[h][DefaultPenalties.INVALID_ACTION.value] == 0

    # from here, every agent is genuinely at (or, on the steps in between, still short of) its own
    # boundary in lockstep with the others; whenever they cross, they all cross together - nobody is
    # ever force-stopped.
    for expected_distance, n_cells_advanced in expected_steps:
        _, rewards, _, _ = env.step(forward)
        for i, agent in enumerate(env.agents):
            assert agent.state == TrainState.MOVING  # never forced back to STOPPED
            assert agent.speed_counter.speed == Fraction(1, 2)
            assert agent.speed_counter.distance == expected_distance
            assert agent.current_entry_point == cells[i + n_cells_advanced]
        for h in range(4):
            assert rewards[h][DefaultPenalties.COLLISION.value] == 0
            assert rewards[h][DefaultPenalties.INVALID_ACTION.value] == 0


@pytest.mark.parametrize("max_speed,lockstep_steps,after_stop_offset,post_stop_distances,leader_distance", [
    # speed 0.2: the leader (D) is safely mid-cell (not at its own boundary) on the step it is told to
    # stop, so it simply brakes in place - no crossing that step; A/B/C then take one extra step still
    # short of their own boundary before all three finally reach it together.
    (Fraction(1, 5), [(Fraction(1, 5), 0), (Fraction(2, 5), 0)], 0, [Fraction(3, 5), Fraction(4, 5)], Fraction(3, 5)),
    # speed 0.5: same, but A/B/C are already exactly one step short of their own boundary the moment the
    # leader stops, so the very next step is already the one where all three reach it together.
    (Fraction(1, 2), [(Fraction(1, 2), 0), (Fraction(0), 1)], 1, [Fraction(1, 2)], Fraction(1, 2)),
    # speed 1.0: the leader is exactly at its own boundary on the step it is told to stop, so that one
    # crossing is already in flight and still completes (see
    # test_agent_blocked_at_boundary_cannot_accelerate_nor_advance_into_stopped_neighbor's STOP_MOVING
    # discussion) - the leader (and, moving with it in lockstep, A/B/C too) still advances one more cell
    # on this step before the leader genuinely holds still from the next step on.
    (Fraction(1), [(Fraction(0), 1), (Fraction(0), 2)], 3, [Fraction(0)], Fraction(0)),
], ids=["speed-0.2", "speed-0.5", "speed-1.0"])
def test_platoon_all_stop_together_once_leader_stops_and_stays_stopped(
    max_speed, lockstep_steps, after_stop_offset, post_stop_distances, leader_distance):
    """
    Four agents A, B, C, D nose-to-tail on four consecutive cells C1, C2, C3, C4 of make_simple_rail's
    row-3 corridor, all heading the same direction, cruising together at a shared max speed (0.2, 0.5 or
    1.0 in the three variants).

    - Setup: all four already MOVING in lockstep at the shared max speed (distance 0).
    - Lockstep steps: two steps of ordinary forward movement - everyone advances identically, nobody
      blocked (same pattern as test_platoon_of_four_agents_starts_and_advances_together_without_force_stops).
    - Leader stops: D is given STOP_MOVING every step from then on and never moves again; A, B and C
      keep being told to move forward. The number of steps before A/B/C catch up to their own boundary
      differs by speed: immediate for 0.5 and 1.0, one extra "still approaching" step first for 0.2.
    - Speed 1.0 detail: D is exactly at its own boundary the moment it is told to stop, so that crossing
      is already in flight and still completes that step before D genuinely holds still.
    - Convergence: A, B and C all transition to STOPPED with distance pinned at 1.0 ("end of cell") in
      the exact same env.step() call - not staggered - while D's own state/position/distance stay
      unchanged, for all three speed variants.
    - Rewards: D's own MOVING->STOPPED transition (the step it first settles) is a voluntary stop
      (STOP_MOVING given, speed reaches 0, movement_allowed stays True - see
      BaseDefaultRewards.step_reward) - no penalty. A, B and C's later, simultaneous MOVING->STOPPED
      transition at convergence is env-forced (denied by D no longer vacating the cell ahead) -
      charged a COLLISION penalty of max_speed times collision_factor each, never INVALID_ACTION,
      since MOVE_FORWARD is a structurally valid action throughout.
    """
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=4, random_seed=1,
                  rewards=BaseDefaultRewards(collision_factor=COLLISION_FACTOR))
    env.reset()
    env.acceleration_delta = Fraction(1)
    env.braking_delta = Fraction(-1)
    cells = [((3, c), Grid4TransitionsEnum.WEST) for c in (8, 7, 6, 5, 4, 3, 2)]

    # place all four directly nose-to-tail on the first four cells, already MOVING at the shared max
    # speed with distance 0 (freshly cruising) - see
    # test_platoon_of_four_agents_starts_and_advances_together_without_force_stops for how a platoon
    # reaches this state from a genuine standstill.
    for agent, cell in zip(env.agents, cells[:4]):
        agent.current_entry_point = cell
        agent._set_state(TrainState.MOVING)
        next_transition = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, cell)
        agent.next_entry_point = _sanitize_entry_point(next_transition)
        agent.speed_counter = SpeedCounter(max_speed=max_speed, speed=max_speed)
        agent.speed_counter.set(speed=max_speed, distance=Fraction(0))  # bootstrap onto the map at distance 0

    forward = {i: RailEnvActions.MOVE_FORWARD for i in range(4)}
    hold_leader = {0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD,
                   2: RailEnvActions.MOVE_FORWARD, 3: RailEnvActions.STOP_MOVING}

    # two steps of ordinary lockstep cruising, leader included - nobody blocked yet.
    for expected_distance, n_cells_advanced in lockstep_steps:
        env.step(forward)
        for i, agent in enumerate(env.agents):
            assert agent.state == TrainState.MOVING
            assert agent.speed_counter.speed == max_speed
            assert agent.speed_counter.distance == expected_distance
            assert agent.current_entry_point == cells[i + n_cells_advanced]

    agent_a, agent_b, agent_c, agent_d = env.agents

    # the leader is told to stop, and keeps being told to stop every step from here on; the others are
    # still told to keep moving forward. For as long as A/B/C haven't yet reached their own cell's
    # boundary, they are not blocked yet - the leader's position (and, once it settles, its distance) no
    # longer changes, but A/B/C keep advancing within their own cell just like before. D's own
    # MOVING->STOPPED transition happens on the first of these steps - a voluntary stop (STOP_MOVING,
    # not denied), so no penalty for D even once collision_factor is nonzero; A/B/C are still MOVING
    # throughout this loop, so no penalty for them either.
    for expected_abc_distance in post_stop_distances:
        _, rewards, _, _ = env.step(hold_leader)
        for i, agent in enumerate(env.agents):
            assert agent.current_entry_point == cells[i + after_stop_offset]
        assert agent_d.state == TrainState.STOPPED
        assert agent_d.speed_counter.speed == Fraction(0)
        assert agent_d.speed_counter.distance == leader_distance
        assert rewards[3][DefaultPenalties.COLLISION.value] == 0  # D's own stop is voluntary
        assert rewards[3][DefaultPenalties.INVALID_ACTION.value] == 0
        for agent in (agent_a, agent_b, agent_c):
            assert agent.state == TrainState.MOVING
            assert agent.speed_counter.speed == max_speed
            assert agent.speed_counter.distance == expected_abc_distance
        for h in range(3):
            assert rewards[h][DefaultPenalties.COLLISION.value] == 0
            assert rewards[h][DefaultPenalties.INVALID_ACTION.value] == 0

    # the leader keeps being told to stay stopped; the others keep trying to move forward - denied
    # together, in this one step, since D is no longer vacating the cell ahead of C (and, transitively,
    # C for B, B for A): all three reach the end of their own cell (distance 1.0) and stop, in the same
    # time step as each other - not staggered one after another. This is an env-forced stop (denied by
    # D), charged a COLLISION penalty (max_speed times collision_factor) for each of A, B and C; D
    # itself has no transition this step (already STOPPED going in), so no penalty either.
    _, rewards, _, _ = env.step(hold_leader)
    for i, agent in enumerate(env.agents):
        assert agent.current_entry_point == cells[i + after_stop_offset]  # nobody advanced any further
    assert agent_d.state == TrainState.STOPPED
    assert agent_d.speed_counter.speed == Fraction(0)
    assert agent_d.speed_counter.distance == leader_distance  # unchanged
    assert rewards[3][DefaultPenalties.COLLISION.value] == 0
    assert rewards[3][DefaultPenalties.INVALID_ACTION.value] == 0
    for agent in (agent_a, agent_b, agent_c):
        assert agent.state == TrainState.STOPPED  # all three, together, in this same step
        assert agent.speed_counter.speed == Fraction(0)
        assert agent.speed_counter.distance == Fraction(1)  # end of cell
    for h in range(3):
        assert rewards[h][DefaultPenalties.COLLISION.value] == -1 * max_speed * COLLISION_FACTOR
        assert rewards[h][DefaultPenalties.INVALID_ACTION.value] == 0


@pytest.mark.parametrize("max_speed,steps", [
    (Fraction(1), [
        (TrainState.MOVING, 7, Fraction(4, 5), 0, 6, Fraction(1, 5)),
        (TrainState.MOVING, 6, Fraction(4, 5), 0, 5, Fraction(1, 5)),
        (TrainState.MOVING, 5, Fraction(4, 5), 0, 4, Fraction(1, 5)),
        (TrainState.MOVING, 4, Fraction(4, 5), 0, 3, Fraction(1, 5)),
    ]),
    (Fraction(1, 2), [
        (TrainState.STOPPED, 8, Fraction(1), -1 * Fraction(1, 2) * COLLISION_FACTOR, 7, Fraction(7, 10)),
        (TrainState.MOVING, 8, Fraction(1), 0, 6, Fraction(1, 5)),
        (TrainState.MOVING, 7, Fraction(1, 2), 0, 6, Fraction(7, 10)),
        (TrainState.MOVING, 6, Fraction(0), 0, 5, Fraction(1, 5)),
        (TrainState.MOVING, 6, Fraction(1, 2), 0, 5, Fraction(7, 10)),
        (TrainState.MOVING, 5, Fraction(0), 0, 4, Fraction(1, 5)),
        (TrainState.MOVING, 5, Fraction(1, 2), 0, 4, Fraction(7, 10)),
    ]),
    (Fraction(1, 5), [
        (TrainState.STOPPED, 8, Fraction(1), -1 * Fraction(1, 5) * COLLISION_FACTOR, 7, Fraction(2, 5)),
        (TrainState.MOVING, 8, Fraction(1), 0, 7, Fraction(3, 5)),
        (TrainState.STOPPED, 8, Fraction(1), -1 * Fraction(1, 5) * COLLISION_FACTOR, 7, Fraction(4, 5)),
        (TrainState.MOVING, 8, Fraction(1), 0, 6, Fraction(0)),
        (TrainState.MOVING, 7, Fraction(1, 5), 0, 6, Fraction(1, 5)),
        (TrainState.MOVING, 7, Fraction(2, 5), 0, 6, Fraction(2, 5)),
        (TrainState.MOVING, 7, Fraction(3, 5), 0, 6, Fraction(3, 5)),
        (TrainState.MOVING, 7, Fraction(4, 5), 0, 6, Fraction(4, 5)),
        (TrainState.MOVING, 6, Fraction(0), 0, 5, Fraction(0)),
    ]),
], ids=["speed-1.0-never-stopped", "speed-0.5-one-stop", "speed-0.2-two-stops"])
def test_two_agents_different_in_cell_distance_converge_to_lockstep(max_speed, steps):
    """
    Two agents, F and R, on adjacent cells C2 and C1 of make_simple_rail's row-3 corridor, both heading
    the same direction and sharing the same max speed, given MOVE_FORWARD every step.

    - Setup: F on C2 at distance 0.2 (freshly entered its cell), R on C1 at distance 0.8 (already close
      to its own exit boundary), both MOVING at the shared max speed.
    - max speed 1.0: R is at (or past) its own exit boundary on every single step regardless of its
      starting distance, so both cross into the next cell together every step from the first one - R is
      never force-stopped.
    - max speed 0.5: R reaches its own boundary before F has vacated the cell ahead and is force-stopped
      once, for one step; from the very next step on, R and F settle into a stable, repeating crossing
      cadence (one step advancing within-cell, the next crossing together) - R is never stopped again.
    - max speed 0.2: R is force-stopped twice, not once - F needs longer, at this slower speed, to first
      reach its own boundary and vacate, so R's first retry after the initial stop is still too early and
      is denied again. From the second recovery on, R and F settle into the same kind of stable, repeating
      crossing cadence, this time with identical distance values every step - R is never stopped again.
    - Rewards: each of R's force-stops is a motion-check conflict with F (MOVE_FORWARD is always a
      valid action for both), so it is charged under COLLISION, never INVALID_ACTION.
    """
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=2, random_seed=1,
                  rewards=BaseDefaultRewards(collision_factor=COLLISION_FACTOR))
    env.reset()
    env.acceleration_delta = Fraction(1)
    C1, C2 = [((3, c), Grid4TransitionsEnum.WEST) for c in (8, 7)]

    # F (front) on C2 at distance 0.2 - freshly entered its cell; R (rear) on C1 at distance 0.8 -
    # already close to its own exit boundary. Both share max_speed.
    agent_f, agent_r = env.agents
    for agent, cell, distance in [(agent_f, C2, Fraction(1, 5)), (agent_r, C1, Fraction(4, 5))]:
        agent.current_entry_point = cell
        agent._set_state(TrainState.MOVING)
        next_transition = env.rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, cell)
        agent.next_entry_point = _sanitize_entry_point(next_transition)
        agent.speed_counter = SpeedCounter(max_speed=max_speed, speed=max_speed)
        agent.speed_counter.set(speed=max_speed, distance=Fraction(0))
        agent.speed_counter._distance = distance

    forward = {0: RailEnvActions.MOVE_FORWARD, 1: RailEnvActions.MOVE_FORWARD}

    for r_state, r_col, r_distance, r_collision, f_col, f_distance in steps:
        _, rewards, _, _ = env.step(forward)
        expected_r_speed = Fraction(0) if r_state == TrainState.STOPPED else max_speed
        assert agent_r.state == r_state
        assert agent_r.current_entry_point == ((3, r_col), Grid4TransitionsEnum.WEST)
        assert agent_r.speed_counter.speed == expected_r_speed
        assert agent_r.speed_counter.distance == r_distance
        assert rewards[1][DefaultPenalties.COLLISION.value] == r_collision
        assert rewards[1][DefaultPenalties.INVALID_ACTION.value] == 0

        assert agent_f.state == TrainState.MOVING  # F is never blocked - open track ahead of it throughout
        assert agent_f.current_entry_point == ((3, f_col), Grid4TransitionsEnum.WEST)
        assert agent_f.speed_counter.speed == max_speed
        assert agent_f.speed_counter.distance == f_distance
        assert rewards[0][DefaultPenalties.COLLISION.value] == 0
        assert rewards[0][DefaultPenalties.INVALID_ACTION.value] == 0
