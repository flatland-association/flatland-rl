#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Callable

import numpy as np
import pytest

from env_generation.env_creator import env_creator
from flatland.core.env_observation_builder import DummyObservationBuilder, ObservationBuilder
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv, Node
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.step_utils.states import TrainState
from flatland.utils.rendertools import RenderTool
from flatland.utils.simple_rail import make_simple_rail

"""Tests for `flatland` package."""


def test_global_obs():
    rail, rail_map, optionals = make_simple_rail()

    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1,
                  obs_builder_object=GlobalObsForRailEnv())

    global_obs, info = env.reset()

    # we have to take step for the agent to enter the grid.
    global_obs, _, _, _ = env.step({0: RailEnvActions.MOVE_FORWARD})

    assert (global_obs[0][0].shape == rail_map.shape + (16,))

    rail_map_recons = np.zeros_like(rail_map)
    for i in range(global_obs[0][0].shape[0]):
        for j in range(global_obs[0][0].shape[1]):
            rail_map_recons[i, j] = int(
                ''.join(global_obs[0][0][i, j].astype(int).astype(str)), 2)

    assert (rail_map_recons.all() == rail_map.all())

    # If this assertion is wrong, it means that the observation returned
    # places the agent on an empty cell
    obs_agents_state = global_obs[0][1]
    obs_agents_state = obs_agents_state + 1
    assert (np.sum(rail_map * obs_agents_state[:, :, :4].sum(2)) > 0)


def _step_along_shortest_path(env, obs_builder, rail):
    actions = {}
    expected_next_position = {}
    for agent in env.agents:
        shortest_distance = np.inf

        for exit_direction in range(4):
            neighbour = get_new_position(agent.position, exit_direction)

            if neighbour[0] >= 0 and neighbour[0] < env.height and neighbour[1] >= 0 and neighbour[1] < env.width:
                desired_movement_from_new_cell = (exit_direction + 2) % 4

                # Check all possible transitions in new_cell
                for agent_orientation in range(4):
                    # Is a transition along movement `entry_direction` to the neighbour possible?
                    is_valid = obs_builder.env.rail.get_transition((neighbour[0], neighbour[1], agent_orientation),
                                                                   desired_movement_from_new_cell)
                    if is_valid:
                        distance_to_target = obs_builder.env.distance_map.get()[
                            (agent.handle, *agent.position, exit_direction)]
                        print("agent {} at {} facing {} taking {} distance {}".format(agent.handle, agent.position,
                                                                                      agent.direction,
                                                                                      exit_direction,
                                                                                      distance_to_target))

                        if distance_to_target < shortest_distance:
                            shortest_distance = distance_to_target
                            actions_to_be_taken_when_facing_north = {
                                Grid4TransitionsEnum.NORTH: RailEnvActions.MOVE_FORWARD,
                                Grid4TransitionsEnum.EAST: RailEnvActions.MOVE_RIGHT,
                                Grid4TransitionsEnum.WEST: RailEnvActions.MOVE_LEFT,
                                Grid4TransitionsEnum.SOUTH: RailEnvActions.DO_NOTHING,
                            }
                            print("   improved (direction) -> {}".format(exit_direction))

                            actions[agent.handle] = actions_to_be_taken_when_facing_north[
                                (exit_direction - agent.direction) % len(rail.transitions.get_direction_enum())]
                            expected_next_position[agent.handle] = neighbour
                            print("   improved (action) -> {}".format(actions[agent.handle]))
    _, rewards, dones, _ = env.step(actions)
    return rewards, dones


def test_reward_function_conflict(rendering=False):
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=2,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    obs_builder: TreeObsForRailEnv = env.obs_builder
    env.reset()

    # set the initial position
    agent = env.agents[0]
    agent.position = (5, 6)  # south dead-end
    agent.initial_position = (5, 6)  # south dead-end
    agent.direction = 0  # north
    agent.initial_direction = 0  # north
    agent.target = (3, 9)  # east dead-end
    agent.moving = True
    agent._set_state(TrainState.MOVING)

    agent = env.agents[1]
    agent.position = (3, 8)  # east dead-end
    agent.initial_position = (3, 8)  # east dead-end
    agent.direction = 3  # west
    agent.initial_direction = 3  # west
    agent.target = (6, 6)  # south dead-end
    agent.moving = True
    agent._set_state(TrainState.MOVING)

    env.reset(False, False)
    env.agents[0].moving = True
    env.agents[1].moving = True
    env.agents[0]._set_state(TrainState.MOVING)
    env.agents[1]._set_state(TrainState.MOVING)
    env.agents[0].position = (5, 6)
    env.agents[1].position = (3, 8)
    print("\n")
    print(env.agents[0])
    print(env.agents[1])

    if rendering:
        renderer = RenderTool(env, gl="PILSVG")
        renderer.render_env(show=True, show_observations=True)

    iteration = 0
    expected_positions = {
        0: {
            0: (5, 6),
            1: (3, 8)
        },
        # both can move
        1: {
            0: (4, 6),
            1: (3, 7)
        },
        # first can move, second stuck
        2: {
            0: (3, 6),
            1: (3, 7)
        },
        # both stuck from now on
        3: {
            0: (3, 6),
            1: (3, 7)
        },
        4: {
            0: (3, 6),
            1: (3, 7)
        },
        5: {
            0: (3, 6),
            1: (3, 7)
        },
    }
    while iteration < 5:
        rewards, dones = _step_along_shortest_path(env, obs_builder, rail)
        if dones["__all__"]:
            break
        for agent in env.agents:
            # assert rewards[agent.handle] == 0
            expected_position = expected_positions[iteration + 1][agent.handle]
            assert agent.position == expected_position, "[{}] agent {} at {}, expected {}".format(iteration + 1,
                                                                                                  agent.handle,
                                                                                                  agent.position,
                                                                                                  expected_position)
        if rendering:
            renderer.render_env(show=True, show_observations=True)

        iteration += 1


def test_reward_function_waiting(rendering=False):
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=2,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  remove_agents_at_target=False, random_seed=1)
    obs_builder: TreeObsForRailEnv = env.obs_builder
    env.reset()

    # set the initial position
    agent = env.agents[0]
    agent.initial_position = (3, 8)  # east dead-end
    agent.position = (3, 8)  # east dead-end
    agent.direction = 3  # west
    agent.initial_direction = 3  # west
    agent.target = (3, 1)  # west dead-end
    agent.moving = True
    agent._set_state(TrainState.MOVING)

    agent = env.agents[1]
    agent.initial_position = (5, 6)  # south dead-end
    agent.position = (5, 6)  # south dead-end
    agent.direction = 0  # north
    agent.initial_direction = 0  # north
    agent.target = (3, 8)  # east dead-end
    agent.moving = True
    agent._set_state(TrainState.MOVING)

    env.reset(False, False)
    env.agents[0].moving = True
    env.agents[1].moving = True
    env.agents[0]._set_state(TrainState.MOVING)
    env.agents[1]._set_state(TrainState.MOVING)
    env.agents[0].position = (3, 8)
    env.agents[1].position = (5, 6)

    if rendering:
        renderer = RenderTool(env, gl="PILSVG")
        renderer.render_env(show=True, show_observations=True)

    iteration = 0
    expectations = {
        0: {
            'positions': {
                0: (3, 8),
                1: (5, 6),
            },
            'rewards': [0, 0],
        },
        1: {
            'positions': {
                0: (3, 7),
                1: (4, 6),
            },
            'rewards': [0, 0],
        },
        # second agent has to wait for first, first can continue
        2: {
            'positions': {
                0: (3, 6),
                1: (4, 6),
            },
            'rewards': [0, 0],
        },
        # both can move again
        3: {
            'positions': {
                0: (3, 5),
                1: (3, 6),
            },
            'rewards': [0, 0],
        },
        4: {
            'positions': {
                0: (3, 4),
                1: (3, 7),
            },
            'rewards': [0, 0],
        },
        # second reached target
        5: {
            'positions': {
                0: (3, 3),
                1: (3, 8),
            },
            'rewards': [0, 0],
        },
        6: {
            'positions': {
                0: (3, 2),
                1: (3, 8),
            },
            'rewards': [0, 0],
        },
        # first reaches, target too
        7: {
            'positions': {
                0: (3, 1),
                1: (3, 8),
            },
            'rewards': [0, 0],
        },
        8: {
            'positions': {
                0: (3, 1),
                1: (3, 8),
            },
            'rewards': [0, 0],
        },
    }
    while iteration < 7:

        rewards, dones = _step_along_shortest_path(env, obs_builder, rail)
        if dones["__all__"]:
            break

        if rendering:
            renderer.render_env(show=True, show_observations=True)

        print(env.dones["__all__"])
        for agent in env.agents:
            print("[{}] agent {} at {}, target {} ".format(iteration + 1, agent.handle, agent.position, agent.target))
        print(np.all([np.array_equal(agent2.position, agent2.target) for agent2 in env.agents]))
        for agent in env.agents:
            expected_position = expectations[iteration + 1]['positions'][agent.handle]
            assert agent.position == expected_position, \
                "[{}] agent {} at {}, expected {}".format(iteration + 1,
                                                          agent.handle,
                                                          agent.position,
                                                          expected_position)
            # expected_reward = expectations[iteration + 1]['rewards'][agent.handle]
            # actual_reward = rewards[agent.handle]
            # assert expected_reward == actual_reward, "[{}] agent {} reward {}, expected {}".format(iteration + 1,
            #                                                                                        agent.handle,
            #                                                                                        actual_reward,
            #                                                                                        expected_reward)
        iteration += 1


@pytest.mark.parametrize(
    "obs_builder,expected_shape",
    [
        pytest.param(obs_builder, expected_shape, id=f"{obid}")
        for obs_builder, obid, expected_shape in
        [
            (TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)), "FlattenTreeObsForRailEnv_max_depth_3_50",
             lambda v: type(v) == Node),
            (DummyObservationBuilder(), "DummyObservationBuilderGym", lambda v: type(v) == bool),
            (GlobalObsForRailEnv(), "GlobalObsForRailEnvGym",
             lambda v: type(v) == tuple
                       and len(v) == 3
                       and v[0].shape == (30, 30, 16) and v[0].dtype == float
                       and v[1].shape == (30, 30, 5) and v[1].dtype == float
                       and v[2].shape == (30, 30, 2) and v[2].dtype == float),
        ]
    ]
)
def test_obs_builder_gym(obs_builder: ObservationBuilder, expected_shape: Callable):
    expected_dtype = float
    expected_agent_ids = [0, 1, 2, 3, 4, 5, 6]

    env = env_creator(obs_builder_object=obs_builder)

    for agent_id in env.agents:
        space_shape = env.get_observation_space(agent_id).shape
        assert space_shape == expected_shape, (expected_shape, space_shape)
        space_dtype = env.get_observation_space(agent_id).dtype
        assert space_dtype == expected_dtype
        sample_shape = env.get_observation_space(agent_id).sample().shape
        assert sample_shape == expected_shape, (expected_shape, sample_shape)
    obs, _ = env.reset()
    assert list(obs.keys()) == expected_agent_ids
    for i in range(7):
        assert expected_shape(obs[i])
    obs, _, _, _ = env.step({})
    assert list(obs.keys()) == expected_agent_ids
    for i in range(7):
        assert expected_shape(obs[i])
