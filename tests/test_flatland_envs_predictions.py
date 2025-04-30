#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pprint

import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv, Node
from flatland.envs.predictions import DummyPredictorForRailEnv, ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState
from flatland.utils.rendertools import RenderTool
from flatland.utils.simple_rail import make_simple_rail, make_simple_rail2, make_invalid_simple_rail

"""Test predictions for `flatland` package."""


def test_dummy_predictor(rendering=False):
    rail, rail_map, optionals = make_simple_rail2()

    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(),
                  number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=DummyPredictorForRailEnv(max_depth=10)),
                  )
    env.reset()

    # set initial position and direction for testing...
    env.agents[0].initial_position = (5, 6)
    env.agents[0].initial_direction = 0
    env.agents[0].direction = 0
    env.agents[0].target = (3, 0)

    env.reset(False, False)
    env.agents[0].earliest_departure = 1
    env._max_episode_steps = 100
    # Make Agent 0 active
    env.step({})
    env.step({0: RailEnvActions.MOVE_FORWARD})

    if rendering:
        renderer = RenderTool(env, gl="PILSVG")
        renderer.render_env(show=True, show_observations=False)
        input("Continue?")

    # test assertions
    predictions = env.obs_builder.predictor.get(None)
    positions = np.array(list(map(lambda prediction: [*prediction[1:3]], predictions[0])))
    directions = np.array(list(map(lambda prediction: [prediction[3]], predictions[0])))
    time_offsets = np.array(list(map(lambda prediction: [prediction[0]], predictions[0])))
    actions = np.array(list(map(lambda prediction: [prediction[4]], predictions[0])))

    # compare against expected values
    expected_positions = np.array([[5., 6.],
                                   [4., 6.],
                                   [3., 6.],
                                   [3., 5.],
                                   [3., 4.],
                                   [3., 3.],
                                   [3., 2.],
                                   [3., 1.],
                                   # at target (3,0): stay in this position from here on
                                   [3., 0.],
                                   [3., 0.],
                                   [3., 0.],
                                   ])
    expected_directions = np.array([[0.],
                                    [0.],
                                    [0.],
                                    [3.],
                                    [3.],
                                    [3.],
                                    [3.],
                                    [3.],
                                    # at target (3,0): stay in this position from here on
                                    [3.],
                                    [3.],
                                    [3.]
                                    ])
    expected_time_offsets = np.array([[0.],
                                      [1.],
                                      [2.],
                                      [3.],
                                      [4.],
                                      [5.],
                                      [6.],
                                      [7.],
                                      [8.],
                                      [9.],
                                      [10.],
                                      ])
    expected_actions = np.array([[0.],
                                 [2.],
                                 [2.],
                                 [2.],
                                 [2.],
                                 [2.],
                                 [2.],
                                 [2.],
                                 # reaching target by straight
                                 [2.],
                                 # at target: stopped moving
                                 [4.],
                                 [4.],
                                 ])
    assert np.array_equal(positions, expected_positions)
    assert np.array_equal(directions, expected_directions)
    assert np.array_equal(time_offsets, expected_time_offsets)
    assert np.array_equal(actions, expected_actions)


def test_shortest_path_predictor(rendering=False):
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(),
                  number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  )
    env.reset()

    # set the initial position
    agent = env.agents[0]
    agent.initial_position = (5, 6)  # south dead-end
    agent.position = (5, 6)  # south dead-end
    agent.direction = 0  # north
    agent.initial_direction = 0  # north
    agent.target = (3, 9)  # east dead-end
    agent.moving = True
    agent._set_state(TrainState.MOVING)

    env.reset(False, False)
    env.distance_map._compute(env.agents, env.rail)

    # Perform DO_NOTHING actions until all trains get to READY_TO_DEPART
    for _ in range(max([agent.earliest_departure for agent in env.agents])):
        env.step({})  # DO_NOTHING for all agents

    if rendering:
        renderer = RenderTool(env, gl="PILSVG")
        renderer.render_env(show=True, show_observations=False)
        input("Continue?")

    # compute the observations and predictions
    distance_map = env.distance_map.get()
    distance_on_map = distance_map[0, agent.initial_position[0], agent.initial_position[1], agent.direction]
    assert distance_on_map == 5.0, "found {} instead of {}".format(distance_on_map, 5.0)

    paths = env.distance_map.get_shortest_paths()[0]
    assert paths == [
        Waypoint((5, 6), 0),
        Waypoint((4, 6), 0),
        Waypoint((3, 6), 0),
        Waypoint((3, 7), 1),
        Waypoint((3, 8), 1),
        Waypoint((3, 9), 1)
    ]

    # extract the data
    predictions = env.obs_builder.predictions
    positions = np.array(list(map(lambda prediction: [*prediction[1:3]], predictions[0])))
    directions = np.array(list(map(lambda prediction: [prediction[3]], predictions[0])))
    time_offsets = np.array(list(map(lambda prediction: [prediction[0]], predictions[0])))

    # test if data meets expectations
    expected_positions = [
        [5, 6],
        [4, 6],
        [3, 6],
        [3, 7],
        [3, 8],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
        [3, 9],
    ]
    expected_directions = [
        [Grid4TransitionsEnum.NORTH],  # next is [5,6] heading north
        [Grid4TransitionsEnum.NORTH],  # next is [4,6] heading north
        [Grid4TransitionsEnum.NORTH],  # next is [3,6] heading north
        [Grid4TransitionsEnum.EAST],  # next is [3,7] heading east
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
        [Grid4TransitionsEnum.EAST],
    ]

    expected_time_offsets = np.array([
        [0.],
        [1.],
        [2.],
        [3.],
        [4.],
        [5.],
        [6.],
        [7.],
        [8.],
        [9.],
        [10.],
        [11.],
        [12.],
        [13.],
        [14.],
        [15.],
        [16.],
        [17.],
        [18.],
        [19.],
        [20.],
    ])

    assert np.array_equal(time_offsets, expected_time_offsets), \
        "time_offsets {}, expected {}".format(time_offsets, expected_time_offsets)

    assert np.array_equal(positions, expected_positions), \
        "positions {}, expected {}".format(positions, expected_positions)
    assert np.array_equal(directions, expected_directions), \
        "directions {}, expected {}".format(directions, expected_directions)


def test_shortest_path_predictor_conflicts(rendering=False):
    rail, rail_map, optionals = make_invalid_simple_rail()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(),
                  number_of_agents=2,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  )
    env.reset()

    # set the initial position
    env.agents[0].initial_position = (5, 6)  # south dead-end
    env.agents[0].position = (5, 6)  # south dead-end
    env.agents[0].direction = 0  # north
    env.agents[0].initial_direction = 0  # north
    env.agents[0].target = (3, 9)  # east dead-end
    env.agents[0].moving = True
    env.agents[0]._set_state(TrainState.MOVING)

    env.agents[1].initial_position = (3, 8)  # east dead-end
    env.agents[1].position = (3, 8)  # east dead-end
    env.agents[1].direction = 3  # west
    env.agents[1].initial_direction = 3  # west
    env.agents[1].target = (6, 6)  # south dead-end
    env.agents[1].moving = True
    env.agents[1]._set_state(TrainState.MOVING)

    observations, info = env.reset(False, False)

    env.agents[0].position = (5, 6)  # south dead-end
    env.agent_positions[env.agents[0].position] = 0
    env.agents[1].position = (3, 8)  # east dead-end
    env.agent_positions[env.agents[1].position] = 1
    env.agents[0]._set_state(TrainState.MOVING)
    env.agents[1]._set_state(TrainState.MOVING)

    observations = env._get_observations()

    if rendering:
        renderer = RenderTool(env, gl="PILSVG")
        renderer.render_env(show=True, show_observations=False)
        input("Continue?")

    # get the trees to test
    obs_builder: TreeObsForRailEnv = env.obs_builder
    pp = pprint.PrettyPrinter(indent=4)
    tree_0 = observations[0]
    tree_1 = observations[1]
    env.obs_builder.util_print_obs_subtree(tree_0)
    env.obs_builder.util_print_obs_subtree(tree_1)

    # check the expectations
    expected_conflicts_0 = [('F', 'R')]
    expected_conflicts_1 = [('F', 'L')]
    _check_expected_conflicts(expected_conflicts_0, obs_builder, tree_0, "agent[0]: ")
    _check_expected_conflicts(expected_conflicts_1, obs_builder, tree_1, "agent[1]: ")


def _check_expected_conflicts(expected_conflicts, obs_builder, tree: Node, prompt=''):
    assert (tree.num_agents_opposite_direction > 0) == (() in expected_conflicts), "{}[]".format(prompt)
    for a_1 in obs_builder.tree_explored_actions_char:
        if tree.childs[a_1] == -np.inf:
            assert False == ((a_1) in expected_conflicts), "{}[{}]".format(prompt, a_1)
            continue
        else:
            conflict = tree.childs[a_1].num_agents_opposite_direction
            assert (conflict > 0) == ((a_1) in expected_conflicts), "{}[{}]".format(prompt, a_1)
        for a_2 in obs_builder.tree_explored_actions_char:
            if tree.childs[a_1].childs[a_2] == -np.inf:
                assert False == ((a_1, a_2) in expected_conflicts), "{}[{}][{}]".format(prompt, a_1, a_2)
            else:
                conflict = tree.childs[a_1].childs[a_2].num_agents_opposite_direction
                assert (conflict > 0) == ((a_1, a_2) in expected_conflicts), "{}[{}][{}]".format(prompt, a_1, a_2)
