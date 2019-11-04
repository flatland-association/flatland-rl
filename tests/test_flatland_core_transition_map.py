from flatland.core.grid.grid4 import Grid4Transitions, Grid4TransitionsEnum
from flatland.core.grid.grid8 import Grid8Transitions, Grid8TransitionsEnum
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.schedule_generators import random_schedule_generator
from flatland.utils.rendertools import RenderTool
from flatland.utils.simple_rail import make_simple_rail, make_simple_rail_unconnected


def test_grid4_get_transitions():
    grid4_map = GridTransitionMap(2, 2, Grid4Transitions([]))
    assert grid4_map.get_transitions(0, 0, Grid4TransitionsEnum.NORTH) == (0, 0, 0, 0)
    assert grid4_map.get_transitions(0, 0, Grid4TransitionsEnum.EAST) == (0, 0, 0, 0)
    assert grid4_map.get_transitions(0, 0, Grid4TransitionsEnum.SOUTH) == (0, 0, 0, 0)
    assert grid4_map.get_transitions(0, 0, Grid4TransitionsEnum.WEST) == (0, 0, 0, 0)
    assert grid4_map.get_full_transitions(0, 0) == 0

    grid4_map.set_transition((0, 0, Grid4TransitionsEnum.NORTH), Grid4TransitionsEnum.NORTH, 1)
    assert grid4_map.get_transitions(0, 0, Grid4TransitionsEnum.NORTH) == (1, 0, 0, 0)
    assert grid4_map.get_transitions(0, 0, Grid4TransitionsEnum.EAST) == (0, 0, 0, 0)
    assert grid4_map.get_transitions(0, 0, Grid4TransitionsEnum.SOUTH) == (0, 0, 0, 0)
    assert grid4_map.get_transitions(0, 0, Grid4TransitionsEnum.WEST) == (0, 0, 0, 0)
    assert grid4_map.get_full_transitions(0, 0) == pow(2, 15)  # the most significant bit is on

    grid4_map.set_transition((0, 0, Grid4TransitionsEnum.NORTH), Grid4TransitionsEnum.WEST, 1)
    assert grid4_map.get_transitions(0, 0, Grid4TransitionsEnum.NORTH) == (1, 0, 0, 1)
    assert grid4_map.get_transitions(0, 0, Grid4TransitionsEnum.EAST) == (0, 0, 0, 0)
    assert grid4_map.get_transitions(0, 0, Grid4TransitionsEnum.SOUTH) == (0, 0, 0, 0)
    assert grid4_map.get_transitions(0, 0, Grid4TransitionsEnum.WEST) == (0, 0, 0, 0)
    # the most significant and the fourth most significant bits are on
    assert grid4_map.get_full_transitions(0, 0) == pow(2, 15) + pow(2, 12)

    grid4_map.set_transition((0, 0, Grid4TransitionsEnum.NORTH), Grid4TransitionsEnum.NORTH, 0)
    assert grid4_map.get_transitions(0, 0, Grid4TransitionsEnum.NORTH) == (0, 0, 0, 1)
    assert grid4_map.get_transitions(0, 0, Grid4TransitionsEnum.EAST) == (0, 0, 0, 0)
    assert grid4_map.get_transitions(0, 0, Grid4TransitionsEnum.SOUTH) == (0, 0, 0, 0)
    assert grid4_map.get_transitions(0, 0, Grid4TransitionsEnum.WEST) == (0, 0, 0, 0)
    # the fourth most significant bits are on
    assert grid4_map.get_full_transitions(0, 0) == pow(2, 12)


def test_grid8_set_transitions():
    grid8_map = GridTransitionMap(2, 2, Grid8Transitions([]))
    assert grid8_map.get_transitions(0, 0, Grid8TransitionsEnum.NORTH) == (0, 0, 0, 0, 0, 0, 0, 0)
    grid8_map.set_transition((0, 0, Grid8TransitionsEnum.NORTH), Grid8TransitionsEnum.NORTH, 1)
    assert grid8_map.get_transitions(0, 0, Grid8TransitionsEnum.NORTH) == (1, 0, 0, 0, 0, 0, 0, 0)
    grid8_map.set_transition((0, 0, Grid8TransitionsEnum.NORTH), Grid8TransitionsEnum.NORTH, 0)
    assert grid8_map.get_transitions(0, 0, Grid8TransitionsEnum.NORTH) == (0, 0, 0, 0, 0, 0, 0, 0)


def check_path(env, rail, position, direction, target, expected, rendering=False):
    agent = env.agents[0]
    agent.position = position  # south dead-end
    agent.direction = direction  # north
    agent.target = target  # east dead-end
    agent.moving = True
    if rendering:
        renderer = RenderTool(env, gl="PILSVG")
        renderer.render_env(show=True, show_observations=False)
        input("Continue?")
    assert rail.check_path_exists(agent.position, agent.direction, agent.target) == expected


def test_path_exists(rendering=False):
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  )
    env.reset()

    check_path(
        env,
        rail,
        (5, 6),  # north of south dead-end
        0,  # north
        (3, 9),  # east dead-end
        True
    )

    check_path(
        env,
        rail,
        (6, 6),  # south dead-end
        2,  # south
        (3, 9),  # east dead-end
        True
    )

    check_path(
        env,
        rail,
        (3, 0),  # east dead-end
        3,  # west
        (0, 3),  # north dead-end
        True
    )
    check_path(
        env,
        rail,
        (5, 6),  # east dead-end
        0,  # west
        (1, 3),  # north dead-end
        True)

    check_path(
        env,
        rail,
        (1, 3),  # east dead-end
        2,  # south
        (3, 3),  # north dead-end
        True
    )

    check_path(
        env,
        rail,
        (1, 3),  # east dead-end
        0,  # north
        (3, 3),  # north dead-end
        True
    )


def test_path_not_exists(rendering=False):
    rail, rail_map = make_simple_rail_unconnected()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  )
    env.reset()

    check_path(
        env,
        rail,
        (5, 6),  # south dead-end
        0,  # north
        (0, 3),  # north dead-end
        False
    )

    if rendering:
        renderer = RenderTool(env, gl="PILSVG")
        renderer.render_env(show=True, show_observations=False)
        input("Continue?")


def test_get_entry_directions():
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    vertical_line = cells[1]
    south_symmetrical_switch = cells[6]
    north_symmetrical_switch = transitions.rotate_transition(south_symmetrical_switch, 180)

    south_east_turn = int('0100000000000010', 2)
    south_west_turn = transitions.rotate_transition(south_east_turn, 90)
    north_east_turn = transitions.rotate_transition(south_east_turn, 270)
    north_west_turn = transitions.rotate_transition(south_east_turn, 180)

    def _assert(transition, expected):
        actual = Grid4Transitions.get_entry_directions(transition)
        assert actual == expected, "Found {}, expected {}.".format(actual, expected)

    _assert(south_east_turn, [True, False, False, True])
    _assert(south_west_turn, [True, True, False, False])
    _assert(north_east_turn, [False, False, True, True])
    _assert(north_west_turn, [False, True, True, False])
    _assert(vertical_line, [True, False, True, False])
    _assert(south_symmetrical_switch, [True, True, False, True])
    _assert(north_symmetrical_switch, [False, True, True, True])
