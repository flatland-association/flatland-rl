from typing import Tuple, Dict, Any

import numpy as np

from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap


def make_simple_rail() -> Tuple[GridTransitionMap, np.array, Dict]:
    # We instantiate a very simple rail network on a 7x10 grid:
    # Note that that cells have invalid RailEnvTransitions!
    #        |
    #        |
    #        |
    #    _ _ _ _\ _ _  _  _ _ _
    #                /
    #                |
    #                |
    #                |
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    empty = cells[0]
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    simple_switch_north_left = cells[2]
    simple_switch_north_right = cells[10]
    simple_switch_east_west_north = transitions.rotate_transition(simple_switch_north_right, 270)
    simple_switch_east_west_south = transitions.rotate_transition(simple_switch_north_left, 270)
    rail_map = np.array(
        [[empty] * 3 + [dead_end_from_south] + [empty] * 6] +
        [[empty] * 3 + [vertical_straight] + [empty] * 6] * 2 +
        [[dead_end_from_east] + [horizontal_straight] * 2 +
         [simple_switch_east_west_north] +
         [horizontal_straight] * 2 + [simple_switch_east_west_south] +
         [horizontal_straight] * 2 + [dead_end_from_west]] +
        [[empty] * 6 + [vertical_straight] + [empty] * 3] * 2 +
        [[empty] * 6 + [dead_end_from_north] + [empty] * 3], dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    city_positions = [(0,3), (6, 6)]
    train_stations = [
                      [( (0, 3), 0 ) ],
                      [( (6, 6), 0 ) ],
                     ]
    city_orientations = [0, 2]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                   }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals


def make_disconnected_simple_rail() -> Tuple[GridTransitionMap, np.array, Dict]:
    # We instantiate a very simple rail network on a 7x10 grid:
    # Note that that cells have invalid RailEnvTransitions!
    #        |
    #        |
    #        |
    # _ _ _ _\ _    _  _ _ _
    #                /
    #                |
    #                |
    #                |

    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    empty = cells[0]
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    simple_switch_north_left = cells[2]
    simple_switch_north_right = cells[10]
    simple_switch_east_west_north = transitions.rotate_transition(simple_switch_north_right, 270)
    simple_switch_east_west_south = transitions.rotate_transition(simple_switch_north_left, 270)
    rail_map = np.array(
        [[empty] * 3 + [dead_end_from_south] + [empty] * 6] +
        [[empty] * 3 + [vertical_straight] + [empty] * 6] * 2 +
        [[dead_end_from_east] + [horizontal_straight] * 2 +
         [simple_switch_east_west_north] +
         [dead_end_from_west] + [dead_end_from_east] + [simple_switch_east_west_south] +
         [horizontal_straight] * 2 + [dead_end_from_west]] +
        [[empty] * 6 + [vertical_straight] + [empty] * 3] * 2 +
        [[empty] * 6 + [dead_end_from_north] + [empty] * 3], dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    city_positions = [(0,3), (6, 6)]
    train_stations = [
                      [( (0, 3), 0 ) ],
                      [( (6, 6), 0 ) ],
                     ]
    city_orientations = [0, 2]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                   }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals


def make_simple_rail2() -> Tuple[GridTransitionMap, np.array, Dict]:
    # We instantiate a very simple rail network on a 7x10 grid:
    #        |
    #        |
    #        |
    # _ _ _ _\ _ _  _  _ _ _
    #               \
    #                |
    #                |
    #                |
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    empty = cells[0]
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    simple_switch_north_right = cells[10]
    simple_switch_east_west_north = transitions.rotate_transition(simple_switch_north_right, 270)
    simple_switch_west_east_south = transitions.rotate_transition(simple_switch_north_right, 90)
    rail_map = np.array(
        [[empty] * 3 + [dead_end_from_south] + [empty] * 6] +
        [[empty] * 3 + [vertical_straight] + [empty] * 6] * 2 +
        [[dead_end_from_east] + [horizontal_straight] * 2 +
         [simple_switch_east_west_north] +
         [horizontal_straight] * 2 + [simple_switch_west_east_south] +
         [horizontal_straight] * 2 + [dead_end_from_west]] +
        [[empty] * 6 + [vertical_straight] + [empty] * 3] * 2 +
        [[empty] * 6 + [dead_end_from_north] + [empty] * 3], dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    city_positions = [(0,3), (6, 6)]
    train_stations = [
                      [( (0, 3), 0 ) ],
                      [( (6, 6), 0 ) ],
                     ]
    city_orientations = [0, 2]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                   }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals


def make_simple_rail_unconnected() -> Tuple[GridTransitionMap, np.array, Dict]:
    # We instantiate a very simple rail network on a 7x10 grid:
    # Note that that cells have invalid RailEnvTransitions!
    #        |
    #        |
    #        |
    # _ _ _  _ _ _  _  _ _ _
    #                /
    #                |
    #                |
    #                |
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    empty = cells[0]
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    simple_switch_north_left = cells[2]
    # simple_switch_north_right = cells[10]
    # simple_switch_east_west_north = transitions.rotate_transition(simple_switch_north_right, 270)
    simple_switch_east_west_south = transitions.rotate_transition(simple_switch_north_left, 270)
    rail_map = np.array(
        [[empty] * 3 + [dead_end_from_south] + [empty] * 6] +
        [[empty] * 3 + [vertical_straight] + [empty] * 6] +
        [[empty] * 3 + [dead_end_from_north] + [empty] * 6] +
        [[dead_end_from_east] + [horizontal_straight] * 5 + [simple_switch_east_west_south] +
         [horizontal_straight] * 2 + [dead_end_from_west]] +
        [[empty] * 6 + [vertical_straight] + [empty] * 3] * 2 +
        [[empty] * 6 + [dead_end_from_north] + [empty] * 3], dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    city_positions = [(0,3), (6, 6)]
    train_stations = [
                      [( (0, 3), 0 ) ],
                      [( (6, 6), 0 ) ],
                     ]
    city_orientations = [0, 2]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                   }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals


def make_simple_rail_with_alternatives() -> Tuple[GridTransitionMap, np.array, Dict]:
    # We instantiate a very simple rail network on a 7x10 grid:
    #  0 1 2 3 4 5 6 7 8 9  10
    # 0        /-------------\
    # 1        |             |
    # 2        |             |
    # 3 _ _ _ /_  _ _        |
    # 4              \   ___ /
    # 5               |/
    # 6               |
    # 7               |
    transitions = RailEnvTransitions()
    cells = transitions.transition_list

    empty = cells[0]
    dead_end_from_south = cells[7]
    right_turn_from_south = cells[8]
    right_turn_from_west = transitions.rotate_transition(right_turn_from_south, 90)
    right_turn_from_north = transitions.rotate_transition(right_turn_from_south, 180)
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    simple_switch_north_left = cells[2]
    simple_switch_north_right = cells[10]
    simple_switch_left_east = transitions.rotate_transition(simple_switch_north_left, 90)
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    double_switch_south_horizontal_straight = horizontal_straight + cells[6]
    double_switch_north_horizontal_straight = transitions.rotate_transition(
        double_switch_south_horizontal_straight, 180)
    rail_map = np.array(
        [[empty] * 3 + [right_turn_from_south] + [horizontal_straight] * 5 + [right_turn_from_west]] +
        [[empty] * 3 + [vertical_straight] + [empty] * 5 + [vertical_straight]] * 2 +
        [[dead_end_from_east] + [horizontal_straight] * 2 + [simple_switch_left_east] + [horizontal_straight] * 2 + [
            right_turn_from_west] + [empty] * 2 + [vertical_straight]] +
        [[empty] * 6 + [simple_switch_north_right] + [horizontal_straight] * 2 + [right_turn_from_north]] +
        [[empty] * 6 + [vertical_straight] + [empty] * 3] +
        [[empty] * 6 + [dead_end_from_north] + [empty] * 3], dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    city_positions = [(0,3), (6, 6)]
    train_stations = [
                      [( (0, 3), 0 ) ],
                      [( (6, 6), 0 ) ],
                     ]
    city_orientations = [0, 2]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                   }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals


def make_invalid_simple_rail() -> Tuple[GridTransitionMap, np.array, Dict[str, Any]]:
    # We instantiate a very simple rail network on a 7x10 grid:
    #        |
    #        |
    #        |
    # _ _ _ /_\ _ _  _  _ _ _
    #               \ /
    #                |
    #                |
    #                |
    transitions = RailEnvTransitions()
    cells = transitions.transition_list

    empty = cells[0]
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    double_switch_south_horizontal_straight = horizontal_straight + cells[6]
    double_switch_north_horizontal_straight = transitions.rotate_transition(
        double_switch_south_horizontal_straight, 180)
    rail_map = np.array(
        [[empty] * 3 + [dead_end_from_south] + [empty] * 6] +
        [[empty] * 3 + [vertical_straight] + [empty] * 6] * 2 +
        [[dead_end_from_east] + [horizontal_straight] * 2 +
         [double_switch_north_horizontal_straight] +
         [horizontal_straight] * 2 + [double_switch_south_horizontal_straight] +
         [horizontal_straight] * 2 + [dead_end_from_west]] +
        [[empty] * 6 + [vertical_straight] + [empty] * 3] * 2 +
        [[empty] * 6 + [dead_end_from_north] + [empty] * 3], dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    city_positions = [(0,3), (6, 6)]
    train_stations = [
                      [( (0, 3), 0 ) ],
                      [( (6, 6), 0 ) ],
                     ]
    city_orientations = [0, 2]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                   }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals

def make_oval_rail() -> Tuple[GridTransitionMap, np.array]:
    transitions = RailEnvTransitions()
    cells = transitions.transition_list

    empty = cells[0]
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    right_turn_from_south = cells[8]
    right_turn_from_west = transitions.rotate_transition(right_turn_from_south, 90)
    right_turn_from_north = transitions.rotate_transition(right_turn_from_south, 180)
    right_turn_from_east = transitions.rotate_transition(right_turn_from_south, 270)

    rail_map = np.array(
        [[empty] * 9] +
        [[empty] + [right_turn_from_south] + [horizontal_straight] * 5 + [right_turn_from_west] + [empty]] +
        [[empty] + [vertical_straight] + [empty] * 5 + [vertical_straight] + [empty]]+
        [[empty] + [vertical_straight] + [empty] * 5 + [vertical_straight] + [empty]] +
        [[empty] + [right_turn_from_east] + [horizontal_straight] * 5 + [right_turn_from_north] + [empty]] +
        [[empty] * 9], dtype=np.uint16)

    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    city_positions = [(1, 4), (4, 4)]
    train_stations = [
        [((1, 4), 0)],
        [((4, 4), 0)],
    ]
    city_orientations = [1, 3]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                    }
    optionals = {'agents_hints': agents_hints}
    return  rail, rail_map, optionals
