#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `flatland` package."""

from flatland.core.grid.grid4 import Grid4Transitions
from flatland.core.grid.grid8 import Grid8Transitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.grid.rail_env_grid import RailEnvTransitionsEnum


# remove whitespace in string; keep whitespace below for easier reading
def rw(s):
    return s.replace(" ", "")


def test_rotate_railenv_transition():
    rail_env_transitions = RailEnvTransitions()

    # TODO test all cases
    transition_cycles = [
        # empty cell - Case 0
        [int('0000000000000000', 2), int('0000000000000000', 2), int('0000000000000000', 2),
         int('0000000000000000', 2)],
        # Case 1 - straight
        #     |
        #     |
        #     |
        [int(rw('1000 0000 0010 0000'), 2), int(rw('0000 0100 0000 0001'), 2)],
        # Case 1b (8)  - simple turn right
        #      _
        #     |
        #     |
        [
            int(rw('0100 0000 0000 0010'), 2),
            int(rw('0001 0010 0000 0000'), 2),
            int(rw('0000 1000 0001 0000'), 2),
            int(rw('0000 0000 0100 1000'), 2),
        ],
        # Case 1c (9)  - simple turn left
        #    _
        #     |
        #     |

        # int('0001001000000000', 2),\ #  noqa: E800

        # Case 2 - simple left switch
        #  _ _|
        #     |
        #     |
        [
            int(rw('1001 0010 0010 0000'), 2),
            int(rw('0000 1100 0001 0001'), 2),
            int(rw('1000 0000 0110 1000'), 2),
            int(rw('0100 0100 0000 0011'), 2),
        ],
        # Case 2b (10) - simple right switch
        #     |
        #     |
        #     |

        # int('1100000000100010', 2) \ #  noqa: E800

        # Case 3 - diamond drossing
        # int('1000010000100001', 2),   \ #  noqa: E800
        # Case 4 - single slip
        # int('1001011000100001', 2),   \ #  noqa: E800
        # Case 5 - double slip
        # int('1100110000110011', 2),   \ #  noqa: E800
        # Case 6 - symmetrical
        # int('0101001000000010', 2),   \ #  noqa: E800

        # Case 7 - dead end
        #
        #
        #     |
        [
            int(rw('0010 0000 0000 0000'), 2),
            int(rw('0000 0001 0000 0000'), 2),
            int(rw('0000 0000 1000 0000'), 2),
            int(rw('0000 0000 0000 0100'), 2),
        ],
    ]

    for index, cycle in enumerate(transition_cycles):
        for i in range(4):
            actual_transition = rail_env_transitions.rotate_transition(cycle[0], i * 90)
            expected_transition = cycle[i % len(cycle)]
            try:
                assert actual_transition == expected_transition, \
                    "Case {}: rotate_transition({}, {}) should equal {} but was {}.".format(
                        i, cycle[0], i, expected_transition, actual_transition)
            except Exception as e:
                print("expected:")
                rail_env_transitions.print(expected_transition)
                print("actual:")
                rail_env_transitions.print(actual_transition)

                raise e


def test_is_valid_railenv_transitions():
    rail_env_trans = RailEnvTransitions()
    transition_list = rail_env_trans.transitions

    for t in transition_list:
        assert (rail_env_trans.is_valid(t) is True)
        for i in range(3):
            rot_trans = rail_env_trans.rotate_transition(t, 90 * i)
            assert (rail_env_trans.is_valid(rot_trans) is True)

    assert (rail_env_trans.is_valid(int('1111111111110010', 2)) is False)
    assert (rail_env_trans.is_valid(int('1001111111110010', 2)) is False)
    assert (rail_env_trans.is_valid(int('1001111001110110', 2)) is False)


def test_adding_new_valid_transition():
    rail_trans = RailEnvTransitions()
    grid_map = GridTransitionMap(width=15, height=15, transitions=rail_trans)

    # adding straight
    assert (grid_map.validate_new_transition((4, 5), (5, 5), (6, 5), (10, 10)) is True)

    # adding valid right turn
    assert (grid_map.validate_new_transition((5, 4), (5, 5), (5, 6), (10, 10)) is True)
    # adding valid left turn
    assert (grid_map.validate_new_transition((5, 6), (5, 5), (5, 6), (10, 10)) is True)

    # adding invalid turn
    grid_map.set_transitions((5, 5), rail_trans.transitions[2])
    assert (grid_map.validate_new_transition((4, 5), (5, 5), (5, 6), (10, 10)) is False)

    # should create #4 -> valid
    grid_map.set_transitions((5, 5), rail_trans.transitions[3])
    assert (grid_map.validate_new_transition((4, 5), (5, 5), (5, 6), (10, 10)) is True)

    # adding invalid turn
    grid_map.set_transitions((5, 5), rail_trans.transitions[7])
    assert (grid_map.validate_new_transition((4, 5), (5, 5), (5, 6), (10, 10)) is False)

    # test path start condition
    grid_map.set_transitions((5, 5), rail_trans.transitions[3])
    assert (grid_map.validate_new_transition(None, (5, 5), (5, 6), (10, 10)) is True)

    # test path end condition
    grid_map.set_transitions((5, 5), rail_trans.transitions[3])
    assert (grid_map.validate_new_transition((5, 4), (5, 5), (6, 5), (6, 5)) is True)


def test_valid_railenv_transitions():
    rail_env_trans = RailEnvTransitions()

    # directions:
    #            'N': 0
    #            'E': 1
    #            'S': 2
    #            'W': 3

    for i in range(2):
        assert (rail_env_trans.get_transitions(
            int('1100110000110011', 2), i) == (1, 1, 0, 0))
        assert (rail_env_trans.get_transitions(
            int('1100110000110011', 2), 2 + i) == (0, 0, 1, 1))

    no_transition_cell = int('0000000000000000', 2)

    for i in range(4):
        assert (rail_env_trans.get_transitions(
            no_transition_cell, i) == (0, 0, 0, 0))

    # Facing south, going south
    north_south_transition = rail_env_trans.set_transitions(no_transition_cell, 2, (0, 0, 1, 0))
    assert (rail_env_trans.set_transition(
        north_south_transition, 2, 2, 0) == no_transition_cell)
    assert (rail_env_trans.get_transition(
        north_south_transition, 2, 2))

    # Facing north, going east
    south_east_transition = \
        rail_env_trans.set_transition(no_transition_cell, 0, 1, 1)
    assert (rail_env_trans.get_transition(
        south_east_transition, 0, 1))

    # The opposite transitions are not feasible
    assert (not rail_env_trans.get_transition(
        north_south_transition, 2, 0))
    assert (not rail_env_trans.get_transition(
        south_east_transition, 2, 1))

    east_west_transition = rail_env_trans.rotate_transition(north_south_transition, 90)
    north_west_transition = rail_env_trans.rotate_transition(south_east_transition, 180)

    # Facing west, going west
    assert (rail_env_trans.get_transition(
        east_west_transition, 3, 3))
    # Facing south, going west
    assert (rail_env_trans.get_transition(
        north_west_transition, 2, 3))

    assert (south_east_transition == rail_env_trans.rotate_transition(
        south_east_transition, 360))


def test_diagonal_transitions():
    diagonal_trans_env = Grid8Transitions([])

    # Facing north, going north-east
    south_northeast_transition = int('01000000' + '0' * 8 * 7, 2)
    assert (diagonal_trans_env.get_transitions(
        south_northeast_transition, 0) == (0, 1, 0, 0, 0, 0, 0, 0))

    # Allowing transition from north to southwest: Facing south, going SW
    north_southwest_transition = \
        diagonal_trans_env.set_transitions(0, 4, (0, 0, 0, 0, 0, 1, 0, 0))

    assert (diagonal_trans_env.rotate_transition(
        south_northeast_transition, 180) == north_southwest_transition)


def test_rail_env_has_deadend():
    deadends = set([int(rw('0010 0000 0000 0000'), 2),
                    int(rw('0000 0001 0000 0000'), 2),
                    int(rw('0000 0000 1000 0000'), 2),
                    int(rw('0000 0000 0000 0100'), 2)])
    ret = RailEnvTransitions()
    transitions_all = ret.transitions_all
    for t in transitions_all:
        expected_has_deadend = t in deadends
        actual_had_deadend = Grid4Transitions.has_deadend(t)
        assert actual_had_deadend == expected_has_deadend, \
            "{} should be deadend = {}, actual = {}".format(t, )


def test_rail_env_remove_deadend():
    ret = Grid4Transitions([])
    rail_env_deadends = set([int(rw('0010 0000 0000 0000'), 2),
                             int(rw('0000 0001 0000 0000'), 2),
                             int(rw('0000 0000 1000 0000'), 2),
                             int(rw('0000 0000 0000 0100'), 2)])
    for t in rail_env_deadends:
        expected_has_deadend = 0
        actual_had_deadend = ret.remove_deadends(t)
        assert actual_had_deadend == expected_has_deadend, \
            "{} should be deadend = {}, actual = {}".format(t, )

    assert ret.remove_deadends(int(rw('0010 0001 1000 0100'), 2)) == 0
    assert ret.remove_deadends(int(rw('0010 0001 1000 0110'), 2)) == int(rw('0000 0000 0000 0010'), 2)


def test_get_neighboring_cells_horizontal_straight():
    rail = GridTransitionMap(3, 3)
    rail.set_transitions((1, 1), RailEnvTransitionsEnum.horizontal_straight)
    pairs = rail.get_neighbor_pairs((1, 1))
    assert ((1, 0), (1, 2)) in pairs
    assert ((1, 2), (1, 0)) in pairs
    assert len(pairs) == 2


def test_is_deadend():
    for member in [RailEnvTransitionsEnum.dead_end_from_east, RailEnvTransitionsEnum.dead_end_from_south,
                   RailEnvTransitionsEnum.dead_end_from_north, RailEnvTransitionsEnum.dead_end_from_west]:
        assert RailEnvTransitionsEnum.is_deadend(member), member.name
    assert not RailEnvTransitionsEnum.is_deadend(RailEnvTransitionsEnum.vertical_straight)


def test_is_turn():
    for member in [RailEnvTransitionsEnum.right_turn_from_south, RailEnvTransitionsEnum.right_turn_from_west,
                   RailEnvTransitionsEnum.right_turn_from_north, RailEnvTransitionsEnum.right_turn_from_east]:
        assert RailEnvTransitionsEnum.is_turn(member), member.name
    assert not RailEnvTransitionsEnum.is_turn(RailEnvTransitionsEnum.vertical_straight)


def test_is_straight():
    for member in [RailEnvTransitionsEnum.vertical_straight, RailEnvTransitionsEnum.horizontal_straight]:
        assert RailEnvTransitionsEnum.is_straight(member), member.name
    assert not RailEnvTransitionsEnum.is_straight(RailEnvTransitionsEnum.dead_end_from_east)


def test_is_double_slip():
    for member in [RailEnvTransitionsEnum.double_slip_NW_SE, RailEnvTransitionsEnum.double_slip_NE_SW]:
        assert RailEnvTransitionsEnum.is_double_slip(member), member.name
    assert not RailEnvTransitionsEnum.is_double_slip(RailEnvTransitionsEnum.vertical_straight)


def test_is_single_slip():
    for member in [RailEnvTransitionsEnum.single_slip_NE, RailEnvTransitionsEnum.single_slip_SE,
                   RailEnvTransitionsEnum.single_slip_SW, RailEnvTransitionsEnum.single_slip_NW]:
        assert RailEnvTransitionsEnum.is_single_slip(member), member.name
    assert not RailEnvTransitionsEnum.is_single_slip(RailEnvTransitionsEnum.vertical_straight)


def test_is_one_one():
    one_one_members = [
        RailEnvTransitionsEnum.dead_end_from_east, RailEnvTransitionsEnum.dead_end_from_south,
        RailEnvTransitionsEnum.dead_end_from_north, RailEnvTransitionsEnum.dead_end_from_west,
        RailEnvTransitionsEnum.right_turn_from_south, RailEnvTransitionsEnum.right_turn_from_west,
        RailEnvTransitionsEnum.right_turn_from_north, RailEnvTransitionsEnum.right_turn_from_east,
        RailEnvTransitionsEnum.vertical_straight, RailEnvTransitionsEnum.horizontal_straight,
    ]
    for member in one_one_members:
        assert RailEnvTransitionsEnum.is_one_one(member), member.name
    not_one_one_members = [
        RailEnvTransitionsEnum.empty, RailEnvTransitionsEnum.diamond_crossing,
        RailEnvTransitionsEnum.simple_switch_north_left, RailEnvTransitionsEnum.simple_switch_north_right,
        RailEnvTransitionsEnum.single_slip_NE, RailEnvTransitionsEnum.double_slip_NW_SE,
        RailEnvTransitionsEnum.symmetric_switch_from_south,
    ]
    for member in not_one_one_members:
        assert not RailEnvTransitionsEnum.is_one_one(member), member.name


def test_is_simple_switch_all_variants():
    for member in [RailEnvTransitionsEnum.simple_switch_north_left, RailEnvTransitionsEnum.simple_switch_east_left,
                   RailEnvTransitionsEnum.simple_switch_south_left, RailEnvTransitionsEnum.simple_switch_west_left,
                   RailEnvTransitionsEnum.simple_switch_north_right, RailEnvTransitionsEnum.simple_switch_east_right,
                   RailEnvTransitionsEnum.simple_switch_south_right, RailEnvTransitionsEnum.simple_switch_west_right]:
        assert RailEnvTransitionsEnum.is_simple_switch(member), member.name
    assert not RailEnvTransitionsEnum.is_simple_switch(RailEnvTransitionsEnum.vertical_straight)


def test_is_empty():
    assert RailEnvTransitionsEnum.is_empty(RailEnvTransitionsEnum.empty)
    assert not RailEnvTransitionsEnum.is_empty(RailEnvTransitionsEnum.vertical_straight)


def test_is_diamond_crossing():
    assert RailEnvTransitionsEnum.is_diamond_crossing(RailEnvTransitionsEnum.diamond_crossing)
    assert not RailEnvTransitionsEnum.is_diamond_crossing(RailEnvTransitionsEnum.vertical_straight)


def test_is_symmetric_switch():
    for member in [RailEnvTransitionsEnum.symmetric_switch_from_south, RailEnvTransitionsEnum.symmetric_switch_from_west,
                   RailEnvTransitionsEnum.symmetric_switch_from_north, RailEnvTransitionsEnum.symmetric_switch_from_east]:
        assert RailEnvTransitionsEnum.is_symmetric_switch(member), member.name
    assert not RailEnvTransitionsEnum.is_symmetric_switch(RailEnvTransitionsEnum.vertical_straight)


def test_is_methods_partition_all_enum_members():
    """Every RailEnvTransitionsEnum member must be classified by exactly one `is_*` case predicate -
    this is the exhaustive check that would have caught is_simple_switch originally covering
    only 2 of its 8 variants."""
    case_predicates = {
        "is_empty": RailEnvTransitionsEnum.is_empty,
        "is_straight": RailEnvTransitionsEnum.is_straight,
        "is_simple_switch": RailEnvTransitionsEnum.is_simple_switch,
        "is_diamond_crossing": RailEnvTransitionsEnum.is_diamond_crossing,
        "is_single_slip": RailEnvTransitionsEnum.is_single_slip,
        "is_double_slip": RailEnvTransitionsEnum.is_double_slip,
        "is_symmetric_switch": RailEnvTransitionsEnum.is_symmetric_switch,
        "is_deadend": RailEnvTransitionsEnum.is_deadend,
        "is_turn": RailEnvTransitionsEnum.is_turn,
    }
    for member in RailEnvTransitionsEnum:
        matches = [name for name, predicate in case_predicates.items() if predicate(member)]
        assert len(matches) == 1, f"{member.name} matched {matches}, expected exactly one case predicate"
