#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `flatland` package."""
import numpy as np

from flatland.core.grid.grid4 import Grid4Transitions
from flatland.core.grid.grid8 import Grid8Transitions
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.grid.grid4_utils import validate_new_transition


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
    rail_array = np.zeros(shape=(15, 15), dtype=np.uint16)

    # adding straight
    assert (validate_new_transition(rail_trans, rail_array, (4, 5), (5, 5), (6, 5), (10, 10)) is True)

    # adding valid right turn
    assert (validate_new_transition(rail_trans, rail_array, (5, 4), (5, 5), (5, 6), (10, 10)) is True)
    # adding valid left turn
    assert (validate_new_transition(rail_trans, rail_array, (5, 6), (5, 5), (5, 6), (10, 10)) is True)

    # adding invalid turn
    rail_array[(5, 5)] = rail_trans.transitions[2]
    assert (validate_new_transition(rail_trans, rail_array, (4, 5), (5, 5), (5, 6), (10, 10)) is False)

    # should create #4 -> valid
    rail_array[(5, 5)] = rail_trans.transitions[3]
    assert (validate_new_transition(rail_trans, rail_array, (4, 5), (5, 5), (5, 6), (10, 10)) is True)

    # adding invalid turn
    rail_array[(5, 5)] = rail_trans.transitions[7]
    assert (validate_new_transition(rail_trans, rail_array, (4, 5), (5, 5), (5, 6), (10, 10)) is False)

    # test path start condition
    rail_array[(5, 5)] = rail_trans.transitions[0]
    assert (validate_new_transition(rail_trans, rail_array, None, (5, 5), (5, 6), (10, 10)) is True)

    # test path end condition
    rail_array[(5, 5)] = rail_trans.transitions[0]
    assert (validate_new_transition(rail_trans, rail_array, (5, 4), (5, 5), (6, 5), (6, 5)) is True)


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
        actual_had_deadend = ret.has_deadend(t)
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
