#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `flatland` package."""
from flatland.core.transitions import RailEnvTransitions, Grid8Transitions
from flatland.envs.rail_env import validate_new_transition
import numpy as np


def test_is_valid_railenv_transitions():
    rail_env_trans = RailEnvTransitions()
    transition_list = rail_env_trans.transitions

    for t in transition_list:
        assert(rail_env_trans.is_valid(t) is True)
        for i in range(3):
            rot_trans = rail_env_trans.rotate_transition(t, 90 * i)
            assert(rail_env_trans.is_valid(rot_trans) is True)

    assert(rail_env_trans.is_valid(int('1111111111110010', 2)) is False)
    assert(rail_env_trans.is_valid(int('1001111111110010', 2)) is False)
    assert(rail_env_trans.is_valid(int('1001111001110110', 2)) is False)


def test_adding_new_valid_transition():
    rail_trans = RailEnvTransitions()
    rail_array = np.zeros(shape=(15, 15), dtype=np.uint16)

    # adding straight
    assert(validate_new_transition(rail_trans, rail_array, (4, 5), (5, 5), (6, 5), (10, 10)) is True)

    # adding valid right turn
    assert(validate_new_transition(rail_trans, rail_array, (5, 4), (5, 5), (5, 6), (10, 10)) is True)
    # adding valid left turn
    assert(validate_new_transition(rail_trans, rail_array, (5, 6), (5, 5), (5, 6), (10, 10)) is True)

    # adding invalid turn
    rail_array[(5, 5)] = rail_trans.transitions[2]
    assert(validate_new_transition(rail_trans, rail_array, (4, 5), (5, 5), (5, 6), (10, 10)) is False)

    # should create #4 -> valid
    rail_array[(5, 5)] = rail_trans.transitions[3]
    assert(validate_new_transition(rail_trans, rail_array, (4, 5), (5, 5), (5, 6), (10, 10)) is True)

    # adding invalid turn
    rail_array[(5, 5)] = rail_trans.transitions[7]
    assert(validate_new_transition(rail_trans, rail_array, (4, 5), (5, 5), (5, 6), (10, 10)) is False)

    # test path start condition
    rail_array[(5, 5)] = rail_trans.transitions[0]
    assert(validate_new_transition(rail_trans, rail_array, None, (5, 5), (5, 6), (10, 10)) is True)

    # test path end condition
    rail_array[(5, 5)] = rail_trans.transitions[0]
    assert(validate_new_transition(rail_trans, rail_array, (5, 4), (5, 5), (6, 5), (6, 5)) is True)


def test_valid_railenv_transitions():
    rail_env_trans = RailEnvTransitions()

    # dir_map = {'N': 0,
    #            'E': 1,
    #            'S': 2,
    #            'W': 3}

    for i in range(2):
        assert(rail_env_trans.get_transitions(
               int('1100110000110011', 2), i) == (1, 1, 0, 0))
        assert(rail_env_trans.get_transitions(
               int('1100110000110011', 2), 2 + i) == (0, 0, 1, 1))

    no_transition_cell = int('0000000000000000', 2)

    for i in range(4):
        assert(rail_env_trans.get_transitions(
               no_transition_cell, i) == (0, 0, 0, 0))

    # Facing south, going south
    north_south_transition = rail_env_trans.set_transitions(no_transition_cell, 2, (0, 0, 1, 0))
    assert(rail_env_trans.set_transition(
           north_south_transition, 2, 2, 0) == no_transition_cell)
    assert(rail_env_trans.get_transition(
           north_south_transition, 2, 2))

    # Facing north, going east
    south_east_transition = \
        rail_env_trans.set_transition(no_transition_cell, 0, 1, 1)
    assert(rail_env_trans.get_transition(
           south_east_transition, 0, 1))

    # The opposite transitions are not feasible
    assert(not rail_env_trans.get_transition(
           north_south_transition, 2, 0))
    assert(not rail_env_trans.get_transition(
           south_east_transition, 2, 1))

    east_west_transition = rail_env_trans.rotate_transition(north_south_transition, 90)
    north_west_transition = rail_env_trans.rotate_transition(south_east_transition, 180)

    # Facing west, going west
    assert(rail_env_trans.get_transition(
           east_west_transition, 3, 3))
    # Facing south, going west
    assert(rail_env_trans.get_transition(
           north_west_transition, 2, 3))

    assert(south_east_transition == rail_env_trans.rotate_transition(
           south_east_transition, 360))


def test_diagonal_transitions():
    diagonal_trans_env = Grid8Transitions([])

    # Facing north, going north-east
    south_northeast_transition = int('01000000' + '0' * 8 * 7, 2)
    assert(diagonal_trans_env.get_transitions(
           south_northeast_transition, 0) == (0, 1, 0, 0, 0, 0, 0, 0))

    # Allowing transition from north to southwest: Facing south, going SW
    north_southwest_transition = \
        diagonal_trans_env.set_transitions(int('0' * 64, 2), 4, (0, 0, 0, 0, 0, 1, 0, 0))

    assert(diagonal_trans_env.rotate_transition(
           south_northeast_transition, 180) == north_southwest_transition)
