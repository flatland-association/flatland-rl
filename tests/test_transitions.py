#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `flatland` package."""
from flatland.core.transitions import RailEnvTransitions, Grid8Transitions


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
                    int('1100110000110011', 2), 2+i) == (0, 0, 1, 1))

    no_transition_cell = int('0000000000000000', 2)

    for i in range(4):
        assert(rail_env_trans.get_transitions(
                    no_transition_cell, i) == (0, 0, 0, 0))

    # Facing south, going south
    north_south_transition = rail_env_trans.set_transitions(
                    no_transition_cell, 2, (0, 0, 1, 0))
    assert(rail_env_trans.set_transition(
                    north_south_transition, 2, 2, 0) == no_transition_cell)
    assert(rail_env_trans.get_transition(
                    north_south_transition, 2, 2))

    # Facing north, going east
    south_east_transition = \
        rail_env_trans.set_transition(
         no_transition_cell, 0, 1, 1)
    assert(rail_env_trans.get_transition(
            south_east_transition, 0, 1))

    # The opposite transitions are not feasible
    assert(not rail_env_trans.get_transition(
            north_south_transition, 2, 0))
    assert(not rail_env_trans.get_transition(
            south_east_transition, 2, 1))

    east_west_transition = rail_env_trans.rotate_transition(
            north_south_transition, 90)
    north_west_transition = rail_env_trans.rotate_transition(
            south_east_transition, 180)

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
    south_northeast_transition = int('01000000' + '0'*8*7, 2)
    assert(diagonal_trans_env.get_transitions(
            south_northeast_transition, 0) == (0, 1, 0, 0, 0, 0, 0, 0))

    # Allowing transition from north to southwest: Facing south, going SW
    north_southwest_transition = \
        diagonal_trans_env.set_transitions(
         int('0' * 64, 2), 4, (0, 0, 0, 0, 0, 1, 0, 0))

    assert(diagonal_trans_env.rotate_transition(
            south_northeast_transition, 180) == north_southwest_transition)
