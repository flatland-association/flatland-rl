#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flatland.core.env import RailEnv
from flatland.core.transitions import GridTransitions
import numpy as np

"""Tests for `flatland` package."""



def test_rail_environment():
    cells = [int('0000000000000000', 2),  # empty cell - Case 0
                       int('1000000000100000', 2),  # Case 1 - straight
                       int('1001001000000000', 2),  # Case 2 - simple switch
                       int('1000010000100001', 2),  # Case 3 - diamond drossing
                       int('1001011000100001', 2),  # Case 4 - single slip switch
                       int('1100110000110011', 2),  # Case 5 - double slip switch
                       int('0101001000000010', 2),  # Case 6 - symmetrical switch
                       int('0010000000000000', 2)]  # Case 7 - dead end

    # We instantiate the following map on a 3x3 grid
    #  _  _
    # / \/ \
    # | |  |
    # \_/\_/

    transitions = GridTransitions([], False)
    vertical_line = cells[1]
    south_symmetrical_switch = cells[6]
    north_symmetrical_switch = transitions.rotate_transition(south_symmetrical_switch, 180)
    south_east_turn = int('0100000000100000', 2)  # Simple turn not in the base transitions ?
    south_west_turn = transitions.rotate_transition(south_east_turn, 90)
    north_east_turn = transitions.rotate_transition(south_east_turn, 270)
    north_west_turn = transitions.rotate_transition(south_east_turn, 180)

    rail_map = np.array([[south_east_turn, south_symmetrical_switch, south_west_turn],
                    [vertical_line, vertical_line, vertical_line],
                    [north_east_turn, north_symmetrical_switch, north_west_turn]],
                   dtype=np.uint16)

    rail_env = RailEnv(rail_map, number_of_agents=1)

    # Check that trains are always initialized at a consistent position / direction.
    # They should always be able to go somewhere.
    for _ in range(1000):
        obs = rail_env.reset()
        assert(transitions.get_transitions_from_orientation(
            rail_map[rail_env.agents_position[0]],
            rail_env.agents_direction[0]) != (0, 0, 0, 0))

