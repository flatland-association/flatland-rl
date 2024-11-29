from enum import IntEnum

from flatland.core.grid.grid4 import Grid4Transitions
from flatland.utils.ordered_set import OrderedSet


class RailEnvTransitions(Grid4Transitions):
    """
    Special case of `GridTransitions` over a 2D-grid, with a pre-defined set
    of transitions mimicking the types of real Swiss rail connections.

    As no diagonal transitions are allowed in the RailEnv environment, the
    possible transitions for RailEnv from a cell to its neighboring ones
    are represented over 16 bits.

    The 16 bits are organized in 4 blocks of 4 bits each, the direction that
    the agent is facing.
    E.g., the most-significant 4-bits represent the possible movements (NESW)
    if the agent is facing North, etc...

    agent's direction:          North    East   South   West
    agent's allowed movements:  [nesw]   [nesw] [nesw]  [nesw]
    example:                     1000     0000   0010    0000

    In the example, the agent can move from North to South and viceversa.
    """

    # Contains the basic transitions;
    # the set of all 30 valid transitions is obtained by successive 90-degree rotation of one of these basic transitions.
    transition_list = [int('0000000000000000', 2),  # Case 0 - empty cell (1)
                       int('1000000000100000', 2),  # Case 1 - straight (2)
                       int('1001001000100000', 2),  # Case 2 - simple switch left (4)
                       int('1000010000100001', 2),  # Case 3 - diamond crossing (1)
                       int('1001011000100001', 2),  # Case 4 - single slip (4)
                       int('1100110000110011', 2),  # Case 5 - double slip (2)
                       int('0101001000000010', 2),  # Case 6 - symmetrical switch (4)
                       int('0010000000000000', 2),  # Case 7 - dead end (4)
                       int('0100000000000010', 2),  # Case 1b (8)  - simple turn right (4)
                       int('0001001000000000', 2),  # Case 1c (9)  - simple turn left (- same as Case 1b)
                       int('1100000000100010', 2)]  # Case 2b (10) - simple switch right (4)

    def __init__(self):
        super(RailEnvTransitions, self).__init__(
            transitions=self.transition_list
        )

        # create this to make validation faster
        self.transitions_all = OrderedSet()
        for index, trans in enumerate(self.transitions):
            self.transitions_all.add(trans)
            if index in (2, 4, 6, 7, 8, 9, 10):
                for _ in range(3):
                    trans = self.rotate_transition(trans, rotation=90)
                    self.transitions_all.add(trans)
            elif index in (1, 5):
                trans = self.rotate_transition(trans, rotation=90)
                self.transitions_all.add(trans)

    def print(self, cell_transition):
        print("  NESW")
        print("N", format(cell_transition >> (3 * 4) & 0xF, '04b'))
        print("E", format(cell_transition >> (2 * 4) & 0xF, '04b'))
        print("S", format(cell_transition >> (1 * 4) & 0xF, '04b'))
        print("W", format(cell_transition >> (0 * 4) & 0xF, '04b'))

    def is_valid(self, cell_transition):
        """
        Checks if a cell transition is a valid cell setup.

        Parameters
        ----------
        cell_transition : int
            64 bits used to encode the valid transitions for a cell.

        Returns
        -------
        Boolean
            True or False
        """
        return cell_transition in self.transitions_all


# TODO not complete
class RailEnvTransitionsEnum(IntEnum):
    # Case 0 - empty cell (1)
    empty = RailEnvTransitions().transition_list[0]

    # Case 1 - straight (2)
    vertical_straight = RailEnvTransitions().transition_list[1]
    horizontal_straight = RailEnvTransitions().rotate_transition(RailEnvTransitions().transition_list[1], 90)

    # Case 2 - simple switch left (4)
    simple_switch_north_left = RailEnvTransitions().transition_list[2]

    # Case 3 - diamond crossing (1)
    # Case 4 - single slip (4)
    # Case 5 - double slip (2)

    # Case 6 - symmetrical (4)
    #   NESW
    # N 0101
    # E 0010
    # S 0000
    # W 0010
    symmetric_switch_south = RailEnvTransitions().transition_list[6]
    symmetric_switch_north = RailEnvTransitions().rotate_transition(symmetric_switch_south, 180)

    # Case 7 - dead end (4)
    dead_end_from_south = RailEnvTransitions().transition_list[7]
    dead_end_from_west = RailEnvTransitions().rotate_transition(dead_end_from_south, 90)
    dead_end_from_north = RailEnvTransitions().rotate_transition(dead_end_from_south, 180)
    dead_end_from_east = RailEnvTransitions().rotate_transition(dead_end_from_south, 270)

    # Case 1b/1c (8)/(9)  - simple turn  (4)
    right_turn_from_south = RailEnvTransitions().transition_list[8]

    right_turn_from_west = RailEnvTransitions().rotate_transition(right_turn_from_south, 90)
    right_turn_from_north = RailEnvTransitions().rotate_transition(right_turn_from_south, 180)

    # Case 2b (10) - simple switch right (4)
    simple_switch_north_right = RailEnvTransitions().transition_list[10]
    simple_switch_left_east = RailEnvTransitions().rotate_transition(simple_switch_north_left, 90)
