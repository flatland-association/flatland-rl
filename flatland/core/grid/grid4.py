from enum import IntEnum
from typing import Type, List

import numpy as np

from flatland.core.transitions import Transitions


class Grid4TransitionsEnum(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    @staticmethod
    def to_char(int: int):
        return {0: 'N',
                1: 'E',
                2: 'S',
                3: 'W'}[int]


class Grid4Transitions(Transitions):
    """
    Grid4Transitions class derived from Transitions.

    Special case of `Transitions` over a 2D-grid (FlatLand).
    Transitions are possible to neighboring cells on the grid if allowed.
    GridTransitions keeps track of valid transitions supplied as `transitions`
    list, each represented as a bitmap of 16 bits.

    Whether a transition is allowed or not depends on which direction an agent
    inside the cell is facing (0=North, 1=East, 2=South, 3=West) and which
    direction the agent wants to move to
    (North, East, South, West, relative to the cell).
    Each transition (orientation, direction)
    can be allowed (1) or forbidden (0).

    For example, in case of no diagonal transitions on the grid, the 16 bits
    of the transition bitmaps are organized in 4 blocks of 4 bits each, the
    direction that the agent is facing.
    E.g., the most-significant 4-bits represent the possible movements (NESW)
    if the agent is facing North, etc...

    agent's direction:          North    East   South   West
    agent's allowed movements:  [nesw]   [nesw] [nesw]  [nesw]
    example:                     1000     0000   0010    0000

    In the example, the agent can move from North to South and viceversa.
    """

    def __init__(self, transitions):
        self.transitions = transitions
        self.sDirs = "NESW"
        self.lsDirs = list(self.sDirs)

        # row,col delta for each direction
        self.gDir2dRC = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

        # These bits represent all the possible dead ends
        self.maskDeadEnds = 0b0010000110000100

    def get_type(self):
        return np.uint16

    def get_transitions(self, cell_transition, orientation):
        """
        Get the 4 possible transitions ((N,E,S,W), 4 elements tuple
        if no diagonal transitions allowed) available for an agent oriented
        in direction `orientation` and inside a cell with
        transitions `cell_transition`.

        Parameters
        ----------
        cell_transition : int
            16 bits used to encode the valid transitions for a cell.
        orientation : int
            Orientation of the agent inside the cell.

        Returns
        -------
        tuple
            List of the validity of transitions in the cell.

        """
        bits = (cell_transition >> ((3 - orientation) * 4))
        return ((bits >> 3) & 1, (bits >> 2) & 1, (bits >> 1) & 1, (bits) & 1)

    def set_transitions(self, cell_transition, orientation, new_transitions):
        """
        Set the possible transitions (e.g., (N,E,S,W), 4 elements tuple
        if no diagonal transitions allowed) available for an agent
        oriented in direction `orientation` and inside a cell with transitions
        `cell_transition'. A new `cell_transition` is returned with
        the specified bits replaced by `new_transitions`.

        Parameters
        ----------
        cell_transition : int
            16 bits used to encode the valid transitions for a cell.
        orientation : int
            Orientation of the agent inside the cell.
        new_transitions : tuple
            Tuple of new transitions validitiy for the cell.

        Returns
        -------
        int
            An updated bitmap that replaces the original transitions validity
            of `cell_transition' with `new_transitions`, for the appropriate
            `orientation`.

        """
        mask = (1 << ((4 - orientation) * 4)) - (1 << ((3 - orientation) * 4))
        negmask = ~mask

        new_transitions = \
            (new_transitions[0] & 1) << 3 | \
            (new_transitions[1] & 1) << 2 | \
            (new_transitions[2] & 1) << 1 | \
            (new_transitions[3] & 1)

        cell_transition = (cell_transition & negmask) | (new_transitions << ((3 - orientation) * 4))

        return cell_transition

    def get_transition(self, cell_transition, orientation, direction):
        """
        Get the transition bit (1 value) that determines whether an agent
        oriented in direction `orientation` and inside a cell with transitions
        `cell_transition' can move to the cell in direction `direction`
        relative to the current cell.

        Parameters
        ----------
        cell_transition : int
            16 bits used to encode the valid transitions for a cell.
        orientation : int
            Orientation of the agent inside the cell.
        direction : int
            Direction of movement whose validity is to be tested.

        Returns
        -------
        int
            Validity of the requested transition: 0/1 allowed/not allowed.

        """
        return ((cell_transition >> ((4 - 1 - orientation) * 4)) >> (4 - 1 - direction)) & 1

    def set_transition(self, cell_transition, orientation, direction, new_transition, remove_deadends=False):
        """
        Set the transition bit (1 value) that determines whether an agent
        oriented in direction `orientation` and inside a cell with transitions
        `cell_transition' can move to the cell in direction `direction`
        relative to the current cell.

        Parameters
        ----------
        cell_transition : int
            16 bits used to encode the valid transitions for a cell.
        orientation : int
            Orientation of the agent inside the cell.
        direction : int
            Direction of movement whose validity is to be tested.
        new_transition : int
            Validity of the requested transition: 0/1 allowed/not allowed.
        remove_deadends -- boolean, default False
            remove all deadend transitions.
        Returns
        -------
        int
            An updated bitmap that replaces the original transitions validity
            of `cell_transition' with `new_transitions`, for the appropriate
            `orientation`.

        """
        if new_transition:
            cell_transition |= (1 << ((4 - 1 - orientation) * 4 + (4 - 1 - direction)))
        else:
            cell_transition &= ~(1 << ((4 - 1 - orientation) * 4 + (4 - 1 - direction)))

        if remove_deadends:
            cell_transition = self.remove_deadends(cell_transition)

        return cell_transition

    def rotate_transition(self, cell_transition, rotation=0):
        """
        Clockwise-rotate a 16-bit transition bitmap by
        rotation={0, 90, 180, 270} degrees.

        Parameters
        ----------
        cell_transition : int
            16 bits used to encode the valid transitions for a cell.
        rotation : int
            Angle by which to clock-wise rotate the transition bits in
            `cell_transition` by. I.e., rotation={0, 90, 180, 270} degrees.

        Returns
        -------
        int
            An updated bitmap that replaces the original transitions bits
            with the equivalent bitmap after rotation.

        """
        # Rotate the individual bits in each block
        value = cell_transition
        rotation = rotation // 90
        for i in range(4):
            block_tuple = self.get_transitions(value, i)
            block_tuple = block_tuple[(4 - rotation):] + block_tuple[:(4 - rotation)]
            value = self.set_transitions(value, i, block_tuple)

        # Rotate the 4-bits blocks
        value = ((value & (2 ** (rotation * 4) - 1)) << ((4 - rotation) * 4)) | (value >> (rotation * 4))

        cell_transition = value
        return cell_transition

    def get_direction_enum(self) -> Type[Grid4TransitionsEnum]:
        return Grid4TransitionsEnum

    def has_deadend(self, cell_transition):
        """
        Checks if one entry can only by exited by a turn-around.
        """
        if cell_transition & self.maskDeadEnds > 0:
            return True
        else:
            return False

    def remove_deadends(self, cell_transition):
        """
        Remove all turn-arounds (e.g. N-S, S-N, E-W,...).
        """
        cell_transition &= cell_transition & (~self.maskDeadEnds) & 0xffff
        return cell_transition

    @staticmethod
    def get_entry_directions(cell_transition) -> List[int]:
        return [(cell_transition >> ((3 - orientation) * 4)) & 15 > 0 for orientation in range(4)]
