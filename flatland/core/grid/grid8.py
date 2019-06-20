from enum import IntEnum

import numpy as np

from flatland.core.transitions import Transitions


class Grid8TransitionsEnum(IntEnum):
    NORTH = 0
    NORTH_EAST = 1
    EAST = 2
    SOUTH_EAST = 3
    SOUTH = 4
    SOUTH_WEST = 5
    WEST = 6
    NORTH_WEST = 7


class Grid8Transitions(Transitions):
    """
    Grid8Transitions class derived from Transitions.

    Special case of `Transitions' over a 2D-grid (FlatLand).
    Transitions are possible to neighboring cells on the grid if allowed.
    GridTransitions keeps track of valid transitions supplied as `transitions'
    list, each represented as a bitmap of 64 bits.

    0=North, 1=North-East, etc.

    """

    def __init__(self, transitions):
        self.transitions = transitions

    def get_type(self):
        return np.uint64

    def get_transitions(self, cell_transition, orientation):
        """
        Get the 8 possible transitions.

        Parameters
        ----------
        cell_transition : int
            64 bits used to encode the valid transitions for a cell.
        orientation : int
            Orientation of the agent inside the cell.

        Returns
        -------
        tuple
            List of the validity of transitions in the cell.

        """
        bits = (np.uint64(cell_transition) >> np.uint64((7 - orientation) * 8))
        cell_transition = (
            (bits >> np.uint64(7)) & np.uint64(1),
            (bits >> np.uint64(6)) & np.uint64(1),
            (bits >> np.uint64(5)) & np.uint64(1),
            (bits >> np.uint64(4)) & np.uint64(1),
            (bits >> np.uint64(3)) & np.uint64(1),
            (bits >> np.uint64(2)) & np.uint64(1),
            (bits >> np.uint64(1)) & np.uint64(1),
            bits & np.uint64(1))

        return cell_transition

    def set_transitions(self, cell_transition, orientation, new_transitions):
        """
        Set the possible transitions.

        Parameters
        ----------
        cell_transition : int
            64 bits used to encode the valid transitions for a cell.
        orientation : int
            Orientation of the agent inside the cell.
        new_transitions : tuple
            Tuple of new transitions validitiy for the cell.

        Returns
        -------
        int
            An updated bitmap that replaces the original transitions validity
            of `cell_transition' with `new_transitions', for the appropriate
            `orientation'.

        """
        mask = (1 << ((8 - orientation) * 8)) - (1 << ((7 - orientation) * 8))
        negmask = ~mask

        new_transitions = \
            (int(new_transitions[0]) & 1) << 7 | \
            (int(new_transitions[1]) & 1) << 6 | \
            (int(new_transitions[2]) & 1) << 5 | \
            (int(new_transitions[3]) & 1) << 4 | \
            (int(new_transitions[4]) & 1) << 3 | \
            (int(new_transitions[5]) & 1) << 2 | \
            (int(new_transitions[6]) & 1) << 1 | \
            (int(new_transitions[7]) & 1)

        cell_transition = (int(cell_transition) & negmask) | (new_transitions << ((7 - orientation) * 8))

        return cell_transition

    def get_transition(self, cell_transition, orientation, direction):
        """
        Get the transition bit (1 value) that determines whether an agent
        oriented in direction `orientation' and inside a cell with transitions
        `cell_transition' can move to the cell in direction `direction'
        relative to the current cell.

        Parameters
        ----------
        cell_transition : int
            64 bits used to encode the valid transitions for a cell.
        orientation : int
            Orientation of the agent inside the cell.
        direction : int
            Direction of movement whose validity is to be tested.

        Returns
        -------
        int
            Validity of the requested transition: 0/1 allowed/not allowed.

        """
        return ((cell_transition >> ((8 - 1 - orientation) * 8)) >> (8 - 1 - direction)) & 1

    def set_transition(self, cell_transition, orientation, direction, new_transition, remove_deadends=False):

        """
        Set the transition bit (1 value) that determines whether an agent
        oriented in direction `orientation' and inside a cell with transitions
        `cell_transition' can move to the cell in direction `direction'
        relative to the current cell.

        Parameters
        ----------
        cell_transition : int
            64 bits used to encode the valid transitions for a cell.
        orientation : int
            Orientation of the agent inside the cell.
        direction : int
            Direction of movement whose validity is to be tested.
        new_transition : int
            Validity of the requested transition: 0/1 allowed/not allowed.

        Returns
        -------
        int
            An updated bitmap that replaces the original transitions validity
            of `cell_transition' with `new_transitions', for the appropriate
            `orientation'.

        """
        if new_transition:
            cell_transition |= (1 << ((8 - 1 - orientation) * 8 + (8 - 1 - direction)))
        else:
            cell_transition &= ~(1 << ((8 - 1 - orientation) * 8 + (8 - 1 - direction)))

        return cell_transition

    def rotate_transition(self, cell_transition, rotation=0):
        """
        Clockwise-rotate a 64-bit transition bitmap by
        rotation={0, 45, 90, 135, 180, 225, 270, 315} degrees.

        Parameters
        ----------
        cell_transition : int
            64 bits used to encode the valid transitions for a cell.
        rotation : int
            Angle by which to clock-wise rotate the transition bits in
            `cell_transition' by. I.e., rotation={0, 45, 90, 135, 180,
            225, 270, 315} degrees.

        Returns
        -------
        int
            An updated bitmap that replaces the original transitions bits
            with the equivalent bitmap after rotation.

        """
        # TODO: WARNING: this part of the function has never been tested!

        # Rotate the individual bits in each block
        value = cell_transition
        rotation = rotation // 45
        for i in range(8):
            block_tuple = self.get_transitions(value, i)
            block_tuple = block_tuple[rotation:] + block_tuple[:rotation]
            value = self.set_transitions(value, i, block_tuple)

        # Rotate the 8bits blocks
        value = ((value & (2 ** (rotation * 8) - 1)) << ((8 - rotation) * 8)) | (value >> (rotation * 8))

        cell_transition = value

        return cell_transition

    def get_direction_enum(self) -> IntEnum:
        return Grid8TransitionsEnum
