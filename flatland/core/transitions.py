"""
The transitions module defines the base Transitions class and a
derived GridTransitions class, which allows for the specification of
possible transitions over a 2D grid.
"""
from enum import IntEnum

import numpy as np


class Transitions:
    """
    Base Transitions class.

    Generic class that implements checks to control whether a
    certain transition is allowed (agent facing a direction
    `orientation' and moving into direction `direction')
    """

    def get_transitions(self, cell_transition, orientation):
        """
        Return a tuple of transitions available in a cell specified by
        `cell_transition' for an agent facing direction `orientation'
        (e.g., a tuple of size of the maximum number of transitions,
        with values 0 or 1, or potentially in between,
        for stochastic transitions).

        Parameters
        ----------
        cell_transition : [cell content]
            The object is specific to each derived class (e.g., for
            GridTransitions, int), and is only manipulated by methods
            of the Transitions derived classes.
        orientation : int
            Orientation of the agent inside the cell.

        Returns
        -------
        tuple
            List of the validity of transitions in the cell.

        """
        raise NotImplementedError()

    def set_transitions(self, cell_transition, orientation, new_transitions):
        """
        Return a `cell_transition' specification where the transitions
        available for an agent facing direction `orientation' are replaced
        with the tuple `new_transitions'. `new_orientations' must have
        one element for each possible transition.

        Parameters
        ----------
        cell_transition : [cell-content]
            The object is specific to each derived class (e.g., for
            GridTransitions, int), and is only manipulated by methods
            of the Transitions derived classes.
        orientation : int
            Orientation of the agent inside the cell.
        new_transitions : tuple
            Tuple of new transitions validitiy for the cell.

        Returns
        -------
        [cell-content]
            An updated class-specific object that replaces the original
            transitions validity of `cell_transition' with `new_transitions',
            for the appropriate `orientation'.

        """
        raise NotImplementedError()

    def get_transition(self, cell_transition, orientation, direction):
        """
        Return the status of whether an agent oriented in directions
        `orientation' and inside a cell with transitions `cell_transition'
        can move to the cell in direction `direction' relative
        to the current cell.

        Parameters
        ----------
        cell_transition : [cell-content]
            The object is specific to each derived class (e.g., for
            GridTransitions, int), and is only manipulated by methods
            of the Transitions derived classes.
        orientation : int
            Orientation of the agent inside the cell.
        direction : int
            Direction of movement whose validity is to be tested.

        Returns
        -------
        int or float (depending on derived class)
            Validity of the requested transition (e.g.,
            0/1 allowed/not allowed, a probability in [0,1], etc...)

        """
        raise NotImplementedError()

    def set_transition(self, cell_transition, orientation, direction,
                       new_transition):
        """
        Return a `cell_transition' specification where the status of
        whether an agent oriented in direction `orientation' and inside
        a cell with transitions `cell_transition' can move to the cell
        in direction `direction' relative to the current cell is set
        to `new_transition'.

        Parameters
        ----------
        cell_transition : [cell-content]
            The object is specific to each derived class (e.g., for
            GridTransitions, int), and is only manipulated by methods
            of the Transitions derived classes.
        orientation : int
            Orientation of the agent inside the cell.
        direction : int
            Direction of movement whose validity is to be tested.
        new_transition : int or float (depending on derived class)
            Validity of the requested transition (e.g.,
            0/1 allowed/not allowed, a probability in [0,1], etc...)

        Returns
        -------
        [cell-content]
            An updated class-specific object that replaces the original
            transitions validity of `cell_transition' with `new_transitions',
            for the appropriate `orientation' to `direction'.

        """
        raise NotImplementedError()

    def get_direction_enum(self) -> IntEnum:
        raise NotImplementedError()


class Grid4TransitionsEnum(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Grid4Transitions(Transitions):
    """
    Grid4Transitions class derived from Transitions.

    Special case of `Transitions' over a 2D-grid (FlatLand).
    Transitions are possible to neighboring cells on the grid if allowed.
    GridTransitions keeps track of valid transitions supplied as `transitions'
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

    def get_transitions(self, cell_transition, orientation):
        """
        Get the 4 possible transitions ((N,E,S,W), 4 elements tuple
        if no diagonal transitions allowed) available for an agent oriented
        in direction `orientation' and inside a cell with
        transitions `cell_transition'.

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
        oriented in direction `orientation' and inside a cell with transitions
        `cell_transition'. A new `cell_transition' is returned with
        the specified bits replaced by `new_transitions'.

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
            of `cell_transition' with `new_transitions', for the appropriate
            `orientation'.

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
        oriented in direction `orientation' and inside a cell with transitions
        `cell_transition' can move to the cell in direction `direction'
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
        oriented in direction `orientation' and inside a cell with transitions
        `cell_transition' can move to the cell in direction `direction'
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
            of `cell_transition' with `new_transitions', for the appropriate
            `orientation'.

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
            `cell_transition' by. I.e., rotation={0, 90, 180, 270} degrees.

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

    def get_direction_enum(self) -> IntEnum:
        return Grid4TransitionsEnum


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
        bits = (cell_transition >> ((7 - orientation) * 8))
        cell_transition = (
            (bits >> 7) & 1,
            (bits >> 6) & 1,
            (bits >> 5) & 1,
            (bits >> 4) & 1,
            (bits >> 3) & 1,
            (bits >> 2) & 1,
            (bits >> 1) & 1,
            (bits) & 1)

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
            (new_transitions[0] & 1) << 7 | \
            (new_transitions[1] & 1) << 6 | \
            (new_transitions[2] & 1) << 5 | \
            (new_transitions[3] & 1) << 4 | \
            (new_transitions[4] & 1) << 3 | \
            (new_transitions[5] & 1) << 2 | \
            (new_transitions[6] & 1) << 1 | \
            (new_transitions[7] & 1)

        cell_transition = (cell_transition & negmask) | (new_transitions << ((7 - orientation) * 8))

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

    def set_transition(self, cell_transition, orientation, direction,
                       new_transition):
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


class RailEnvTransitions(Grid4Transitions):
    """
    Special case of `GridTransitions' over a 2D-grid, with a pre-defined set
    of transitions mimicking the types of real Swiss rail connections.

    --------------------------------------------------------------------------

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

    """
    transitions[] is indexed by case type/id, and returns the 4x4-bit [NESW]
    transitions available as a function of the agent's orientation
    (north, east, south, west)
    """

    transition_list = [int('0000000000000000', 2),  # empty cell - Case 0
                       int('1000000000100000', 2),  # Case 1 - straight
                       int('1001001000100000', 2),  # Case 2 - simple switch
                       int('1000010000100001', 2),  # Case 3 - diamond drossing
                       int('1001011000100001', 2),  # Case 4 - single slip
                       int('1100110000110011', 2),  # Case 5 - double slip
                       int('0101001000000010', 2),  # Case 6 - symmetrical
                       int('0010000000000000', 2),  # Case 7 - dead end
                       int('0100000000000010', 2),  # Case 1b (8)  - simple turn right
                       int('0001001000000000', 2),  # Case 1c (9)  - simple turn left
                       int('1100000000100010', 2)]  # Case 2b (10) - simple switch mirrored

    def __init__(self):
        super(RailEnvTransitions, self).__init__(
            transitions=self.transition_list
        )

        # These bits represent all the possible dead ends
        self.maskDeadEnds = 0b0010000110000100

        # create this to make validation faster
        self.transitions_all = set()
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

    def repr(self, cell_transition, version=0):
        """
        Provide a string representation of the cell transitions.
        This class doesn't represent an individual cell,
        but a way of interpreting the contents of a cell.
        So using the ad hoc name repr rather than __repr__.
        """
        # binary format string without leading 0b
        sbinTrans = format(cell_transition, "#018b")[2:]
        if version == 0:
            sRepr = " ".join([
                "{}:{}".format(sDir, sbinTrans[i:(i + 4)])
                for i, sDir in
                zip(
                    range(0, len(sbinTrans), 4),
                    self.lsDirs)])  # NESW
            return sRepr

        if version == 1:
            lsRepr = []
            for iDirIn in range(0, 4):
                sDirTrans = sbinTrans[(iDirIn * 4):(iDirIn * 4 + 4)]
                if sDirTrans == "0000":
                    continue
                sDirsOut = [
                    self.lsDirs[iDirOut]
                    for iDirOut in range(0, 4)
                    if sDirTrans[iDirOut] == "1"]
                lsRepr.append(self.lsDirs[iDirIn] + ":" + "".join(sDirsOut))

            return ", ".join(lsRepr)

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

    def has_deadend(self, cell_transition):
        if cell_transition & self.maskDeadEnds > 0:
            return True
        else:
            return False

    def remove_deadends(self, cell_transition):
        cell_transition &= cell_transition & (~self.maskDeadEnds) & 0xffff
        return cell_transition
