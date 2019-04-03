"""
The transitions module defines the base Transitions class and a
derived GridTransitions class, which allows for the specification of
possible transitions over a 2D grid.
"""


class Transitions:
    """
    Generic class that implements checks to control whether a
    certain transition is allowed (agent facing a direction
    `orientation' and moving into direction `direction')
    """

    def get_transitions_from_orientation(self, cell_transition, orientation):
        """
        Return a tuple of transitions available in a cell specified by
        `cell_transition' for an agent facing direction `orientation'
        (e.g., a tuple of size of the maximum number of transitions,
        with values 0 or 1, or potentially in between,
        for stochastic transitions).
        """
        raise NotImplementedError()

    def set_transitions_from_orientation(self, cell_transition, orientation,
                                         new_transitions):
        """
        Return a `cell_transition' specification where the transitions
        available for an agent facing direction `orientation' are replaced
        with the tuple `new_transitions'. `new_orientations' must have
        one element for each possible transition.
        """
        raise NotImplementedError()

    def get_transition_from_orientation_to_direction(self, cell_transition,
                                                     orientation, direction):
        """
        Return the status of whether an agent oriented in directions
        `orientation' and inside a cell with transitions `cell_transition'
        can move to the cell in direction `direction' relative
        to the current cell.
        """
        raise NotImplementedError()

    def set_transition_from_orientation_to_direction(self,
                                                     cell_transition,
                                                     orientation,
                                                     direction,
                                                     new_transition):
        """
        Return a `cell_transition' specification where the status of
        whether an agent oriented in direction `orientation' and inside
        a cell with transitions `cell_transition' can move to the cell
        in direction `direction' relative to the current cell is set
        to `new_transition'.
        """
        raise NotImplementedError()


class GridTransitions(Transitions):
    """
    Special case of `Transitions' over a 2D-grid (FlatLand).
    Transitions are possible to neighboring cells on the grid if allowed.
    """

    def __init__(self,
                 transitions,
                 allow_diagonal_transitions=False
                 ):
        self.number_of_cell_neighbors = 4
        if allow_diagonal_transitions:
            self.number_of_cell_neighbors = 8

        self.transitions = transitions

    def get_transitions_from_orientation(self, cell_transition, orientation):
        """
        Get the 4 possible transitions ((N,E,S,W), 4 elements tuple
        if no diagonal transitions allowed) available for an agent oriented
        in direction `orientation' and inside a cell with
        transitions `cell_transition'.
        """
        if self.number_of_cell_neighbors == 4:
            bits = (cell_transition >> ((3-orientation)*4))
            cell_transition = ((bits >> 3) & 1, (bits >> 2)
                               & 1, (bits >> 1) & 1, (bits) & 1)
        elif self.number_of_cell_neighbors == 8:
            bits = (cell_transition >> ((7-orientation)*8))
            cell_transition = (
                (bits >> 7) & 1,
                (bits >> 6) & 1,
                (bits >> 5) & 1,
                (bits >> 4) & 1,
                (bits >> 3) & 1,
                (bits >> 2) & 1,
                (bits >> 1) & 1,
                (bits) & 1)
        else:
            raise NotImplementedError()

        return cell_transition

    def set_transitions_from_orientation(self, cell_transition, orientation,
                                         new_transitions):
        """
        Set the possible transitions (e.g., (N,E,S,W), 4 elements tuple
        if no diagonal transitions allowed) available for an agent
        oriented in direction `orientation' and inside a cell with transitions
        `cell_transition'. A new `cell_transition' is returned with
        the specified bits replaced by `new_transitions'.
        """
        if self.number_of_cell_neighbors == 4:
            mask = (1 << ((4-orientation)*4)) - (1 << ((3-orientation)*4))
            negmask = ~mask

            new_transitions = \
                (new_transitions[0] & 1) << 3 | \
                (new_transitions[1] & 1) << 2 | \
                (new_transitions[2] & 1) << 1 | \
                (new_transitions[3] & 1)

            cell_transition = \
                (
                    cell_transition & negmask) | \
                (new_transitions << ((3-orientation)*4))
        elif self.number_of_cell_neighbors == 8:
            mask = (1 << ((8-orientation)*8)) - (1 << ((7-orientation)*8))
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

            cell_transition = (cell_transition & negmask) | (
                new_transitions << ((7-orientation)*8))
        else:
            raise NotImplementedError()

        return cell_transition

    def get_transition_from_orientation_to_direction(self, cell_transition,
                                                     orientation, direction):
        """
        Get the transition bit (1 value) that determines whether an agent
        oriented in direction `orientation' and inside a cell with transitions
        `cell_transition' can move to the cell in direction `direction'
        relative to the current cell.
        """
        return ((cell_transition >>
                 ((self.number_of_cell_neighbors-1-orientation) *
                  self.number_of_cell_neighbors)) >>
                (self.number_of_cell_neighbors-1-direction)) & 1

    def set_transition_from_orientation_to_direction(self, cell_transition,
                                                     orientation, direction,
                                                     new_transition):
        """
        Set the transition bit (1 value) that determines whether an agent
        oriented in direction `orientation' and inside a cell with transitions
        `cell_transition' can move to the cell in direction `direction'
        relative to the current cell.
        """
        if new_transition:
            cell_transition |= \
                (1 << ((self.number_of_cell_neighbors-1-orientation) *
                       self.number_of_cell_neighbors +
                       (self.number_of_cell_neighbors
                        - 1 - direction)))
        else:
            cell_transition &= \
                ~(1 << ((self.number_of_cell_neighbors-1-orientation) *
                        self.number_of_cell_neighbors +
                        (self.number_of_cell_neighbors
                         - 1 - direction)))

        return cell_transition

    def rotate_transition(self, cell_transition, rotation=0):
        """
        Clockwise-rotate a 16-bit or 64-bit transition bitmap by
        rotation={0, 90, 180, 270} degrees in diagonal steps are not allowed,
        or by rotation={0, 45, 90, 135, 180, 225, 270, 315} degrees if \
        they are.
        """
        if self.number_of_cell_neighbors == 4:
            # Rotate the individual bits in each block
            value = cell_transition
            rotation = rotation // 90
            for i in range(4):
                block_tuple = self.get_transitions_from_orientation(value, i)
                block_tuple = block_tuple[(
                    4-rotation):] + block_tuple[:(4-rotation)]
                value = self.set_transitions_from_orientation(
                    value, i, block_tuple)

            # Rotate the 4bits blocks
            value = ((value & (2**(rotation*4)-1)) <<
                     ((4-rotation)*4)) | (value >> (rotation*4))

            cell_transition = value

        elif self.number_of_cell_neighbors == 8:
            # TODO: WARNING: this part of the function has never been tested!

            # Rotate the individual bits in each block
            value = cell_transition
            rotation = rotation // 45
            for i in range(8):
                block_tuple = self.get_transitions_from_orientation(value, i)
                block_tuple = block_tuple[rotation:] + block_tuple[:rotation]
                value = self.set_transitions_from_orientation(
                    value, i, block_tuple)

            # Rotate the 8bits blocks
            value = ((value & (2**(rotation*8)-1)) <<
                     ((8-rotation)*8)) | (value >> (rotation*8))

            cell_transition = value

        else:
            raise NotImplementedError()

        return cell_transition


"""
Special case of `GridTransitions' over a 2D-grid, with a pre-defined set
of transitions mimicking the types of real Swiss rail connections.

-----------------------------------------------------------------------------------------------

The possible transitions for RailEnv from a cell to its neighboring ones
are represented over 16 bits.

Whether a transition is allowed or not depends on which direction an agent
inside the cell is facing (0=North, 1=East, 2=South, 3=West) and which
direction the agent wants to move to
(North, East, South, West, relative to the cell).
Each transition (orientation, direction) can be allowed (1) or forbidden (0).

The 16 bits are organized in 4 blocks of 4 bits each, the direction that
the agent is facing.
E.g., the most-significant 4-bits represent the possible movements (NESW)
if the agent is facing North, etc...

agent's direction:          North    East   South   West
agent's allowed movements:  [nesw]   [nesw] [nesw]  [nesw]
example:                     0010     0000   1000    0000

In the example, the agent can move from North to South and viceversa.
"""

"""
transitions[] is indexed by case type/id, and returns the 4x4-bit [NESW]
transitions available as a function of the agent's orientation
(north, east, south, west)
"""
RailEnvTransitionsList = [int('0000000000000000', 2),
                          int('1000000000100000', 2),
                          int('1001001000100000', 2),
                          int('1000010000100001', 2),
                          int('1001011000100001', 2),
                          int('1100110000110011', 2),
                          int('0101001000000010', 2),
                          int('0000000000100000', 2)]

RailEnvTransitions = GridTransitions(transitions=RailEnvTransitionsList,
                                     allow_diagonal_transitions=False)
