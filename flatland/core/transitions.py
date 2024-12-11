"""
The transitions module defines the base Transitions class and a
derived GridTransitions class, which allows for the specification of
possible transitions over a 2D grid.
"""
from enum import IntEnum


class Transitions:
    """
    Base Transitions class.

    Generic class that implements checks to control whether a
    certain transition is allowed (agent facing a direction
    `orientation' and moving into direction `orientation`)
    """

    def get_type(self):
        raise NotImplementedError()

    def get_transitions(self, cell_transition, orientation):
        """
        Return a tuple of transitions available in a cell specified by
        `cell_transition' for an agent facing direction `orientation`
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
        Return a `cell_transition` specification where the transitions
        available for an agent facing direction `orientation` are replaced
        with the tuple `new_transitions'. `new_orientations` must have
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
            transitions validity of `cell_transition' with `new_transitions`,
            for the appropriate `orientation`.

        """
        raise NotImplementedError()

    def get_transition(self, cell_transition, orientation, direction):
        """
        Return the status of whether an agent oriented in directions
        `orientation' and inside a cell with transitions `cell_transition`
        can move to the cell in direction `direction` relative
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
        Return a `cell_transition` specification where the status of
        whether an agent oriented in direction `orientation` and inside
        a cell with transitions `cell_transition` can move to the cell
        in direction `direction` relative to the current cell is set
        to `new_transition`.

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
            transitions validity of `cell_transition' with `new_transitions`,
            for the appropriate `orientation' to `direction`.

        """
        raise NotImplementedError()

    def get_direction_enum(self) -> IntEnum:
        raise NotImplementedError()
