"""
TransitionMap and derived classes.
"""

import numpy as np

from .transitions import Grid4Transitions, Grid8Transitions, RailEnvTransitions


class TransitionMap:
    """
    Base TransitionMap class.

    Generic class that implements a collection of transitions over a set of
    cells.
    """

    def get_transitions(self, cell_id):
        """
        Return a tuple of transitions available in a cell specified by
        `cell_id' (e.g., a tuple of size of the maximum number of transitions,
        with values 0 or 1, or potentially in between,
        for stochastic transitions).

        Parameters
        ----------
        cell_id : [cell identifier]
            The cell_id object depends on the specific implementation.
            It generally is an int (e.g., an index) or a tuple of indices.

        Returns
        -------
        tuple
            List of the validity of transitions in the cell.

        """
        raise NotImplementedError()

    def set_transitions(self, cell_id, new_transitions):
        """
        Replaces the available transitions in cell `cell_id' with the tuple
        `new_transitions'. `new_transitions' must have
        one element for each possible transition.

        Parameters
        ----------
        cell_id : [cell identifier]
            The cell_id object depends on the specific implementation.
            It generally is an int (e.g., an index) or a tuple of indices.
        new_transitions : tuple
            Tuple of new transitions validitiy for the cell.

        """
        raise NotImplementedError()

    def get_transition(self, cell_id, transition_index):
        """
        Return the status of whether an agent in cell `cell_id' can perform a
        movement along transition `transition_index (e.g., the NESW direction
        of movement, for agents on a grid).

        Parameters
        ----------
        cell_id : [cell identifier]
            The cell_id object depends on the specific implementation.
            It generally is an int (e.g., an index) or a tuple of indices.
        transition_index : int
            Index of the transition to probe, as index in the tuple returned by
            get_transitions(). e.g., the NESW direction of movement, for agents
            on a grid.

        Returns
        -------
        int or float (depending on derived class)
            Validity of the requested transition (e.g.,
            0/1 allowed/not allowed, a probability in [0,1], etc...)

        """
        raise NotImplementedError()

    def set_transition(self, cell_id, transition_index, new_transition):
        """
        Replaces the validity of transition to `transition_index' in cell
        `cell_id' with the new `new_transition'.


        Parameters
        ----------
        cell_id : [cell identifier]
            The cell_id object depends on the specific implementation.
            It generally is an int (e.g., an index) or a tuple of indices.
        transition_index : int
            Index of the transition to probe, as index in the tuple returned by
            get_transitions(). e.g., the NESW direction of movement, for agents
            on a grid.
        new_transition : int or float (depending on derived class)
            Validity of the requested transition (e.g.,
            0/1 allowed/not allowed, a probability in [0,1], etc...)

        """
        raise NotImplementedError()


class GridTransitionMap(TransitionMap):
    """
    Implements a TransitionMap over a 2D grid.

    GridTransitionMap implements utility functions.
    """

    def __init__(self, width, height, transitions=Grid4Transitions([])):
        """
        Builder for GridTransitionMap object.

        Parameters
        ----------
        width : int
            Width of the grid.
        height : int
            Height of the grid.
        transitions : Transitions object
            The Transitions object to use to encode/decode transitions over the
            grid.

        """

        self.width = width
        self.height = height
        self.transitions = transitions

        if isinstance(self.transitions, Grid4Transitions) or isinstance(self.transitions, RailEnvTransitions):
            self.grid = np.ndarray((height, width), dtype=np.uint16)
        elif isinstance(self.transitions, Grid8Transitions):
            self.grid = np.ndarray((height, width), dtype=np.uint64)

    def get_transitions(self, cell_id):
        """
        Return a tuple of transitions available in a cell specified by
        `cell_id' (e.g., a tuple of size of the maximum number of transitions,
        with values 0 or 1, or potentially in between,
        for stochastic transitions).

        Parameters
        ----------
        cell_id : tuple
            The cell_id indices a cell as (column, row, orientation),
            where orientation is the direction an agent is facing within a cell.
            Alternatively, it can be accessed as (column, row) to return the
            full cell content.

        Returns
        -------
        tuple
            List of the validity of transitions in the cell.

        """
        if len(cell_id) == 3:
            return self.transitions.get_transitions(self.grid[cell_id[0]][cell_id[1]], cell_id[2])
        elif len(cell_id) == 2:
            return self.grid[cell_id[0]][cell_id[1]]
        else:
            print('GridTransitionMap.get_transitions() ERROR: \
                   wrong cell_id tuple.')
            return ()

    def set_transitions(self, cell_id, new_transitions):
        """
        Replaces the available transitions in cell `cell_id' with the tuple
        `new_transitions'. `new_transitions' must have
        one element for each possible transition.

        Parameters
        ----------
        cell_id : tuple
            The cell_id indices a cell as (column, row, orientation),
            where orientation is the direction an agent is facing within a cell.
            Alternatively, it can be accessed as (column, row) to replace the
            full cell content.
        new_transitions : tuple
            Tuple of new transitions validitiy for the cell.

        """
        if len(cell_id) == 3:
            self.transitions.set_transitions(self.grid[cell_id[0]][cell_id[1]], cell_id[2], new_transitions)
        elif len(cell_id) == 2:
            self.grid[cell_id[0]][cell_id[1]] = new_transitions
        else:
            print('GridTransitionMap.get_transitions() ERROR: \
                   wrong cell_id tuple.')

    def get_transition(self, cell_id, transition_index):
        """
        Return the status of whether an agent in cell `cell_id' can perform a
        movement along transition `transition_index (e.g., the NESW direction
        of movement, for agents on a grid).

        Parameters
        ----------
        cell_id : tuple
            The cell_id indices a cell as (column, row, orientation),
            where orientation is the direction an agent is facing within a cell.
        transition_index : int
            Index of the transition to probe, as index in the tuple returned by
            get_transitions(). e.g., the NESW direction of movement, for agents
            on a grid.

        Returns
        -------
        int or float (depending on derived class)
            Validity of the requested transition (e.g.,
            0/1 allowed/not allowed, a probability in [0,1], etc...)

        """
        if len(cell_id) != 3:
            print('GridTransitionMap.get_transition() ERROR: \
                   wrong cell_id tuple.')
            return ()
        return self.transitions.get_transition(self.grid[cell_id[0]][cell_id[1]], cell_id[2], transition_index)

    def set_transition(self, cell_id, transition_index, new_transition):
        """
        Replaces the validity of transition to `transition_index' in cell
        `cell_id' with the new `new_transition'.


        Parameters
        ----------
        cell_id : tuple
            The cell_id indices a cell as (column, row, orientation),
            where orientation is the direction an agent is facing within a cell.
        transition_index : int
            Index of the transition to probe, as index in the tuple returned by
            get_transitions(). e.g., the NESW direction of movement, for agents
            on a grid.
        new_transition : int or float (depending on derived class)
            Validity of the requested transition (e.g.,
            0/1 allowed/not allowed, a probability in [0,1], etc...)

        """
        if len(cell_id) != 3:
            print('GridTransitionMap.set_transition() ERROR: \
                   wrong cell_id tuple.')
            return
        self.transitions.set_transition(self.grid[cell_id[0]][cell_id[1]], cell_id[2], transition_index, new_transition)

    def save_transition_map(self, filename):
        """
        Save the transitions grid as `filename', in npy format.

        Parameters
        ----------
        filename : string
            Name of the file to which to save the transitions grid.

        """
        np.save(filename, self.grid)

    def load_transition_map(self, filename, override_gridsize=True):
        """
        Load the transitions grid from `filename' (npy format).
        The load function only updates the transitions grid, and possibly width and height, but the object has to be
        initialized with the correct `transitions' object anyway.

        Parameters
        ----------
        filename : string
            Name of the file from which to load the transitions grid.
        override_gridsize : bool
            If override_gridsize=True, the width and height of the GridTransitionMap object are replaced with the size
            of the map loaded from `filename'. If override_gridsize=False, the transitions grid is either cropped (if
            the grid size is larger than (height,width) ) or padded with zeros (if the grid size is smaller than (height,width) )

        """
        new_grid = np.load(filename)

        new_height = new_grid.shape[0]
        new_width = new_grid.shape[1]

        if override_gridsize:
            self.width = new_width
            self.height = new_height
            self.grid = new_grid

        else:
            if new_grid.dtype == np.uint16:
                self.grid = np.zeros((self.height, self.width), dtype=np.uint16)
            elif new_grid.dtype == np.uint64:
                self.grid = np.zeros((self.height, self.width), dtype=np.uint64)

            self.grid[0:min(self.height, new_height), 0:min(self.width, new_width)] = new_grid[0:min(self.height, new_height), 0:min(self.width, new_width)]

# TODO: GIACOMO: is it better to provide those methods with lists of cell_ids
# (most general implementation) or to make Grid-class specific methods for
# slicing over the 3 dimensions?  I'd say both perhaps.

# TODO: override __getitem__ and __setitem__ (cell contents, not transitions?)
