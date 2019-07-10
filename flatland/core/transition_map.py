"""
TransitionMap and derived classes.
"""

import numpy as np
from importlib_resources import path
from numpy import array

from flatland.core.grid.grid4 import Grid4Transitions
from flatland.core.transitions import Transitions


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
        int or float (depending on Transitions used)
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
        new_transition : int or float (depending on Transitions used)
            Validity of the requested transition (e.g.,
            0/1 allowed/not allowed, a probability in [0,1], etc...)

        """
        raise NotImplementedError()


class GridTransitionMap(TransitionMap):
    """
    Implements a TransitionMap over a 2D grid.

    GridTransitionMap implements utility functions.
    """

    def __init__(self, width, height, transitions: Transitions = Grid4Transitions([])):
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

        self.grid = np.zeros((height, width), dtype=self.transitions.get_type())

    def get_full_transitions(self, row, column):
        """
        Returns the full transitions for the cell at (row, column) in the format transition_map's transitions.

        Parameters
        ----------
        row: int
        column: int
            (row,column) specifies the cell in this transition map.

        Returns
        -------
        self.transitions.get_type()
            The cell content int the format of this map's Transitions.

        """
        return self.grid[row][column]

    def get_transitions(self, row, column, orientation):
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
            List of the validity of transitions in the cell as given by the maps transitions.

        """
        return self.transitions.get_transitions(self.grid[row][column], orientation)

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
        assert len(cell_id) in (2, 3), \
            'GridTransitionMap.set_transitions() ERROR: cell_id tuple must have length 2 or 3.'
        if len(cell_id) == 3:
            self.grid[cell_id[0]][cell_id[1]] = self.transitions.set_transitions(self.grid[cell_id[0]][cell_id[1]],
                                                                                 cell_id[2],
                                                                                 new_transitions)
        elif len(cell_id) == 2:
            self.grid[cell_id[0]][cell_id[1]] = new_transitions

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
        int or float (depending on Transitions used in the )
            Validity of the requested transition (e.g.,
            0/1 allowed/not allowed, a probability in [0,1], etc...)

        """

        assert len(cell_id) == 3, \
            'GridTransitionMap.get_transition() ERROR: cell_id tuple must have length 2 or 3.'
        return self.transitions.get_transition(self.grid[cell_id[0]][cell_id[1]], cell_id[2], transition_index)

    def set_transition(self, cell_id, transition_index, new_transition, remove_deadends=False):
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
        new_transition : int or float (depending on Transitions used in the map.)
            Validity of the requested transition (e.g.,
            0/1 allowed/not allowed, a probability in [0,1], etc...)

        """
        assert len(cell_id) == 3, \
            'GridTransitionMap.set_transition() ERROR: cell_id tuple must have length 3.'
        self.grid[cell_id[0]][cell_id[1]] = self.transitions.set_transition(
            self.grid[cell_id[0]][cell_id[1]],
            cell_id[2],
            transition_index,
            new_transition,
            remove_deadends)

    def save_transition_map(self, filename):
        """
        Save the transitions grid as `filename', in npy format.

        Parameters
        ----------
        filename : string
            Name of the file to which to save the transitions grid.

        """
        np.save(filename, self.grid)

    def load_transition_map(self, package, resource):
        """
        Load the transitions grid from `filename' (npy format).
        The load function only updates the transitions grid, and possibly width and height, but the object has to be
        initialized with the correct `transitions' object anyway.

        Parameters
        ----------
        package : string
            Name of the package from which to load the transitions grid.
        resource : string
            Name of the file from which to load the transitions grid within the package.
        override_gridsize : bool
            If override_gridsize=True, the width and height of the GridTransitionMap object are replaced with the size
            of the map loaded from `filename'. If override_gridsize=False, the transitions grid is either cropped (if
            the grid size is larger than (height,width) ) or padded with zeros (if the grid size is smaller than
            (height,width) )

        """
        with path(package, resource) as file_in:
            new_grid = np.load(file_in)

        new_height = new_grid.shape[0]
        new_width = new_grid.shape[1]

        self.width = new_width
        self.height = new_height
        self.grid = new_grid

    def cell_neighbours_valid(self, rcPos, check_this_cell=False):
        """
        Check validity of cell at rcPos = tuple(row, column)
        Checks that:
        - surrounding cells have inbound transitions for all the
            outbound transitions of this cell.

        These are NOT checked - see transition.is_valid:
        - all transitions have the mirror transitions (N->E <=> W->S)
        - Reverse transitions (N -> S) only exist for a dead-end
        - a cell contains either no dead-ends or exactly one

        Returns: True (valid) or False (invalid)
        """
        cell_transition = self.grid[tuple(rcPos)]

        if check_this_cell:
            if not self.transitions.is_valid(cell_transition):
                return False

        gDir2dRC = self.transitions.gDir2dRC  # [[-1,0] = N, [0,1]=E, etc]
        grcPos = array(rcPos)
        grcMax = self.grid.shape

        binTrans = self.get_full_transitions(*rcPos)  # 16bit integer - all trans in/out
        lnBinTrans = array([binTrans >> 8, binTrans & 0xff], dtype=np.uint8)  # 2 x uint8
        g2binTrans = np.unpackbits(lnBinTrans).reshape(4, 4)  # 4x4 x uint8 binary(0,1)
        gDirOut = g2binTrans.any(axis=0)  # outbound directions as boolean array (4)
        giDirOut = np.argwhere(gDirOut)[:, 0]  # valid outbound directions as array of int

        # loop over available outbound directions (indices) for rcPos
        for iDirOut in giDirOut:
            gdRC = gDir2dRC[iDirOut]  # row,col increment
            gPos2 = grcPos + gdRC  # next cell in that direction

            # Check the adjacent cell is within bounds
            # if not, then this transition is invalid!
            if np.any(gPos2 < 0):
                return False
            if np.any(gPos2 >= grcMax):
                return False

            # Get the transitions out of gPos2, using iDirOut as the inbound direction
            # if there are no available transitions, ie (0,0,0,0), then rcPos is invalid
            t4Trans2 = self.get_transitions(*gPos2, iDirOut)
            if any(t4Trans2):
                continue
            else:
                return False

        return True

# TODO: improvement override __getitem__ and __setitem__ (cell contents, not transitions?)
