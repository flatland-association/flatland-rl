from flatland.core.grid.grid4 import Grid4Transitions


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

    # Contains the basic transitions;
    # the set of all valid transitions is obtained by successive 90-degree rotation of one of these basic transitions.
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
