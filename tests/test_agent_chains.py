from typing import Tuple, List

import networkx as nx
from matplotlib import pyplot as plt

from flatland.envs.agent_chains import MotionCheckLegacy, MotionCheck


def test_add_agent_waiting():
    mc = MotionCheckLegacy()

    # two agents desiring to enter the same cell from off the map at the same time step
    mc.add_agent(0, None, (0, 0))
    mc.add_agent(1, None, (0, 0))

    mc.find_conflicts()

    assert mc.check_motion(0, None)
    assert not mc.check_motion(1, None)


def test_add_agent_waiting_blocked():
    mc = MotionCheckLegacy()

    # first agent already on the cell
    mc.add_agent(0, (0, 0), (0, 0))

    # second agent from off the map desiring to enter the same cell
    mc.add_agent(1, None, (0, 0))

    mc.find_conflicts()

    assert not mc.check_motion(0, (0, 0))  # current behaviour is motion check false if staying
    assert not mc.check_motion(1, None)


# TODO unused - dump?
def create_test_agents(omc: MotionCheckLegacy):
    # blocked chain
    omc.add_agent(1, (1, 2), (1, 3))
    omc.add_agent(2, (1, 3), (1, 4))
    omc.add_agent(3, (1, 4), (1, 5))
    omc.add_agent(31, (1, 5), (1, 5))

    # unblocked chain
    omc.add_agent(4, (2, 1), (2, 2))
    omc.add_agent(5, (2, 2), (2, 3))

    # blocked short chain
    omc.add_agent(6, (3, 1), (3, 2))
    omc.add_agent(7, (3, 2), (3, 2))

    # solitary agent
    omc.add_agent(8, (4, 1), (4, 2))

    # solitary stopped agent
    omc.add_agent(9, (5, 1), (5, 1))

    # blocked short chain (opposite direction)
    omc.add_agent(10, (6, 4), (6, 3))
    omc.add_agent(11, (6, 3), (6, 3))

    # swap conflict
    omc.add_agent(12, (7, 1), (7, 2))
    omc.add_agent(13, (7, 2), (7, 1))


class ChainTestEnv(object):
    """ Just for testing agent chains
    """

    def __init__(self, omc: MotionCheckLegacy):
        self.iAgNext = 0
        self.iRowNext = 1
        self.omc = omc

    def addAgent(self, rc1, rc2, xlabel=None):
        self.omc.add_agent(self.iAgNext, rc1, rc2, xlabel=xlabel)
        self.iAgNext += 1

    def addAgentToRow(self, c1, c2, xlabel=None):
        self.addAgent((self.iRowNext, c1), (self.iRowNext, c2), xlabel=xlabel)

    def create_test_chain(self,
                          nAgents: int,
                          rcVel: Tuple[int] = (0, 1),
                          liStopped: List[int] = [],
                          xlabel=None):
        """ create a chain of agents
        """
        lrcAgPos = [(self.iRowNext, i * rcVel[1]) for i in range(nAgents)]

        for iAg, rcPos in zip(range(nAgents), lrcAgPos):
            if iAg in liStopped:
                rcVel1 = (0, 0)
            else:
                rcVel1 = rcVel
            self.omc.add_agent(iAg + self.iAgNext, rcPos, (rcPos[0] + rcVel1[0], rcPos[1] + rcVel1[1]))

        if xlabel:
            self.omc.G.nodes[lrcAgPos[0]]["xlabel"] = xlabel

        self.iAgNext += nAgents
        self.iRowNext += 1

    def nextRow(self):
        self.iRowNext += 1


def create_test_agents2(omc: MotionCheckLegacy):
    # blocked chain
    cte = ChainTestEnv(omc)
    cte.create_test_chain(4, liStopped=[3], xlabel="stopped\nchain")
    cte.create_test_chain(4, xlabel="running\nchain")

    cte.create_test_chain(2, liStopped=[1], xlabel="stopped \nshort\n chain")

    cte.addAgentToRow(1, 2, "swap")
    cte.addAgentToRow(2, 1)

    cte.nextRow()

    cte.addAgentToRow(1, 2, "chain\nswap")
    cte.addAgentToRow(2, 3)
    cte.addAgentToRow(3, 2)

    cte.nextRow()

    cte.addAgentToRow(1, 2, "midchain\nstop")
    cte.addAgentToRow(2, 3)
    cte.addAgentToRow(3, 4)
    cte.addAgentToRow(4, 4)
    cte.addAgentToRow(5, 6)
    cte.addAgentToRow(6, 7)

    cte.nextRow()

    cte.addAgentToRow(1, 2, "midchain\nswap")
    cte.addAgentToRow(2, 3)
    cte.addAgentToRow(3, 4)
    cte.addAgentToRow(4, 3)
    cte.addAgentToRow(5, 4)
    cte.addAgentToRow(6, 5)

    cte.nextRow()

    cte.addAgentToRow(1, 2, "Land on\nSame")
    cte.addAgentToRow(3, 2)

    cte.nextRow()
    cte.addAgentToRow(1, 2, "chains\nonto\nsame")
    cte.addAgentToRow(2, 3)
    cte.addAgentToRow(3, 4)
    cte.addAgentToRow(5, 4)
    cte.addAgentToRow(6, 5)
    cte.addAgentToRow(7, 6)

    cte.nextRow()
    cte.addAgentToRow(1, 2, "3-way\nsame")
    cte.addAgentToRow(3, 2)
    cte.addAgent((cte.iRowNext + 1, 2), (cte.iRowNext, 2))
    cte.nextRow()

    cte.nextRow()
    cte.addAgentToRow(1, 2, "Tee")
    cte.addAgentToRow(2, 3)
    cte.addAgentToRow(3, 4)
    cte.addAgent((cte.iRowNext + 1, 3), (cte.iRowNext, 3))
    cte.nextRow()

    cte.nextRow()
    cte.addAgentToRow(1, 2, "Tree")
    cte.addAgentToRow(2, 3)
    cte.addAgentToRow(3, 4)
    r1 = cte.iRowNext
    r2 = cte.iRowNext + 1
    r3 = cte.iRowNext + 2
    cte.addAgent((r2, 3), (r1, 3))
    cte.addAgent((r2, 2), (r2, 3))
    cte.addAgent((r3, 2), (r2, 3))

    cte.nextRow()
    cte.nextRow()
    cte.nextRow()

    r1 = cte.iRowNext
    r2 = cte.iRowNext + 1
    r3 = cte.iRowNext + 2

    cte.addAgent((r2, 3), (r1, 3))
    cte.addAgent((r2, 2), (r2, 3))
    cte.addAgent((r3, 2), (r2, 3))
    cte.addAgent((r1, 1), (r1, 2), "Tree different order")
    cte.addAgent((r1, 2), (r1, 3))
    cte.addAgent((r1, 3), (r1, 4))

    cte.nextRow()
    cte.nextRow()
    cte.nextRow()


def test_agent_unordered_close_following(show=False):
    expected = {
        (1, 0): "red",
        (1, 1): "red",
        (1, 2): "red",
        (1, 3): "red",
        (3, 0): "red",
        (3, 1): "red",
        (4, 1): "purple",
        (4, 2): "purple",
        (5, 1): "purple",
        (5, 2): "purple",
        (5, 3): "purple",
        (6, 1): "red",
        (6, 2): "red",
        (6, 3): "red",
        (6, 4): "red",
        (7, 1): "purple",
        (7, 2): "purple",
        (7, 3): "purple",
        (7, 4): "purple",
        (7, 5): "purple",
        (7, 6): "purple",
        (8, 2): "blue",
        (8, 3): "red",
        (9, 4): "blue",
        (9, 5): "red",
        (9, 6): "red",
        (9, 7): "red",
        (10, 2): "blue",
        (10, 3): "red",
        (11, 2): "red",
        (12, 3): "magenta",
        (13, 3): "red",
        (14, 3): "magenta",
        (15, 3): "red",
        (15, 2): "red",
        (16, 2): "red",
        (17, 1): "red",
        (17, 2): "red",
        (17, 3): "magenta",
        (18, 3): "magenta",
        (19, 2): "red"
    }

    omc = MotionCheckLegacy()
    create_test_agents2(omc)
    omc.find_conflicts()
    actual = {i: n['color'] for i, n in omc.G.nodes.data() if 'color' in n}

    agents = {
        0: ((1, 0), (1, 1)),
        1: ((1, 1), (1, 2)),
        2: ((1, 2), (1, 3)),
        3: ((1, 3), (1, 3)),
        4: ((2, 0), (2, 1)),
        5: ((2, 1), (2, 2)),
        6: ((2, 2), (2, 3)),
        7: ((2, 3), (2, 4)),
        8: ((3, 0), (3, 1)),
        9: ((3, 1), (3, 1)),
        10: ((4, 1), (4, 2)),
        11: ((4, 2), (4, 1)),
        12: ((5, 1), (5, 2)),
        13: ((5, 2), (5, 3)),
        14: ((5, 3), (5, 2)),
        15: ((6, 1), (6, 2)),
        16: ((6, 2), (6, 3)),
        17: ((6, 3), (6, 4)),
        18: ((6, 4), (6, 4)),
        19: ((6, 5), (6, 6)),
        20: ((6, 6), (6, 7)),
        21: ((7, 1), (7, 2)),
        22: ((7, 2), (7, 3)),
        23: ((7, 3), (7, 4)),
        24: ((7, 4), (7, 3)),
        25: ((7, 5), (7, 4)),
        26: ((7, 6), (7, 5)),
        27: ((8, 1), (8, 2)),
        28: ((8, 3), (8, 2)),
        29: ((9, 1), (9, 2)),
        30: ((9, 2), (9, 3)),
        31: ((9, 3), (9, 4)),
        32: ((9, 5), (9, 4)),
        33: ((9, 6), (9, 5)),
        34: ((9, 7), (9, 6)),
        35: ((10, 1), (10, 2)),
        36: ((10, 3), (10, 2)),
        37: ((11, 2), (10, 2)),
        38: ((12, 1), (12, 2)),
        39: ((12, 2), (12, 3)),
        40: ((12, 3), (12, 4)),
        41: ((13, 3), (12, 3)),
        42: ((14, 1), (14, 2)),
        43: ((14, 2), (14, 3)),
        44: ((14, 3), (14, 4)),
        45: ((15, 3), (14, 3)),
        46: ((15, 2), (15, 3)),
        47: ((16, 2), (15, 3)),
        48: ((18, 3), (17, 3)),
        49: ((18, 2), (18, 3)),
        50: ((19, 2), (18, 3)),
        51: ((17, 1), (17, 2)),
        52: ((17, 2), (17, 3)),
        53: ((17, 3), (17, 4)),
    }

    print({
        i: omc.check_motion(i, pos) for i, (pos, _) in agents.items()
    })

    assert set(actual.keys()) == set(expected.keys())
    for k in actual.keys():
        assert expected[k] == actual[k], (k, expected[k], actual[k])

    if show:
        nx.draw(omc.G,
                with_labels=True, arrowsize=20,
                pos={p: p for p in omc.G.nodes},
                node_color=[n["color"] if "color" in n else "lightblue" for _, n in omc.G.nodes.data()]
                )
        plt.show()


def test_agent_chains_new():
    agents = {
        # stopped chain
        0: ((1, 0), (1, 1)),
        1: ((1, 1), (1, 2)),
        2: ((1, 2), (1, 3)),
        3: ((1, 3), (1, 3)),

        # running chain
        4: ((2, 0), (2, 1)),
        5: ((2, 1), (2, 2)),
        6: ((2, 2), (2, 3)),
        7: ((2, 3), (2, 4)),

        # stopped short chain
        8: ((3, 0), (3, 1)),
        9: ((3, 1), (3, 1)),

        # swap
        10: ((4, 1), (4, 2)),
        11: ((4, 2), (4, 1)),

        # mid-chain stop
        12: ((5, 1), (5, 2)),
        13: ((5, 2), (5, 3)),
        14: ((5, 3), (5, 2)),
        15: ((6, 1), (6, 2)),
        16: ((6, 2), (6, 3)),
        17: ((6, 3), (6, 4)),
        18: ((6, 4), (6, 4)),
        19: ((6, 5), (6, 6)),
        20: ((6, 6), (6, 7)),

        # mid-chain swap
        21: ((7, 1), (7, 2)),
        22: ((7, 2), (7, 3)),
        23: ((7, 3), (7, 4)),
        24: ((7, 4), (7, 3)),
        25: ((7, 5), (7, 4)),
        26: ((7, 6), (7, 5)),

        # land on same
        27: ((8, 1), (8, 2)),
        28: ((8, 3), (8, 2)),

        # chains onto same
        29: ((9, 1), (9, 2)),
        30: ((9, 2), (9, 3)),
        31: ((9, 3), (9, 4)),
        32: ((9, 5), (9, 4)),
        33: ((9, 6), (9, 5)),
        34: ((9, 7), (9, 6)),

        # 3-way same
        35: ((10, 1), (10, 2)),
        36: ((10, 3), (10, 2)),
        37: ((11, 2), (10, 2)),

        # tee
        38: ((12, 1), (12, 2)),
        39: ((12, 2), (12, 3)),
        40: ((12, 3), (12, 4)),
        41: ((13, 3), (12, 3)),

        # tree
        42: ((14, 1), (14, 2)),
        43: ((14, 2), (14, 3)),
        44: ((14, 3), (14, 4)),
        45: ((15, 3), (14, 3)),
        46: ((15, 2), (15, 3)),
        47: ((16, 2), (15, 3)),
        48: ((18, 3), (17, 3)),
        49: ((18, 2), (18, 3)),
        50: ((19, 2), (18, 3)),
        51: ((17, 1), (17, 2)),
        52: ((17, 2), (17, 3)),
        53: ((17, 3), (17, 4)),
    }
    expected = {0: False, 1: False, 2: False, 3: False, 4: True, 5: True, 6: True, 7: True, 8: False, 9: False, 10: False, 11: False, 12: False, 13: False,
                14: False, 15: False, 16: False, 17: False, 18: False, 19: True, 20: True, 21: False, 22: False, 23: False, 24: False, 25: False, 26: False,
                27: True, 28: False, 29: True, 30: True, 31: True, 32: False, 33: False, 34: False, 35: True, 36: False, 37: False, 38: True, 39: True,
                40: True, 41: False, 42: True, 43: True, 44: True, 45: False, 46: False, 47: False, 48: True, 49: True, 50: False, 51: False, 52: False,
                53: True}

    mc = MotionCheck()
    for i, (a, b) in agents.items():
        mc.add_agent(i, a, b)
    mc.find_conflicts()

    for i, b in expected.items():
        if i not in agents:
            continue
        assert mc.check_motion(i, None) == b, i


def test_edge_case():
    agents = {0: ((10, 20), (11, 20)), 1: ((24, 22), (25, 22)), 2: ((17, 17), (17, 16)), 3: ((17, 16), (16, 16)), 4: ((25, 12), (24, 12)),
              5: ((16, 16), (17, 16)), 6: ((17, 15), (17, 16))}
    mc = MotionCheck()
    for i, (r1, r2) in agents.items():
        mc.add_agent(i, r1, r2)
    mc.find_conflicts()
    assert mc.stopped == {2, 3, 5, 6}
    assert mc.deadlocked == {3, 5}
