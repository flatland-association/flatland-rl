from typing import Tuple, List

import networkx as nx

from flatland.envs.agent_chains import MotionCheck


def create_test_agents(omc: MotionCheck):
    # blocked chain
    omc.addAgent(1, (1, 2), (1, 3))
    omc.addAgent(2, (1, 3), (1, 4))
    omc.addAgent(3, (1, 4), (1, 5))
    omc.addAgent(31, (1, 5), (1, 5))

    # unblocked chain
    omc.addAgent(4, (2, 1), (2, 2))
    omc.addAgent(5, (2, 2), (2, 3))

    # blocked short chain
    omc.addAgent(6, (3, 1), (3, 2))
    omc.addAgent(7, (3, 2), (3, 2))

    # solitary agent
    omc.addAgent(8, (4, 1), (4, 2))

    # solitary stopped agent
    omc.addAgent(9, (5, 1), (5, 1))

    # blocked short chain (opposite direction)
    omc.addAgent(10, (6, 4), (6, 3))
    omc.addAgent(11, (6, 3), (6, 3))

    # swap conflict
    omc.addAgent(12, (7, 1), (7, 2))
    omc.addAgent(13, (7, 2), (7, 1))


class ChainTestEnv(object):
    """ Just for testing agent chains
    """

    def __init__(self, omc: MotionCheck):
        self.iAgNext = 0
        self.iRowNext = 1
        self.omc = omc

    def addAgent(self, rc1, rc2, xlabel=None):
        self.omc.addAgent(self.iAgNext, rc1, rc2, xlabel=xlabel)
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
            self.omc.addAgent(iAg + self.iAgNext, rcPos, (rcPos[0] + rcVel1[0], rcPos[1] + rcVel1[1]))

        if xlabel:
            self.omc.G.nodes[lrcAgPos[0]]["xlabel"] = xlabel

        self.iAgNext += nAgents
        self.iRowNext += 1

    def nextRow(self):
        self.iRowNext += 1


def create_test_agents2(omc: MotionCheck):
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

    if False:
        cte.nextRow()
        cte.nextRow()
        cte.addAgentToRow(1, 2, "4-way\nsame")
        cte.addAgentToRow(3, 2)
        cte.addAgent((cte.iRowNext + 1, 2), (cte.iRowNext, 2))
        cte.addAgent((cte.iRowNext - 1, 2), (cte.iRowNext, 2))
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


def test_agent_following():
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
    }
    omc = MotionCheck()
    create_test_agents2(omc)
    omc.find_conflicts()
    nx.draw(omc.G,
            with_labels=True, arrowsize=20,
            pos={p: p for p in omc.G.nodes},
            node_color=[n["color"] if "color" in n else "lightblue" for _, n in omc.G.nodes.data()]
            )
    actual = {i: n['color'] for i, n in omc.G.nodes.data() if 'color' in n}

    assert set(actual.keys()) == set(expected.keys())
    for k in actual.keys():
        assert expected[k] == actual[k], (k, expected[k], actual[k])
