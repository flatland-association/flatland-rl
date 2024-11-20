"""
Agent Chains: Unordered Close Following Agents

Think of a chain of agents, in random order, moving in the same direction.
For any adjacent pair of agents, there's a 0.5 chance that it is in index order, ie index(A) < index(B) where A is in front of B.
So roughly half the adjacent pairs will need to leave a gap and half won't, and the chain of agents will typically be one-third empty space.
By removing the restriction, we can keep the agents close together and
so move up to 50% more agents through a junction or segment of rail in the same number of steps.

We are still using index order to resolve conflicts between two agents trying to move into the same spot, for example, head-on collisions, or agents "merging" at junctions.

Implementation: We did it by storing an agent's position as a graph node, and a movement as a directed edge, using the NetworkX graph library.
We create an empty graph for each step, and add the agents into the graph in order,
using their (row, column) location for the node. Stationary agents get a self-loop.
Agents in an adjacent chain naturally get "connected up".
We then use some NetworkX algorithms (https://github.com/networkx/networkx):
    * `weakly_connected_components` to find the chains.
    * `selfloop_edges` to find the stopped agents
    * `dfs_postorder_nodes` to traverse a chain
    * `simple_cycles` to find agents colliding head-on
"""
from typing import Tuple, Set, Union

import graphviz as gv
import networkx as nx


class MotionCheck(object):
    """ Class to find chains of agents which are "colliding" with a stopped agent.
        This is to allow close-packed chains of agents, ie a train of agents travelling
        at the same speed with no gaps between them,
    """

    def __init__(self):
        self.G = nx.DiGraph()
        self.Grev = nx.DiGraph()  # reversed graph for finding predecessors
        self.nDeadlocks = 0
        self.svDeadlocked = set()
        self._G_reversed: Union[nx.DiGraph, None] = None

    def get_G_reversed(self):
        return self.Grev

    def reset_G_reversed(self):
        self._G_reversed = None

    def addAgent(self, iAg, rc1, rc2, xlabel=None):
        """ add an agent and its motion as row,col tuples of current and next position.
            The agent's current position is given an "agent" attribute recording the agent index.
            If an agent does not want to move this round (rc1 == rc2) then a self-loop edge is created.
            xlabel is used for test cases to give a label (see graphviz)
        """

        # Agents which have not yet entered the env have position None.
        # Substitute this for the row = -1, column = agent index
        if rc1 is None:
            rc1 = (-1, iAg)

        if rc2 is None:
            rc2 = (-1, iAg)

        self.G.add_node(rc1, agent=iAg)
        if xlabel:
            self.G.nodes[rc1]["xlabel"] = xlabel
        self.G.add_edge(rc1, rc2)
        self.Grev.add_edge(rc2, rc1)

    def find_stops2(self):
        """ alternative method to find stopped agents, using a networkx call to find selfloop edges
        """
        svStops = {u for u, v in nx.classes.function.selfloop_edges(self.G)}
        return svStops

    def find_stop_preds(self, svStops=None):
        """ Find the predecessors to a list of stopped agents (ie the nodes / vertices)
            Returns the set of predecessors.
            Includes "chained" predecessors.
        """

        if svStops is None:
            svStops = self.find_stops2()

        # Get all the chains of agents - weakly connected components.
        # Weakly connected because it's a directed graph and you can traverse a chain of agents
        # in only one direction
        lWCC = list(nx.algorithms.components.weakly_connected_components(self.G))

        svBlocked = set()
        reversed_G = None

        for oWCC in lWCC:
            if (len(oWCC) == 1):
                continue
            # print("Component:", len(oWCC), oWCC)
            # Get the node details for this WCC in a subgraph
            Gwcc = self.G.subgraph(oWCC)

            # Find all the stops in this chain or tree
            svCompStops = svStops.intersection(Gwcc)
            # print(svCompStops)

            if len(svCompStops) > 0:
                if reversed_G is None:
                    reversed_G = self.get_G_reversed()

                # We need to traverse it in reverse - back up the movement edges
                Gwcc_rev = reversed_G.subgraph(oWCC)  # Gwcc.reverse()
                for vStop in svCompStops:
                    # Find all the agents stopped by vStop by following the (reversed) edges
                    # This traverses a tree - dfs = depth first seearch
                    iter_stops = nx.algorithms.traversal.dfs_postorder_nodes(Gwcc_rev, vStop)
                    lStops = list(iter_stops)
                    svBlocked.update(lStops)

        # the set of all the nodes/agents blocked by this set of stopped nodes
        return svBlocked

    def find_swaps2(self) -> Set[Tuple[int, int]]:
        svSwaps = set()
        sEdges = self.G.edges()

        for u, v in sEdges:
            if u == v:
                # print("self loop", u, v)
                pass
            else:
                if (v, u) in sEdges:
                    # print("swap", uv)
                    svSwaps.update([u, v])
        return svSwaps

    def find_same_dest(self):
        """ find groups of agents which are trying to land on the same cell.
            ie there is a gap of one cell between them and they are both landing on it.
        """
        pass

    def block_preds(self, svStops, color="red"):
        """ Take a list of stopped agents, and apply a stop color to any chains/trees
            of agents trying to head toward those cells.
            Count the number of agents blocked, ignoring those which are already marked.
            (Otherwise it can double count swaps)

        """
        iCount = 0
        svBlocked = set()
        if len(svStops) == 0:
            return svBlocked

        # The reversed graph allows us to follow directed edges to find affected agents.
        Grev = self.get_G_reversed()
        for v in svStops:

            # Use depth-first-search to find a tree of agents heading toward the blocked cell.
            lvPred = list(nx.traversal.dfs_postorder_nodes(Grev, source=v))
            svBlocked |= set(lvPred)
            svBlocked.add(v)
            # print("node:", v, "set", svBlocked)
            # only count those not already marked
            for v2 in [v] + lvPred:
                if self.G.nodes[v2].get("color") != color:
                    self.G.nodes[v2]["color"] = color
                    iCount += 1

        return svBlocked

    def find_conflicts(self):
        self.reset_G_reversed()

        svStops = self.find_stops2()  # voluntarily stopped agents - have self-loops
        svSwaps = self.find_swaps2()  # deadlocks - adjacent head-on collisions

        # Block all swaps and their tree of predecessors
        self.svDeadlocked = self.block_preds(svSwaps, color="purple")

        # Take the union of the above, and find all the predecessors
        # svBlocked = self.find_stop_preds(svStops.union(svSwaps))

        # Just look for the tree of preds for each voluntarily stopped agent
        svBlocked = self.find_stop_preds(svStops)

        # iterate the nodes v with their predecessors dPred (dict of nodes->{})
        for (v, dPred) in self.G.pred.items():
            # mark any swaps with purple - these are directly deadlocked
            # if v in svSwaps:
            #    self.G.nodes[v]["color"] = "purple"
            # If they are not directly deadlocked, but are in the union of stopped + deadlocked
            # elif v in svBlocked:

            # if in blocked, it will not also be in a swap pred tree, so no need to worry about overwriting
            if v in svBlocked:
                self.G.nodes[v]["color"] = "red"
            # not blocked but has two or more predecessors, ie >=2 agents waiting to enter this node
            elif len(dPred) > 1:

                # if this agent is already red/blocked, ignore. CHECK: why?
                # certainly we want to ignore purple so we don't overwrite with red.
                if self.G.nodes[v].get("color") in ("red", "purple"):
                    continue

                # if this node has no agent, and >=2 want to enter it.
                if self.G.nodes[v].get("agent") is None:
                    self.G.nodes[v]["color"] = "blue"
                # this node has an agent and >=2 want to enter
                else:
                    self.G.nodes[v]["color"] = "magenta"

                # predecessors of a contended cell: {agent index -> node}
                diAgCell = {self.G.nodes[vPred].get("agent"): vPred for vPred in dPred}

                # remove the agent with the lowest index, who wins
                iAgWinner = min(diAgCell)
                diAgCell.pop(iAgWinner)

                # Block all the remaining predessors, and their tree of preds
                # for iAg, v in diAgCell.items():
                #    self.G.nodes[v]["color"] = "red"
                #    for vPred in nx.traversal.dfs_postorder_nodes(self.G.reverse(), source=v):
                #        self.G.nodes[vPred]["color"] = "red"
                self.block_preds(diAgCell.values(), "red")

    def check_motion(self, iAgent, rcPos):
        """ Returns tuple of boolean can the agent move, and the cell it will move into.
            If agent position is None, we use a dummy position of (-1, iAgent)
        """

        if rcPos is None:
            rcPos = (-1, iAgent)

        dAttr = self.G.nodes.get(rcPos)
        # print("pos:", rcPos, "dAttr:", dAttr)

        if dAttr is None:
            dAttr = {}

        # If it's been marked red or purple then it can't move
        if "color" in dAttr:
            sColor = dAttr["color"]
            if sColor in ["red", "purple"]:
                return False

        dSucc = self.G.succ[rcPos]

        # This should never happen - only the next cell of an agent has no successor
        if len(dSucc) == 0:
            print(f"error condition - agent {iAgent} node {rcPos} has no successor")
            return False

        # This agent has a successor
        rcNext = self.G.successors(rcPos).__next__()
        if rcNext == rcPos:  # the agent didn't want to move
            return False
        # The agent wanted to move, and it can
        return True


def render(omc: MotionCheck, horizontal=True):
    try:
        oAG = nx.drawing.nx_agraph.to_agraph(omc.G)
        oAG.layout("dot")
        sDot = oAG.to_string()
        if horizontal:
            sDot = sDot.replace('{', '{ rankdir="LR" ')
        # return oAG.draw(format="png")
        # This returns a graphviz object which implements __repr_svg
        return gv.Source(sDot)
    except ImportError as oError:
        print("Flatland agent_chains ignoring ImportError - install pygraphviz to render graphs")
        return None
