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
using their (row, column) location for the node. In this way, agents staying in the same cell (stop action or not at cell exit yet) get a self-loop.
Agents in an adjacent chain naturally get "connected up".

Pseudocode:
* purple = deadlocked if in deadlock or predecessor of deadlocked (`mark_preds(find_swaps(), 'purple')`)
* red = blocked, i.e. wanting to move, but blocked by an agent ahead not wanting to move or blocked itself (`mark_preds(find_stopped_agents(), 'red')`)
* blue = no agent and >1 wanting to enter, blocking after conflict resolution (`mark_preds(losers, 'red')`)
* magenta: agent (able to move) and >1 wanting to enter, blocking after conflict resolution (`mark_preds(losers, 'red')`)


We then use some NetworkX algorithms (https://github.com/networkx/networkx):
    * `weakly_connected_components` to find the chains.
    * `selfloop_edges` to find the stopped agents
    * `dfs_postorder_nodes` to traverse a chain
"""
from typing import Tuple, Dict, Any

AgentHandle = int
Cell = Tuple[int, int]
from typing import Set

import networkx as nx


class MotionCheck(object):
    """ Class to find chains of agents which are "colliding" with a stopped agent.
        This is to allow close-packed chains of agents, ie a train of agents travelling
        at the same speed with no gaps between them,
    """

    def __init__(self):
        self.G = nx.DiGraph()  # nodes of type `Cell`
        # TODO do we need the reversed graph at all?
        self.Grev = nx.DiGraph()  # reversed graph for finding predecessors
        self.nDeadlocks = 0
        self.svDeadlocked = set()

    def get_G_reversed(self):
        return self.Grev

    def addAgent(self, iAg: AgentHandle, rc1: Cell, rc2: Cell, xlabel=None):
        """ add an agent and its motion as row,col tuples of current and next position.
            The agent's current position is given an "agent" attribute recording the agent index.
            If an agent does not want to move this round (rc1 == rc2) then a self-loop edge is created.
            xlabel is used for test cases to give a label (see graphviz)
        """

        # Agents which have not yet entered the env have position None.
        # Substitute this for the row = -1, column = agent index, i.e. they are isolated nodes in the graph!
        if rc1 is None:
            rc1 = (-1, iAg)

        if rc2 is None:
            rc2 = (-1, iAg)

        self.G.add_node(rc1, agent=iAg)
        if xlabel:
            self.G.nodes[rc1]["xlabel"] = xlabel
        self.G.add_edge(rc1, rc2)
        self.Grev.add_edge(rc2, rc1)

    def find_stopped_agents(self) -> Set[Cell]:
        """
        Find stopped agents, using a networkx call to find self-loop nodes.
        :return: set of stopped agents
        """
        svStops = {u for u, v in nx.classes.function.selfloop_edges(self.G)}
        return svStops

    def find_stop_preds(self, svStops: Set[Cell]) -> Set[Cell]:
        """ Find the predecessors to a list of stopped agents (ie the nodes / vertices). Includes "chained" predecessors.
            :param svStops: list of voluntarily stopped agents
            :return: the set of predecessors.
        """

        # Get all the chains of agents - weakly connected components.
        # Weakly connected because it's a directed graph and you can traverse a chain of agents
        # in only one direction
        # TODO why do we need weakly connected components at all? Just use reverse traversal of directed edges?
        lWCC = list(nx.algorithms.components.weakly_connected_components(self.G))

        svBlocked = set()
        reversed_G = self.get_G_reversed()

        for oWCC in lWCC:
            if (len(oWCC) == 1):
                continue
            # Get the node details for this WCC in a subgraph
            Gwcc: Set[Cell] = self.G.subgraph(oWCC)

            # Find all the stops in this chain or tree
            svCompStops: Set[Cell] = svStops.intersection(Gwcc)

            if len(svCompStops) > 0:

                # We need to traverse it in reverse - back up the movement edges
                Gwcc_rev = reversed_G.subgraph(oWCC)
                for vStop in svCompStops:
                    # Find all the agents stopped by vStop by following the (reversed) edges
                    # This traverses a tree - dfs = depth first seearch
                    iter_stops = nx.algorithms.traversal.dfs_postorder_nodes(Gwcc_rev, vStop)
                    lStops = list(iter_stops)
                    svBlocked.update(lStops)

        # the set of all the nodes/agents blocked by this set of stopped nodes
        return svBlocked

    def find_swaps(self) -> Set[Cell]:
        """
        Find loops of size 2 in the graph, i.e. swaps leading to head-on collisions.
        :return: set of all cells in swaps.
        """
        svSwaps = set()
        sEdges = self.G.edges()

        for u, v in sEdges:
            if u == v:
                pass
            else:
                if (v, u) in sEdges:
                    svSwaps.update([u, v])
        return svSwaps

    def mark_preds(self, svStops: Set[Cell], color: object = "red") -> Set[Cell]:
        """ Take a list of stopped agents, and apply a stop color to any chains/trees
            of agents trying to head toward those cells.
            :param svStops: list of stopped agents
            :param color: color to apply to predecessor of stopped agents
            :return: all predecessors of any stopped agent

        """
        predecessors = set()
        if len(svStops) == 0:
            return predecessors

        # The reversed graph allows us to follow directed edges to find affected agents.
        Grev = self.get_G_reversed()
        for v in svStops:

            # Use depth-first-search to find a tree of agents heading toward the blocked cell.
            lvPred = list(nx.traversal.dfs_postorder_nodes(Grev, source=v))
            predecessors |= set(lvPred)
            predecessors.add(v)

            # only color those not already marked (not updating previous colors)
            for v2 in [v] + lvPred:
                if self.G.nodes[v2].get("color") != color:
                    self.G.nodes[v2]["color"] = color
        return predecessors

    def find_conflicts(self):
        """Called in env.step() before the agents execute their actions."""

        svStops: Set[Cell] = self.find_stopped_agents()  # voluntarily stopped agents - have self-loops ("same cell to same cell")
        svSwaps: Set[Cell] = self.find_swaps()  # deadlocks - adjacent head-on collisions

        # Mark all swaps and their tree of predecessors with purple - these are directly deadlocked
        self.svDeadlocked: Set[Cell] = self.mark_preds(svSwaps, color="purple")

        # Just look for the tree of preds for each voluntarily stopped agent (i.e. not wanting to move)
        # TODO why not re-use mark_preds(swStops, color="red")?
        # TODO refactoring suggestion: only one "blocked" red = 1. all deadlocked and their predecessor, 2. all predecessors of self-loopers, 3.
        svBlocked: Set[Cell] = self.find_stop_preds(svStops)

        # iterate the nodes v with their predecessors dPred (dict of nodes->{})
        for (v, dPred) in self.G.pred.items():

            dPred: Set[Cell] = dPred

            # if in blocked, it will not also be in a swap pred tree, so no need to worry about overwriting (outdegree always  <= 1!)
            # TODO why not mark outside of the loop? The loop would then only need to go over nodes with indegree >2 not marked purple or red yet
            if v in svBlocked:
                self.G.nodes[v]["color"] = "red"

            # not blocked but has two or more predecessors, ie >=2 agents waiting to enter this node
            elif len(dPred) > 1:
                # if this agent is already red or purple, all its predecessors are in svDeadlocked or svBlocked and will eventually be marked red or purple

                # no conflict resolution if deadlocked or blocked
                if self.G.nodes[v].get("color") in ("red", "purple"):
                    continue

                # if this node has no agent, and >=2 want to enter it.
                if self.G.nodes[v].get("agent") is None:
                    self.G.nodes[v]["color"] = "blue"
                # this node has an agent and >=2 want to enter
                else:
                    self.G.nodes[v]["color"] = "magenta"

                # predecessors of a contended cell: {agent index -> node}
                diAgCell: Dict[AgentHandle, Cell] = {self.G.nodes[vPred].get("agent"): vPred for vPred in dPred}

                # remove the agent with the lowest index, who wins
                iAgWinner = min(diAgCell)
                diAgCell.pop(iAgWinner)

                self.mark_preds(set(diAgCell.values()), "red")

    def check_motion(self, iAgent: AgentHandle, rcPos: Cell) -> bool:
        """ Returns tuple of boolean can the agent move, and the cell it will move into.
            If agent position is None, we use a dummy position of (-1, iAgent).
            Called in env.step() after conflicts are collected in find_conflicts() - each agent now can execute their position update independently (valid_movement) by calling check_motion.
            :param iAgent: agent handle
            :param rcPos:  cell
            :return: true iff the agent wants to move and it has no conflict
        """

        if rcPos is None:
            # no successor
            rcPos = (-1, iAgent)

        dAttr = self.G.nodes.get(rcPos)

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


class MotionCheck2(object):
    """
    Agent-Close Following is an edge cose in mutual exclusive resource allocation, i.e. the same resource can be hold by at most one agent at the same time.
    In Flatland, grid cell is a resource, and only one agent can be in a grid cell.
    The only exception are level-free crossings, where there is horizontal and a vertical resource at the same grid cell.

    The naive solution to this problem is to iterate among all agents, verify that their targeted resource is not occupied, if so allow the movement, else stop the agent.
    However, when agents follow each other in chain of cells, different agent orderings lead to different decisions under this algorithm.

    MotionCheck ensures
    - no swaps (i.e. collisions)
    - no two agents must be allowed to move to the same target cell
    - if all agents in the chain run at the same speeed, all can run (behaviour not depending on agent indices)

    Implementation based on Bochatay (2024), Speeding up Railway Generation and Train Simulation for the Flatland Challenge.

    An alternative would be to introduce > 0 release times and then reject

    Release times reflect physical or IT constraints for the system to be ready again, or buffer times ensuring safety constraints.

    Release times known also in Job Shop Scheduling Problems in the Operations Research literature, see e.g. Bürgy (2014), Complex Job Shop Scheduling: A General Model and Method (https://reinhardbuergy.ch/research/pdf/buergy14_phd-Thesis.pdf)
    """

    def __init__(self):
        # TODO init with size to avoid
        # Agent -> (Resource,Resource)
        self.agents = {}
        # Resource -> List[Agent]
        self.reverse_target = {}
        # List[Tuple[Agent,Agent]] # a1 < a2
        self.conflicts = []
        # Set[Agent]
        self.stopped = set()

        # TODO we do not need to process deadlocked any more!
        self.deadlocked = set()

    def addAgent(self, iAg: int, rc1: Any, rc2: Any):
        """ add an agent and its motion as row,col tuples of current and next position.
            The agent's current position is given an "agent" attribute recording the agent index.
            If an agent does not want to move this round (rc1 == rc2) then a self-loop edge is created.
            xlabel is used for test cases to give a label (see graphviz)
        """
        if rc1 is None:
            rc1 = (None, iAg)
        if rc2 is None:
            rc2 = (None, iAg)
        self.agents[iAg] = (rc1, rc2)
        if rc2 not in self.reverse_target:
            self.reverse_target[rc2] = [iAg]
        else:
            self.reverse_target[rc2].append(iAg)

    # TODO refactor
    def find_conflicts(self):
        # TODO can we prune more? are edges sets or tuples?

        # construct_graph
        for iAg, (pos, target) in self.agents.items():
            conflict_list = []
            # TODO simplify, do not add swaps
            if pos in self.reverse_target:
                conflict_list += self.reverse_target[pos]
            if target in self.reverse_target:
                conflict_list += self.reverse_target[target]
            for a2 in conflict_list:
                if iAg >= a2:
                    continue
                a2pos, a2_target = self.agents[a2]
                # same target
                if a2_target == target:
                    self.conflicts.append((iAg, a2))
                # swap/collision
                elif pos == a2_target and target == a2pos:
                    self.stopped.add(iAg)
                    self.stopped.add(a2)
                    self.deadlocked.add(iAg)
                    self.deadlocked.add(a2)
                    # TODO these should be marked deadlocked

            if pos == target:
                self.stopped.add(iAg)
        for a in self.deadlocked:
            self.agents[a] = (self.agents[a][0], self.agents[a][0])

        # fix_conflicts:
        while len(self.conflicts) > 0:
            u, v = self.conflicts[0]
            u_pos, u_target = self.agents[u]
            v_pos, v_target = self.agents[v]

            if v_pos == v_target:
                # if v is stopped or does not want to move, stop u
                self.agents[u] = (u_pos, u_pos)
                self.stopped.add(u)

                # update_graph and reverse_target
                if u_pos in self.reverse_target:
                    self.conflicts += [(u, other) if u < other else (other, u) for other in self.reverse_target[u_pos] if other != u]
                    self.reverse_target[u_pos].append(u)
                else:
                    self.reverse_target[u_pos] = [u]
                self.reverse_target[u_target].remove(u)
            else:
                # stop v, which has larger index
                self.agents[v] = (v_pos, v_pos)
                self.stopped.add(v)

                # update_graph and reverse_target
                if v_pos in self.reverse_target:
                    self.conflicts += [(v, other) if v < other else (other, v) for other in self.reverse_target[v_pos] if other != v]
                    self.reverse_target[v_pos].append(v)
                else:
                    self.reverse_target[v_pos] = [v]
                self.reverse_target[v_target].remove(v)
            self.conflicts = self.conflicts[1:]

    def check_motion(self, iAgent: int, rcPos: Any) -> bool:
        """ Returns tuple of boolean can the agent move, and the cell it will move into.
            If agent position is None, we use a dummy position of (-1, iAgent)
        """
        return iAgent not in self.stopped

# Moving in a chain would be non-intelligible to ML agents as depending on the order of processing. Reject would not help!
# Remaining only conflict is now if two agents into same cell, here choose by agent index (we could also reject).
