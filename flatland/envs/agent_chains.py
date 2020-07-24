
import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
from typing import List, Tuple

class MotionCheck(object):
    """ Class to find chains of agents which are "colliding" with a stopped agent.
        This is to allow close-packed chains of agents, ie a train of agents travelling
        at the same speed with no gaps between them,
    """
    def __init__(self):
        self.G = nx.DiGraph()

    def addAgent(self, iAg, rc1, rc2, xlabel=None):
        """ add an agent and its motion as row,col tuples of current and next position.
            The agent's current position is given an "agent" attribute recording the agent index.
            If an agent does not move this round then its cell is 
        """
        self.G.add_node(rc1, agent=iAg)
        if xlabel:
            self.G.nodes[rc1]["xlabel"] = xlabel
        self.G.add_edge(rc1, rc2)
    
    def find_stops(self):
        """ find all the stopped agents as a set of rc position nodes
            A stopped agent is a self-loop on a cell node.
        """
        
        # get the (sparse) adjacency matrix
        spAdj = nx.linalg.adjacency_matrix(self.G)

        # the stopped agents appear as 1s on the diagonal
        # the where turns this into a list of indices of the 1s
        giStops = np.where(spAdj.diagonal())[0]

        # convert the cell/node indices into the node rc values 
        lvAll = list(self.G.nodes())
        # pick out the stops by their indices
        lvStops = [ lvAll[i] for i in giStops ]
        # make it into a set ready for a set intersection
        svStops = set(lvStops)
        return svStops

    def find_stops2(self):
        """ alternative method to find stopped agents, using a networkx call to find selfloop edges
        """
        svStops = { u for u,v in nx.classes.function.selfloop_edges(self.G) }
        return svStops

    def find_stop_preds(self, svStops=None):

        if svStops is None:
            svStops = self.find_stops2()

        # Get all the chains of agents - weakly connected components.
        # Weakly connected because it's a directed graph and you can traverse a chain of agents
        # in only one direction
        lWCC = list(nx.algorithms.components.weakly_connected_components(self.G))

        svBlocked = set()

        for oWCC in lWCC:
            #print("Component:", oWCC)
            Gwcc = self.G.subgraph(oWCC)
            
            #lChain = list(nx.topological_sort(Gwcc))
            #print("path:     ", lChain)
            
            # Find all the stops in this chain
            svCompStops = svStops.intersection(Gwcc)
            #print(svCompStops)
            
            if len(svCompStops) > 0:
                print("component contains a stop")
                for vStop in svCompStops:
                    
                    iter_stops = nx.algorithms.traversal.dfs_postorder_nodes(Gwcc.reverse(), vStop)
                    lStops = list(iter_stops)
                    print(vStop, "affected preds:", lStops)
                    svBlocked.update(lStops)
        
        return svBlocked

    def find_swaps(self):
        """ find all the swap conflicts where two agents are trying to exchange places.
            These appear as simple cycles of length 2.
        """
        #svStops = self.find_stops2()
        llvLoops = list(nx.algorithms.cycles.simple_cycles(self.G))
        llvSwaps = [lvLoop for lvLoop in llvLoops if len(lvLoop) == 2 ]
        return llvSwaps
    
    def find_same_dest(self):
        """ find groups of agents which are trying to land on the same cell.
            ie there is a gap of one cell between them and they are both landing on it.
        """
        


def render(omc:MotionCheck):
    oAG = nx.drawing.nx_agraph.to_agraph(omc.G)
    oAG.layout("dot")
    return oAG.draw(format="png")

class ChainTestEnv(object):
    """ Just for testing agent chains
    """
    def __init__(self, omc:MotionCheck):
        self.iAgNext = 0
        self.iRowNext = 1
        self.omc = omc

    def addAgent(self, rc1, rc2, xlabel=None):
        self.omc.addAgent(self.iAgNext, rc1, rc2, xlabel=xlabel)
        self.iAgNext+=1

    def addAgentToRow(self, c1, c2, xlabel=None):
        self.addAgent((self.iRowNext, c1), (self.iRowNext, c2), xlabel=xlabel)
        

    def create_test_chain(self, 
            nAgents:int, 
            rcVel:Tuple[int] = (0,1),
            liStopped:List[int]=[],
            xlabel=None):
        """ create a chain of agents
        """
        lrcAgPos = [ (self.iRowNext, i * rcVel[1]) for i in range(nAgents) ]

        for iAg, rcPos in zip(range(nAgents), lrcAgPos):
            if iAg in liStopped:
                rcVel1 = (0,0)
            else:
                rcVel1 = rcVel
            self.omc.addAgent(iAg+self.iAgNext, rcPos, (rcPos[0] + rcVel1[0], rcPos[1] + rcVel1[1]) )
        
        if xlabel:
            self.omc.G.nodes[lrcAgPos[0]]["xlabel"] = xlabel

        self.iAgNext += nAgents
        self.iRowNext += 1

    def nextRow(self):
        self.iRowNext+=1

    

def create_test_agents(omc:MotionCheck):

    # blocked chain
    omc.addAgent(1, (1,2), (1,3))
    omc.addAgent(2, (1,3), (1,4))
    omc.addAgent(3, (1,4), (1,5))
    omc.addAgent(31, (1,5), (1,5))

    # unblocked chain
    omc.addAgent(4, (2,1), (2,2))
    omc.addAgent(5, (2,2), (2,3))

    # blocked short chain
    omc.addAgent(6, (3,1), (3,2))
    omc.addAgent(7, (3,2), (3,2))

    # solitary agent
    omc.addAgent(8, (4,1), (4,2))

    # solitary stopped agent
    omc.addAgent(9, (5,1), (5,1))

    # blocked short chain (opposite direction)
    omc.addAgent(10, (6,4), (6,3))
    omc.addAgent(11, (6,3), (6,3))

    # swap conflict
    omc.addAgent(12, (7,1), (7,2))
    omc.addAgent(13, (7,2), (7,1))


def create_test_agents2(omc:MotionCheck):

    # blocked chain
    cte = ChainTestEnv(omc)
    cte.create_test_chain(4, liStopped=[3], xlabel="stopped\nchain")
    cte.create_test_chain(4, xlabel="running\nchain")

    cte.create_test_chain(2, liStopped = [1], xlabel="stopped \nshort\n chain")

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
    cte.addAgent((cte.iRowNext+1, 2), (cte.iRowNext, 2))
    cte.nextRow()
    
    if False:
        cte.nextRow()
        cte.nextRow()
        cte.addAgentToRow(1, 2, "4-way\nsame")
        cte.addAgentToRow(3, 2)
        cte.addAgent((cte.iRowNext+1, 2), (cte.iRowNext, 2))
        cte.addAgent((cte.iRowNext-1, 2), (cte.iRowNext, 2))
        cte.nextRow()

    cte.nextRow()
    cte.addAgentToRow(1, 2, "Tee")
    cte.addAgentToRow(2, 3)
    cte.addAgentToRow(3, 4)
    cte.addAgent((cte.iRowNext+1, 3), (cte.iRowNext, 3))
    cte.nextRow()
    


def test_agent_following():
    omc = MotionCheck()
    create_test_agents2(omc)

    svStops = omc.find_stops()
    svBlocked = omc.find_stop_preds()
    llvSwaps = omc.find_swaps()
    svSwaps = { v for lvSwap in llvSwaps for v in lvSwap }
    print(list(svBlocked))

    lvCells = omc.G.nodes()

    lColours = [ "magenta" if v in svStops 
            else "red" if v in svBlocked 
            else "purple" if v in svSwaps
            else "lightblue"
            for v in lvCells ]
    dPos = dict(zip(lvCells, lvCells))

    #plt.ion()
    nx.draw(omc.G, 
        with_labels=True, arrowsize=20, 
        pos=dPos,
        node_color = lColours)

    
    #plt.pause(20)
    #plt.show()
    


def main():

    test_agent_following()

if __name__=="__main__":
    main()