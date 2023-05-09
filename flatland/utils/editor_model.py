
from flatland.utils.editor_interfaces import AbstractController, AbstractModel, AbstractView

from flatland.core.grid.grid4_utils import mirror
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_generators import sparse_rail_generator, empty_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_env import RailEnv

import os
import time
import numpy as np
from numpy import array


class EditorModel(AbstractModel):
    def __init__(self, env, env_filename="temp.pkl"):
        self.view:AbstractView = None
        self.env = env
        self.regen_size_width = 10
        self.regen_size_height = 10

        self.lrcStroke = []
        self.iTransLast = -1
        self.gRCTrans = array([[-1, 0], [0, 1], [1, 0], [0, -1]])  # NESW in RC

        self.debug_bool = True
        self.debug_move_bool = False
        self.wid_output = None
        self.draw_mode = "Draw"
        self.env_filename = env_filename
        self.set_env(env)
        self.selected_agent = None
        self.thread = None
        self.save_image_count = 0

    def set_env(self, env):
        """
        set a new env for the editor, used by load and regenerate.
        """
        self.env = env

    def set_debug(self, debug):
        self.debug_bool = debug
        self.log("Set Debug:", self.debug_bool)

    def set_debug_move(self, debug):
        self.debug_move_bool = debug
        self.log("Set DebugMove:", self.debug_move_bool)

    def set_draw_mode(self, draw_mode):
        self.draw_mode = draw_mode

    def interpolate_pair(self, rcLast, rc_cell):
        if np.array_equal(rcLast, rc_cell):
            return []
        rcLast = array(rcLast)
        rc_cell = array(rc_cell)
        rcDelta = rc_cell - rcLast

        lrcInterp = []  # extra row,col points

        if np.any(np.abs(rcDelta) >= 1):
            iDim0 = np.argmax(np.abs(rcDelta))  # the dimension with the bigger move
            iDim1 = 1 - iDim0  # the dim with the smaller move
            rcRatio = rcDelta[iDim1] / rcDelta[iDim0]
            delta0 = rcDelta[iDim0]
            sgn0 = np.sign(delta0)

            iDelta1 = 0

            # count integers along the larger dimension
            for iDelta0 in range(sgn0, delta0 + sgn0, sgn0):
                rDelta1 = iDelta0 * rcRatio

                if np.abs(rDelta1 - iDelta1) >= 1:
                    rcInterp = (iDelta0, iDelta1)  # fill in the "corner" for "Manhattan interpolation"
                    lrcInterp.append(rcInterp)
                    iDelta1 = int(rDelta1)

                rcInterp = (iDelta0, int(rDelta1))
                lrcInterp.append(rcInterp)
            g2Interp = array(lrcInterp)
            if iDim0 == 1:  # if necessary, swap c,r to make r,c
                g2Interp = g2Interp[:, [1, 0]]
            g2Interp += rcLast
            # Convert the array to a list of tuples
            lrcInterp = list(map(tuple, g2Interp))
        return lrcInterp

    def interpolate_path(self, lrcPath):
        lrcPath2 = []  # interpolated version of the path
        rcLast = None
        for rcCell in lrcPath:
            if rcLast is not None:
                lrcPath2.extend(self.interpolate_pair(rcLast, rcCell))
            rcLast = rcCell
        return lrcPath2

    def drag_path_element(self, rc_cell):
        """ Mouse motion event handler for drawing.
            Only stores the row,col location of the drag, at the start,
            or when we enter a new cell, ie cross a boundary / transition.
        """
        lrcStroke = self.lrcStroke

        # Store the row,col location of the click, if we have entered a new cell
        if len(lrcStroke) > 0:
            rcLast = lrcStroke[-1]
            if not np.array_equal(rcLast, rc_cell):  # only save at transition
                lrcInterp = self.interpolate_pair(rcLast, rc_cell)
                lrcStroke.extend(lrcInterp)
                self.debug("dragpath lrcStroke ", len(lrcStroke), rc_cell, "interp:", lrcInterp)

        else:
            # This is the first cell in a mouse stroke
            lrcStroke.append(rc_cell)
            self.debug("new dragpath lrcStroke ", len(lrcStroke), rc_cell)

    def mod_path(self, bAddRemove):
        self.debug("mod_path", bAddRemove)
        # disabled functionality (no longer required)
        if bAddRemove is False:
            return
        
        # This elif means we wait until all the mouse events have been processed (black square drawn)
        # before trying to draw rails.  (We could change this behaviour)
        # Equivalent to waiting for mouse button to be lifted (and a mouse event is necessary:
        # the mouse may need to be moved)
        lrcStroke = self.lrcStroke
        if len(lrcStroke) >= 2:  # we have a stroke of at least 2 cells - ignore single cell drags
            self.debug("mod_path lrcStroke:", lrcStroke)
            self.mod_rail_cell_seq(lrcStroke, bAddRemove)
            self.redraw()

    def mod_rail_cell_seq(self, lrcStroke, bAddRemove=True):
        # If we have already touched 3 cells
        # We have a transition into a cell, and out of it.

        #print(lrcStroke)

        if len(lrcStroke) >= 2:
            # If the first cell in a stroke is empty, add a deadend to cell 0
            if self.env.rail.get_full_transitions(*lrcStroke[0]) == 0:
                self.mod_rail_2cells(lrcStroke, bAddRemove, iCellToMod=0)

        # Add transitions for groups of 3 cells
        # hence inbound and outbound transitions for middle cell
        while len(lrcStroke) >= 3:
            #print(lrcStroke)
            self.mod_rail_3cells(lrcStroke, bAddRemove=bAddRemove)

        # If final cell empty, insert deadend:
        if len(lrcStroke) == 2:
            if self.env.rail.get_full_transitions(*lrcStroke[1]) == 0:
                self.mod_rail_2cells(lrcStroke, bAddRemove, iCellToMod=1)

        #print("final:", lrcStroke)

        # now empty out the final two cells from the queue
        lrcStroke.clear()

    def mod_rail_3cells(self, lrcStroke, bAddRemove=True, bPop=True):
        """
        Add transitions for rail spanning three cells.
        lrcStroke -- list containing "stroke" of cells across grid
        bAddRemove -- whether to add (True) or remove (False) the transition
        The transition is added to or removed from the 2nd cell, consistent with
        entering from the 1st cell, and exiting into the 3rd.
        Both the forward and backward transitions are added,
        eg rcCells [(3,4), (2,4), (2,5)] would result in the transitions
        N->E and W->S in cell (2,4).
        """

        rc3Cells = array(lrcStroke[:3])  # the 3 cells
        rcMiddle = rc3Cells[1]  # the middle cell which we will update
        bDeadend = np.all(lrcStroke[0] == lrcStroke[2])  # deadend means cell 0 == cell 2

        # get the 2 row, col deltas between the 3 cells, eg [[-1,0],[0,1]] = North, East
        rc2Trans = np.diff(rc3Cells, axis=0)

        # get the direction index for the 2 transitions
        liTrans = []
        for rcTrans in rc2Trans:
            # gRCTrans - rcTrans gives an array of vector differences between our rcTrans
            # and the 4 directions stored in gRCTrans.
            # Where the vector difference is zero, we have a match...
            # np.all detects where the whole row,col vector is zero.
            # argwhere gives the index of the zero vector, ie the direction index
            iTrans = np.argwhere(np.all(self.gRCTrans - rcTrans == 0, axis=1))
            if len(iTrans) > 0:
                iTrans = iTrans[0][0]
                liTrans.append(iTrans)

        # check that we have two transitions
        if len(liTrans) == 2:
            # Set the transition
            # If this transition spans 3 cells, it is not a deadend, so remove any deadends.
            # The user will need to resolve any conflicts.
            self.env.rail.set_transition((*rcMiddle, liTrans[0]),
                                         liTrans[1],
                                         bAddRemove,
                                         remove_deadends=not bDeadend)

            # Also set the reverse transition
            # use the reversed outbound transition for inbound
            # and the reversed inbound transition for outbound
            self.env.rail.set_transition((*rcMiddle, mirror(liTrans[1])),
                                         mirror(liTrans[0]), bAddRemove, remove_deadends=not bDeadend)

        if bPop:
            lrcStroke.pop(0)  # remove the first cell in the stroke

    def mod_rail_2cells(self, lrcCells, bAddRemove=True, iCellToMod=0, bPop=False):
        """
        Add transitions for rail between two cells
        lrcCells -- list of two rc cells
        bAddRemove -- whether to add (True) or remove (False) the transition
        iCellToMod -- the index of the cell to modify: either 0 or 1
        """
        rc2Cells = array(lrcCells[:2])  # the 2 cells
        rcMod = rc2Cells[iCellToMod]  # the cell which we will update

        # get the row, col delta between the 2 cells, eg [-1,0] = North
        rc1Trans = np.diff(rc2Cells, axis=0)

        # get the direction index for the transition
        liTrans = []
        for rcTrans in rc1Trans:
            iTrans = np.argwhere(np.all(self.gRCTrans - rcTrans == 0, axis=1))
            if len(iTrans) > 0:
                iTrans = iTrans[0][0]
                liTrans.append(iTrans)

        #self.log("liTrans:", liTrans)

        # check that we have one transition
        if len(liTrans) == 1:
            # Set the transition as a deadend
            # The transition is going from cell 0 to cell 1.
            if iCellToMod == 0:
                # if 0, reverse the transition, we need to be entering cell 0
                self.env.rail.set_transition((*rcMod, mirror(liTrans[0])), liTrans[0], bAddRemove)
            else:
                # if 1, the transition is entering cell 1
                self.env.rail.set_transition((*rcMod, liTrans[0]), mirror(liTrans[0]), bAddRemove)

        if bPop:
            lrcCells.pop(0)

    def redraw(self):
        self.view.redraw()

    def clear(self):
        self.env.rail.grid[:, :] = 0
        self.env.agents = []

        self.redraw()

    def clear_cell(self, cell_row_col):
        self.debug_cell(cell_row_col)
        self.env.rail.grid[cell_row_col[0], cell_row_col[1]] = 0
        self.redraw()

    def reset(self, regenerate_schedule=False, nAgents=0):
        self.regenerate("complex", nAgents=nAgents)
        self.redraw()

    def restart_agents(self):
        self.env.reset_agents()
        self.redraw()

    def set_filename(self, filename):
        self.env_filename = filename

    def load(self):
        if os.path.exists(self.env_filename):
            self.log("load file: ", self.env_filename)
            #self.env.load(self.env_filename)
            RailEnvPersister.load(self.env, self.env_filename)
            if not self.regen_size_height == self.env.height or not self.regen_size_width == self.env.width:
                self.regen_size_height = self.env.height
                self.regen_size_width = self.env.width
                self.regenerate(None, 0, self.env)
                RailEnvPersister.load(self.env, self.env_filename)

            self.env.reset_agents()
            self.env.reset(False, False)
            self.view.oRT.update_background()
            self.fix_env()
            self.set_env(self.env)
            self.redraw()
        else:
            self.log("File does not exist:", self.env_filename, " Working directory: ", os.getcwd())

    def save(self):
        self.log("save to ", self.env_filename, " working dir: ", os.getcwd())
        #self.env.save(self.env_filename)
        RailEnvPersister.save(self.env, self.env_filename)

    def save_image(self):
        self.view.oRT.gl.save_image('frame_{:04d}.bmp'.format(self.save_image_count))
        self.save_image_count += 1
        self.view.redraw()

    def regenerate(self, method=None, nAgents=0, env=None):
        self.log("Regenerate size",
                 self.regen_size_width,
                 self.regen_size_height)

        if method is None or method == "Empty":
            fnMethod = empty_rail_generator()
        else:
            fnMethod = sparse_rail_generator(nr_start_goal=nAgents, nr_extra=20, min_dist=12, seed=int(time.time()))

        if env is None:
            self.env = RailEnv(width=self.regen_size_width, height=self.regen_size_height, rail_generator=fnMethod,
                               number_of_agents=nAgents, obs_builder_object=TreeObsForRailEnv(max_depth=2))
        else:
            self.env = env
        self.env.reset(regenerate_rail=True)
        self.fix_env()
        self.selected_agent = None  # clear the selected agent.
        self.set_env(self.env)
        self.view.new_env()
        self.redraw()

    def set_regen_width(self, size):
        self.regen_size_width = size

    def set_regen_height(self, size):
        self.regen_size_height = size

    def find_agent_at(self, cell_row_col):
        for agent_idx, agent in enumerate(self.env.agents):
            if agent.position is None:
                rc_pos = agent.initial_position
            else:
                rc_pos = agent.position
            if tuple(rc_pos) == tuple(cell_row_col):
                return agent_idx
        return None

    def click_agent(self, cell_row_col):
        """ The user has clicked on a cell -
            * If there is an agent, select it
              * If that agent was already selected, then deselect it
            * If there is no agent selected, and no agent in the cell, create one
            * If there is an agent selected, and no agent in the cell, move the selected agent to the cell
        """

        # Has the user clicked on an existing agent?
        agent_idx = self.find_agent_at(cell_row_col)

        # This is in case we still have a selected agent even though the env has been recreated
        # with no agents.
        if (self.selected_agent is not None) and (self.selected_agent > len(self.env.agents)):
            self.selected_agent = None

        # Defensive coding below - for cell_row_col to be a tuple, not a numpy array:
        # numpy array breaks various things when loading the env.

        if agent_idx is None:
            # No
            if self.selected_agent is None:
                # Create a new agent and select it.
                agent = EnvAgent(initial_position=tuple(cell_row_col),
                    initial_direction=0,
                    direction=0,
                    target=tuple(cell_row_col),
                    moving=False,
                    )
                self.selected_agent = self.env.add_agent(agent)
                # self.env.set_agent_active(agent)
                self.view.oRT.update_background()
            else:
                # Move the selected agent to this cell
                agent = self.env.agents[self.selected_agent]
                agent.initial_position = tuple(cell_row_col)
                agent.position = tuple(cell_row_col)
                agent.old_position = tuple(cell_row_col)
        else:
            # Yes
            # Have they clicked on the agent already selected?
            if self.selected_agent is not None and agent_idx == self.selected_agent:
                # Yes - deselect the agent
                self.selected_agent = None
            else:
                # No - select the agent
                self.selected_agent = agent_idx

        self.redraw()

    def add_target(self, rc_cell):
        if self.selected_agent is not None:
            self.env.agents[self.selected_agent].target = tuple(rc_cell)
            self.view.oRT.update_background()
            self.redraw()

    def fix_env(self):
        self.env.width = self.env.rail.width
        self.env.height = self.env.rail.height

    def clear_stroke(self):
        self.debug("clear_stroke - len:", len(self.lrcStroke))
        self.lrcStroke = []

    def get_len_stroke(self):
        return len(self.lrcStroke)

    def log(self, *args, **kwargs):
        if self.view is None:
            print(*args, **kwargs)
        else:
            self.view.log(*args, **kwargs)

    def debug_event(self, event:dict):
        
        if self.debug_bool:
            lsKeys = "type relative button shift ctrl alt meta".split(" ")
            sMsg = "event "
            for sKey2 in lsKeys:
                for sKey, sVal in event.items():
                    if str(sKey).startswith(sKey2):
                        sMsg += ", " + sKey + ":" + str(sVal)

            self.log(sMsg)

    def debug(self, *args, **kwargs):
        if self.debug_bool:
            self.log(*args, **kwargs)

    def debug_cell(self, rc_cell):
        binTrans = self.env.rail.get_full_transitions(*rc_cell)
        sbinTrans = format(binTrans, "#018b")[2:]
        self.debug("cell ",
                   rc_cell,
                   "Transitions: ",
                   binTrans,
                   sbinTrans,
                   [sbinTrans[i:(i + 4)] for i in range(0, len(sbinTrans), 4)])
