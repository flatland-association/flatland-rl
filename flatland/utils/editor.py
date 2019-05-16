import numpy as np
from numpy import array
import time
from collections import deque
from matplotlib import pyplot as plt
import threading
import os

# from contextlib import redirect_stdout
# import os
# import sys

# import io
# from PIL import Image
# from ipywidgets import IntSlider, link, VBox

from flatland.envs.rail_env import RailEnv, random_rail_generator
from flatland.envs.generators import complex_rail_generator
# from flatland.core.transitions import RailEnvTransitions
from flatland.envs.observations import TreeObsForRailEnv
import flatland.utils.rendertools as rt
from examples.play_model import Player
from flatland.envs.env_utils import mirror
from flatland.envs.agent_utils import EnvAgent, EnvAgentStatic

import ipywidgets
from ipywidgets import IntSlider, VBox, HBox, Checkbox, Output, Text, RadioButtons, Tab
import jpy_canvas


class EditorMVC(object):
    def __init__(self, env=None, sGL="MPL"):
        if env is None:
            env = RailEnv(width=10,
                          height=10,
                          rail_generator=random_rail_generator(),
                          number_of_agents=0,
                          obs_builder_object=TreeObsForRailEnv(max_depth=2))

        env.reset()

        self.editor = EditorModel(env)
        self.editor.view = self.view = View(self.editor, sGL=sGL)
        self.view.controller = self.editor.controller = self.controller = Controller(self.editor, self.view)
        self.view.init_canvas()
        self.view.init_widgets()   # has to be done after controller


class View(object):
    def __init__(self, editor, sGL="MPL"):
        self.editor = self.model = editor
        self.sGL = sGL

    def display(self):
        self.wOutput.clear_output()
        return self.wMain

    def init_canvas(self):
        # update the rendertool with the env
        self.new_env()
        self.oRT.renderEnv(spacing=False, arrows=False, sRailColor="gray", show=False)
        img = self.oRT.getImage()
        plt.clf()
        self.wImage = jpy_canvas.Canvas(img)
        self.yxSize = self.wImage.data.shape[:2]
        self.writableData = np.copy(self.wImage.data)  # writable copy of image - wid_img.data is somehow readonly
        self.wImage.register_move(self.controller.on_mouse_move)
        self.wImage.register_click(self.controller.on_click)

        # TODO: These are currently estimated values
        # self.yxBase = array([6, 21])  # pixel offset
        # self.nPixCell = 700 / self.model.env.rail.width  # 35
        self.yxBase = self.oRT.gl.yxBase
        self.nPixCell = self.oRT.gl.nPixCell

    def init_widgets(self):
        # Radiobutton for drawmode - TODO: replace with shift/ctrl/alt keys
        # self.wDrawMode = RadioButtons(options=["Draw", "Erase", "Origin", "Destination"])
        # self.wDrawMode.observe(self.editor.setDrawMode, names="value")

        # Debug checkbox - enable logging in the Output widget
        self.wDebug = ipywidgets.Checkbox(description="Debug")
        self.wDebug.observe(self.controller.setDebug, names="value")

        # Separate checkbox for mouse move events - they are very verbose
        self.wDebug_move = Checkbox(description="Debug mouse move")
        self.wDebug_move.observe(self.controller.setDebugMove, names="value")

        # This is like a cell widget where loggin goes
        self.wOutput = Output()

        # Filename textbox
        self.wFilename = Text(description="Filename")
        self.wFilename.value = self.model.env_filename
        self.wFilename.observe(self.controller.setFilename, names="value")

        # Size of environment when regenerating
        self.wSize = IntSlider(value=10, min=5, max=30, step=5, description="Regen Size")
        self.wSize.observe(self.controller.setRegenSize, names="value")

        # Number of Agents when regenerating
        self.wNAgents = IntSlider(value=1, min=0, max=20, step=1, description="# Agents")

        self.wRegenMethod = RadioButtons(description="Regen\nMethod", options=["Random Cell", "Path-based"])
        self.wReplaceAgents = Checkbox(value=True, description="Replace Agents")

        self.wTab = Tab()
        tab_contents = ["Debug", "Regen"]
        for i, title in enumerate(tab_contents):
            self.wTab.set_title(i, title)
        self.wTab.children = [
            VBox([self.wDebug, self.wDebug_move]),
            VBox([self.wRegenMethod, self.wReplaceAgents])]

        # Progress bar intended for stepping in the background (not yet working)
        self.wProg_steps = ipywidgets.IntProgress(value=0, min=0, max=20, step=1, description="Step")

        # abbreviated description of buttons and the methods they call
        ldButtons = [
            dict(name="Refresh", method=self.controller.refresh, tip="Redraw only"),
            dict(name="Clear", method=self.controller.clear, tip="Clear rails and agents"),
            dict(name="Reset", method=self.controller.reset,
                 tip="Standard env reset, including regen rail + agents"),
            dict(name="Restart Agents", method=self.controller.restartAgents,
                 tip="Move agents back to start positions"),
            dict(name="Regenerate", method=self.controller.regenerate,
                 tip="Regenerate the rails using the method selected below"),
            dict(name="Load", method=self.controller.load),
            dict(name="Save", method=self.controller.save),
            dict(name="Step", method=self.controller.step),
            dict(name="Run Steps", method=self.controller.start_run)]

        self.lwButtons = []
        for dButton in ldButtons:
            wButton = ipywidgets.Button(description=dButton["name"],
                                        tooltip=dButton["tip"] if "tip" in dButton else dButton["name"])
            wButton.on_click(dButton["method"])
            self.lwButtons.append(wButton)

        self.wVbox_controls = VBox([
            self.wFilename,  # self.wDrawMode,
            *self.lwButtons,
            self.wSize,
            self.wNAgents,
            self.wProg_steps,
            self.wTab])

        self.wMain = HBox([self.wImage, self.wVbox_controls])

    def drawStroke(self):
        pass

    def new_env(self):
        """ Tell the view to update its graphics when a new env is created.
        """
        self.oRT = rt.RenderTool(self.editor.env, gl=self.sGL)

    def redraw(self):
        # TODO: bit of a hack - can we suppress the console messages from MPL at source?
        # with redirect_stdout(stdout_dest):
        with self.wOutput:
            # plt.figure(figsize=(10, 10))
            self.oRT.renderEnv(spacing=False, arrows=False, sRailColor="gray",
                               show=False, iSelectedAgent=self.model.iSelectedAgent)
            img = self.oRT.getImage()
            # plt.clf()
            # plt.close()

            self.wImage.data = img
            self.writableData = np.copy(self.wImage.data)
            return img

    def redisplayImage(self):
        if self.writableData is not None:
            # This updates the image in the browser to be the new edited version
            self.wImage.data = self.writableData

    def drag_path_element(self, x, y):
        # Draw a black square on the in-memory copy of the image
        if x > 10 and x < self.yxSize[1] and y > 10 and y < self.yxSize[0]:
            self.writableData[(y - 2):(y + 2), (x - 2):(x + 2), :3] = 0

    def xy_to_rc(self, x, y):
        rcCell = ((array([y, x]) - self.yxBase) / self.nPixCell).astype(int)
        return rcCell

    def log(self, *args, **kwargs):
        if self.wOutput:
            with self.wOutput:
                print(*args, **kwargs)
        else:
            print(*args, **kwargs)


class Controller(object):
    """
    Controller to handle incoming events from the ipywidgets
    Updates the editor/model.
    Calls the View directly for things which do not directly effect the model
    (this means the mouse drag path before it is interpreted as transitions)
    """
    def __init__(self, model, view):
        self.editor = self.model = model
        self.view = view
        self.qEvents = deque()
        self.drawMode = "Draw"

    def setModel(self, model):
        self.model = model

    def on_click(self, wid, event):
        x = event['canvasX']
        y = event['canvasY']
        self.debug("debug:", event)

        rcCell = self.view.xy_to_rc(x, y)

        bShift = event["shiftKey"]
        bCtrl = event["ctrlKey"]
        if bCtrl and not bShift:
            self.model.click_agent(rcCell)
        elif bShift and bCtrl:
            self.model.add_target(rcCell)

        self.debug("click in cell", rcCell)
        self.model.debug_cell(rcCell)

    def setDebug(self, dEvent):
        self.model.setDebug(dEvent["new"])

    def setDebugMove(self, dEvent):
        self.model.setDebug_move(dEvent["new"])

    def setDrawMode(self, dEvent):
        self.drawMode = dEvent["new"]

    def setFilename(self, event):
        self.model.setFilename(event["new"])

    def on_mouse_move(self, wid, event):
        """Mouse motion event handler for drawing.
        """
        x = event['canvasX']
        y = event['canvasY']
        qEvents = self.qEvents

        if self.model.bDebug and (event["buttons"] > 0 or self.model.bDebug_move):
            self.debug("debug:", len(qEvents), event)

        # assert wid == self.wid_img, "wid not same as wid_img"

        # If the mouse is held down, enqueue an event in our own queue
        # The intention was to avoid too many redraws.
        if event["buttons"] > 0:
            qEvents.append((time.time(), x, y))

        # Process the events in our queue:
        # Draw a black square to indicate a trail
        # TODO: infer a vector of moves between these squares to avoid gaps
        # Convert the xy position to a cell rc
        # Enqueue transitions across cells in another queue
        if len(qEvents) > 0:
            tNow = time.time()
            if tNow - qEvents[0][0] > 0.1:   # wait before trying to draw
                # height, width = wid.data.shape[:2]
                # writableData = np.copy(self.wid_img.data)  # writable copy of image - wid_img.data is somehow readonly

                # with self.wid_img.hold_sync():

                while len(qEvents) > 0:
                    t, x, y = qEvents.popleft()  # get events from our queue
                    self.view.drag_path_element(x, y)

                    # Translate and scale from x,y to integer row,col (note order change)
                    # rcCell = ((array([y, x]) - self.yxBase) / self.nPixCell).astype(int)
                    rcCell = self.view.xy_to_rc(x, y)
                    self.editor.drag_path_element(rcCell)

                    #  Store the row,col location of the click, if we have entered a new cell
                    # if len(lrcStroke) > 0:
                    #     rcLast = lrcStroke[-1]
                    #     if not np.array_equal(rcLast, rcCell):  # only save at transition
                    #         # print(y, x, rcCell)
                    #         lrcStroke.append(rcCell)
                    # else:
                    #     # This is the first cell in a mouse stroke
                    #     lrcStroke.append(rcCell)
                self.view.redisplayImage()
        else:
            self.model.mod_path(not event["shiftKey"])

    def refresh(self, event):
        self.debug("refresh")
        self.view.redraw()

    def clear(self, event):
        self.model.clear()

    def reset(self, event):
        self.log("Reset - nAgents:", self.view.wNAgents.value)
        self.model.reset(replace_agents=self.view.wReplaceAgents.value,
                         nAgents=self.view.wNAgents.value)

    def restartAgents(self, event):
        self.log("Restart Agents - nAgents:", self.view.wNAgents.value)
        self.model.restartAgents()

    def regenerate(self, event):
        method = self.view.wRegenMethod.value
        nAgents = self.view.wNAgents.value
        self.model.regenerate(method, nAgents)

    def setRegenSize(self, event):
        self.model.setRegenSize(event["new"])

    def load(self, event):
        self.model.load()

    def save(self, event):
        self.model.save()

    def step(self, event):
        self.model.step()

    def start_run(self, event):
        self.model.start_run()

    def log(self, *args, **kwargs):
        if self.view is None:
            print(*args, **kwargs)
        else:
            self.view.log(*args, **kwargs)

    def debug(self, *args, **kwargs):
        self.model.debug(*args, **kwargs)


class EditorModel(object):
    def __init__(self, env):
        self.view = None
        self.env = env
        self.regen_size = 10

        self.lrcStroke = []
        self.iTransLast = -1
        self.gRCTrans = array([[-1, 0], [0, 1], [1, 0], [0, -1]])  # NESW in RC

        self.bDebug = False
        self.bDebug_move = False
        self.wid_output = None
        self.drawMode = "Draw"
        self.env_filename = "temp.pkl"
        self.set_env(env)
        self.iSelectedAgent = None
        self.player = None
        self.thread = None

    def set_env(self, env):
        """
        set a new env for the editor, used by load and regenerate.
        """
        self.env = env
        # self.yxBase = array([6, 21])  # pixel offset
        # self.nPixCell = 700 / self.env.rail.width  # 35
        # self.oRT = rt.RenderTool(env)

    def setDebug(self, bDebug):
        self.bDebug = bDebug
        self.log("Set Debug:", self.bDebug)

    def setDebugMove(self, bDebug):
        self.bDebug_move = bDebug
        self.log("Set DebugMove:", self.bDebug_move)

    def setDrawMode(self, sDrawMode):
        self.drawMode = sDrawMode

    def drag_path_element(self, rcCell):
        """Mouse motion event handler for drawing.
        """
        lrcStroke = self.lrcStroke

        # Store the row,col location of the click, if we have entered a new cell
        if len(lrcStroke) > 0:
            rcLast = lrcStroke[-1]
            if not np.array_equal(rcLast, rcCell):  # only save at transition
                lrcStroke.append(rcCell)
                self.debug("lrcStroke ", len(lrcStroke), rcCell)

        else:
            # This is the first cell in a mouse stroke
            lrcStroke.append(rcCell)
            self.debug("lrcStroke ", len(lrcStroke), rcCell)

    def mod_path(self, bAddRemove):
        # This elif means we wait until all the mouse events have been processed (black square drawn)
        # before trying to draw rails.  (We could change this behaviour)
        # Equivalent to waiting for mouse button to be lifted (and a mouse event is necessary:
        # the mouse may need to be moved)
        lrcStroke = self.lrcStroke
        if len(lrcStroke) >= 2:
            self.mod_rail_cell_seq(lrcStroke, bAddRemove)
            self.redraw()

    def mod_rail_cell_seq(self, lrcStroke, bAddRemove=True):
        # If we have already touched 3 cells
        # We have a transition into a cell, and out of it.

        if len(lrcStroke) >= 2:
            # If the first cell in a stroke is empty, add a deadend to cell 0
            if self.env.rail.get_transitions(lrcStroke[0]) == 0:
                self.mod_rail_2cells(lrcStroke, bAddRemove, iCellToMod=0)

        # Add transitions for groups of 3 cells
        # hence inbound and outbound transitions for middle cell
        while len(lrcStroke) >= 3:
            self.mod_rail_3cells(lrcStroke, bAddRemove=bAddRemove)

        # If final cell empty, insert deadend:
        if len(lrcStroke) == 2:
            if self.env.rail.get_transitions(lrcStroke[1]) == 0:
                self.mod_rail_2cells(lrcStroke, bAddRemove, iCellToMod=1)

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

        # Save the original state of the cell
        # oTransrcMiddle = self.env.rail.get_transitions(rcMiddle)
        # sTransrcMiddle = self.env.rail.cell_repr(rcMiddle)

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

            # bValid = self.env.rail.is_cell_valid(rcMiddle)
            # if not bValid:
            #    # Reset cell transition values
            #    self.env.rail.grid[tuple(rcMiddle)] = oTransrcMiddle

        # self.log(rcMiddle, "Orig:", sTransrcMiddle, "Mod:", self.env.rail.cell_repr(rcMiddle))
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
        # self.env.number_of_agents = 0
        self.env.agents = []
        self.env.agents_static = []
        # self.env.agents_handles = []
        self.player = None

        self.redraw()

    def reset(self, replace_agents=False, nAgents=0):
        # if replace_agents:
        #    self.env.agents_handles = range(nAgents)
        self.env.reset(regen_rail=True, replace_agents=replace_agents)
        self.player = Player(self.env)
        self.redraw()

    def restartAgents(self):
        self.env.agents = EnvAgent.list_from_static(self.env.agents_static)
        self.player = Player(self.env)
        self.redraw()

    def setFilename(self, filename):
        self.log("filename = ", filename, type(filename))
        self.env_filename = filename

    def load(self):
        if os.path.exists(self.env_filename):
            self.log("load file: ", self.env_filename)
            # self.env.rail.load_transition_map(self.env_filename, override_gridsize=True)
            self.env.load(self.env_filename)
            self.fix_env()
            self.set_env(self.env)
            self.redraw()
        else:
            self.log("File does not exist:", self.env_filename, " Working directory: ", os.getcwd())

    def save(self):
        self.log("save to ", self.env_filename, " working dir: ", os.getcwd())
        # self.env.rail.save_transition_map(self.env_filename)
        self.env.save(self.env_filename)

    def regenerate(self, method=None, nAgents=0):
        self.log("Regenerate size", self.regen_size)

        if method is None or method == "Random Cell":
            fnMethod = random_rail_generator(cell_type_relative_proportion=[1] * 11)
        else:
            fnMethod = complex_rail_generator(nr_start_goal=5, nr_extra=20, min_dist=12)

        self.env = RailEnv(width=self.regen_size,
                           height=self.regen_size,
                           rail_generator=fnMethod,
                           # number_of_agents=self.env.get_num_agents(),
                           number_of_agents=nAgents,
                           obs_builder_object=TreeObsForRailEnv(max_depth=2))
        self.env.reset(regen_rail=True)
        self.fix_env()
        self.set_env(self.env)
        self.player = Player(self.env)
        self.view.new_env()
        self.redraw()

    def setRegenSize(self, size):
        self.regen_size = size

    def find_agent_at(self, rcCell):
        for iAgent, agent in enumerate(self.env.agents_static):
            if tuple(agent.position) == tuple(rcCell):
                return iAgent
        return None

    def click_agent(self, rcCell):
        """ The user has clicked on a cell -
            - If there is an agent, select it
                - If that agent was already selected, then deselect it
            - If there is no agent selected, and no agent in the cell, create one
            - If there is an agent selected, and no agent in the cell, move the selected agent to the cell
        """

        # Has the user clicked on an existing agent?
        iAgent = self.find_agent_at(rcCell)

        if iAgent is None:
            # No
            if self.iSelectedAgent is None:
                # Create a new agent and select it.
                agent_static = EnvAgentStatic(rcCell, 0, rcCell)
                self.iSelectedAgent = self.env.add_agent_static(agent_static)
                self.player = None  # will need to start a new player
            else:
                # Move the selected agent to this cell
                agent_static = self.env.agents_static[self.iSelectedAgent]
                agent_static.position = rcCell
        else:
            # Yes
            # Have they clicked on the agent already selected?
            if self.iSelectedAgent is not None and iAgent == self.iSelectedAgent:
                # Yes - deselect the agent
                self.iSelectedAgent = None
            else:
                # No - select the agent
                self.iSelectedAgent = iAgent

        self.redraw()

    def add_target(self, rcCell):
        if self.iSelectedAgent is not None:
            self.env.agents_static[self.iSelectedAgent].target = rcCell
            self.redraw()

    def step(self):
        if self.player is None:
            self.player = Player(self.env)
            self.env.reset(regen_rail=False, replace_agents=False)
        self.player.step()
        self.redraw()

    def start_run(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.bg_updater, args=(self.view.wProg_steps,))
            self.thread.start()
        else:
            self.log("thread already present")

    def bg_updater(self, wProg_steps):
        try:
            for i in range(20):
                # self.log("step ", i)
                self.step()
                time.sleep(0.2)
                wProg_steps.value = i + 1   # indicate progress on bar
        finally:
            self.thread = None

    def fix_env(self):
        self.env.width = self.env.rail.width
        self.env.height = self.env.rail.height

    def log(self, *args, **kwargs):
        if self.view is None:
            print(*args, **kwargs)
        else:
            self.view.log(*args, **kwargs)

    def debug(self, *args, **kwargs):
        if self.bDebug:
            self.log(*args, **kwargs)

    def debug_cell(self, rcCell):
        binTrans = self.env.rail.get_transitions(rcCell)
        sbinTrans = format(binTrans, "#018b")[2:]
        self.debug("cell ",
                   rcCell,
                   "Transitions: ",
                   binTrans,
                   sbinTrans,
                   [sbinTrans[i:(i + 4)] for i in range(0, len(sbinTrans), 4)])
