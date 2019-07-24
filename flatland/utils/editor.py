import os
import time
from collections import deque

import ipywidgets
import jpy_canvas
import numpy as np
from ipywidgets import IntSlider, VBox, HBox, Checkbox, Output, Text, RadioButtons, Tab
from numpy import array

import flatland.utils.rendertools as rt
from flatland.core.grid.grid4_utils import mirror
from flatland.envs.agent_utils import EnvAgent, EnvAgentStatic
from flatland.envs.generators import complex_rail_generator, empty_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv, random_rail_generator


class EditorMVC(object):
    """ EditorMVC - a class to encompass and assemble the Jupyter Editor Model-View-Controller.
    """

    def __init__(self, env=None, sGL="PIL"):
        """ Create an Editor MVC assembly around a railenv, or create one if None.
        """
        if env is None:
            env = RailEnv(width=10,
                          height=10,
                          rail_generator=empty_rail_generator(),
                          number_of_agents=0,
                          obs_builder_object=TreeObsForRailEnv(max_depth=2))

        env.reset()

        self.editor = EditorModel(env)
        self.editor.view = self.view = View(self.editor, sGL=sGL)
        self.view.controller = self.editor.controller = self.controller = Controller(self.editor, self.view)
        self.view.init_canvas()
        self.view.init_widgets()  # has to be done after controller


class View(object):
    """ The Jupyter Editor View - creates and holds the widgets comprising the Editor.
    """

    def __init__(self, editor, sGL="MPL"):
        self.editor = self.model = editor
        self.sGL = sGL

    def display(self):
        self.output_generator.clear_output()
        return self.wMain

    def init_canvas(self):
        # update the rendertool with the env
        self.new_env()
        self.oRT.render_env(show=False)
        img = self.oRT.get_image()
        self.wImage = jpy_canvas.Canvas(img)
        self.yxSize = self.wImage.data.shape[:2]
        self.writableData = np.copy(self.wImage.data)  # writable copy of image - wid_img.data is somehow readonly
        self.wImage.register_move(self.controller.on_mouse_move)
        self.wImage.register_click(self.controller.on_click)

        self.yxBase = self.oRT.gl.yxBase
        self.nPixCell = self.oRT.gl.nPixCell

    def init_widgets(self):
        # Debug checkbox - enable logging in the Output widget
        self.debug = ipywidgets.Checkbox(description="Debug")
        self.debug.observe(self.controller.set_debug, names="value")

        # Separate checkbox for mouse move events - they are very verbose
        self.debug_move = Checkbox(description="Debug mouse move")
        self.debug_move.observe(self.controller.set_debug_move, names="value")

        # This is like a cell widget where loggin goes
        self.output_generator = Output()

        # Filename textbox
        self.filename = Text(description="Filename")
        self.filename.value = self.model.env_filename
        self.filename.observe(self.controller.set_filename, names="value")

        # Size of environment when regenerating

        self.regen_width = IntSlider(value=10, min=5, max=100, step=5, description="Regen Size (Width)",
                                     tip="Click Regenerate after changing this")
        self.regen_width.observe(self.controller.set_regen_width, names="value")

        self.regen_height = IntSlider(value=10, min=5, max=100, step=5, description="Regen Size (Height)",
                                      tip="Click Regenerate after changing this")
        self.regen_height.observe(self.controller.set_regen_height, names="value")

        # Number of Agents when regenerating
        self.regen_n_agents = IntSlider(value=1, min=0, max=5, step=1, description="# Agents",
                                        tip="Click regenerate or reset after changing this")
        self.regen_method = RadioButtons(description="Regen\nMethod", options=["Empty", "Random Cell"])

        self.replace_agents = Checkbox(value=True, description="Replace Agents")

        self.wTab = Tab()
        tab_contents = ["Regen", "Observation"]
        for i, title in enumerate(tab_contents):
            self.wTab.set_title(i, title)
        self.wTab.children = [
            VBox([self.regen_width, self.regen_height, self.regen_n_agents, self.regen_method])
        ]

        # abbreviated description of buttons and the methods they call
        ldButtons = [
            dict(name="Refresh", method=self.controller.refresh, tip="Redraw only"),
            dict(name="Rotate Agent", method=self.controller.rotate_agent, tip="Rotate selected agent"),
            dict(name="Restart Agents", method=self.controller.restart_agents,
                 tip="Move agents back to start positions"),
            dict(name="Random", method=self.controller.reset,
                 tip="Generate a randomized scene, including regen rail + agents"),
            dict(name="Regenerate", method=self.controller.regenerate,
                 tip="Regenerate the rails using the method selected below"),
            dict(name="Load", method=self.controller.load),
            dict(name="Save", method=self.controller.save),
            dict(name="Save as image", method=self.controller.save_image)
        ]

        self.lwButtons = []
        for dButton in ldButtons:
            wButton = ipywidgets.Button(description=dButton["name"],
                                        tooltip=dButton["tip"] if "tip" in dButton else dButton["name"])
            wButton.on_click(dButton["method"])
            self.lwButtons.append(wButton)

        self.wVbox_controls = VBox([
            self.filename,
            *self.lwButtons,
            self.wTab])

        self.wMain = HBox([self.wImage, self.wVbox_controls])

    def draw_stroke(self):
        pass

    def new_env(self):
        """ Tell the view to update its graphics when a new env is created.
        """
        self.oRT = rt.RenderTool(self.editor.env, gl=self.sGL)

    def redraw(self):
        with self.output_generator:
            self.oRT.set_new_rail()
            self.model.env.agents = self.model.env.agents_static
            for a in self.model.env.agents:
                if hasattr(a, 'old_position') is False:
                    a.old_position = a.position
                if hasattr(a, 'old_direction') is False:
                    a.old_direction = a.direction

            self.oRT.render_env(agents=True,
                                show=False,
                                selected_agent=self.model.selected_agent,
                                show_observations=False)
            img = self.oRT.get_image()

            self.wImage.data = img
            self.writableData = np.copy(self.wImage.data)

            # the size should only be updated on regenerate at most
            self.yxSize = self.wImage.data.shape[:2]
            return img

    def redisplay_image(self):
        if self.writableData is not None:
            # This updates the image in the browser to be the new edited version
            self.wImage.data = self.writableData

    def drag_path_element(self, x, y):
        # Draw a black square on the in-memory copy of the image
        if x > 10 and x < self.yxSize[1] and y > 10 and y < self.yxSize[0]:
            self.writableData[(y - 2):(y + 2), (x - 2):(x + 2), :3] = 0

    def xy_to_rc(self, x, y):
        rcCell = ((array([y, x]) - self.yxBase))
        nX = np.floor((self.yxSize[0] - self.yxBase[0]) / self.model.env.height)
        nY = np.floor((self.yxSize[1] - self.yxBase[1]) / self.model.env.width)
        rcCell[0] = max(0, min(np.floor(rcCell[0] / nY), self.model.env.height - 1))
        rcCell[1] = max(0, min(np.floor(rcCell[1] / nX), self.model.env.width - 1))
        return rcCell

    def log(self, *args, **kwargs):
        if self.output_generator:
            with self.output_generator:
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
        self.q_events = deque()
        self.drawMode = "Draw"

    def set_model(self, model):
        self.model = model

    def on_click(self, wid, event):
        x = event['canvasX']
        y = event['canvasY']
        self.debug("debug:", x, y)

        rcCell = self.view.xy_to_rc(x, y)

        bShift = event["shiftKey"]
        bCtrl = event["ctrlKey"]
        bAlt = event["altKey"]
        if bCtrl and not bShift and not bAlt:
            self.model.click_agent(rcCell)
            self.lrcStroke = []
        elif bShift and bCtrl:
            self.model.add_target(rcCell)
            self.lrcStroke = []
        elif bAlt and not bShift and not bCtrl:
            self.model.clear_cell(rcCell)
            self.lrcStroke = []

        self.debug("click in cell", rcCell)
        self.model.debug_cell(rcCell)

        if self.model.selected_agent is not None:
            self.lrcStroke = []

    def set_debug(self, event):
        self.model.set_debug(event["new"])

    def set_debug_move(self, event):
        self.model.set_debug_move(event["new"])

    def set_draw_mode(self, event):
        self.set_draw_mode = event["new"]

    def set_filename(self, event):
        self.model.set_filename(event["new"])

    def on_mouse_move(self, wid, event):
        """Mouse motion event handler for drawing.
        """

        x = event['canvasX']
        y = event['canvasY']
        q_events = self.q_events

        if self.model.debug_bool and (event["buttons"] > 0 or self.model.debug_move_bool):
            self.debug("debug:", len(q_events), event)

        # If the mouse is held down, enqueue an event in our own queue
        # The intention was to avoid too many redraws.
        # Reset the lrcStroke list, if ALT, CTRL or SHIFT pressed
        if event["buttons"] > 0:
            q_events.append((time.time(), x, y))
            bShift = event["shiftKey"]
            bCtrl = event["ctrlKey"]
            bAlt = event["altKey"]
            if bShift:
                self.lrcStroke = []
                while len(q_events) > 0:
                    t, x, y = q_events.popleft()
                return
            if bCtrl:
                self.lrcStroke = []
                while len(q_events) > 0:
                    t, x, y = q_events.popleft()
                return
            if bAlt:
                self.lrcStroke = []
                while len(q_events) > 0:
                    t, x, y = q_events.popleft()
                return
        else:
            self.lrcStroke = []

        if self.model.selected_agent is not None:
            self.lrcStroke = []
            while len(q_events) > 0:
                t, x, y = q_events.popleft()
            return

        # Process the events in our queue:
        # Draw a black square to indicate a trail
        # Convert the xy position to a cell rc
        # Enqueue transitions across cells in another queue
        if len(q_events) > 0:
            t_now = time.time()
            if t_now - q_events[0][0] > 0.1:  # wait before trying to draw

                while len(q_events) > 0:
                    t, x, y = q_events.popleft()  # get events from our queue
                    self.view.drag_path_element(x, y)

                    # Translate and scale from x,y to integer row,col (note order change)
                    rcCell = self.view.xy_to_rc(x, y)
                    self.editor.drag_path_element(rcCell)

                self.view.redisplay_image()

        else:
            self.model.mod_path(not event["shiftKey"])

    def refresh(self, event):
        self.debug("refresh")
        self.view.redraw()

    def clear(self, event):
        self.model.clear()

    def reset(self, event):
        self.log("Reset - nAgents:", self.view.regen_n_agents.value)
        self.log("Reset - size:", self.model.regen_size_width)
        self.log("Reset - size:", self.model.regen_size_height)
        self.model.reset(replace_agents=self.view.replace_agents.value,
                         nAgents=self.view.regen_n_agents.value)

    def rotate_agent(self, event):
        self.log("Rotate Agent:", self.model.selected_agent)
        if self.model.selected_agent is not None:
            for agent_idx, agent in enumerate(self.model.env.agents_static):
                if agent is None:
                    continue
                if agent_idx == self.model.selected_agent:
                    agent.direction = (agent.direction + 1) % 4
                    agent.old_direction = agent.direction
        self.model.redraw()

    def restart_agents(self, event):
        self.log("Restart Agents - nAgents:", self.view.regen_n_agents.value)
        if self.model.init_agents_static is not None:
            self.model.env.agents_static = [EnvAgentStatic(d[0], d[1], d[2], moving=False) for d in
                                            self.model.init_agents_static]
            self.model.env.agents = None
            self.model.init_agents_static = None
            self.model.env.restart_agents()
            self.model.env.reset(False, False)
        self.refresh(event)

    def regenerate(self, event):
        method = self.view.regen_method.value
        n_agents = self.view.regen_n_agents.value
        self.model.regenerate(method, n_agents)

    def set_regen_width(self, event):
        self.model.set_regen_width(event["new"])

    def set_regen_height(self, event):
        self.model.set_regen_height(event["new"])

    def load(self, event):
        self.model.load()

    def save(self, event):
        self.model.save()

    def save_image(self, event):
        self.model.save_image()

    def step(self, event):
        self.model.step()

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
        self.regen_size_width = 10
        self.regen_size_height = 10

        self.lrcStroke = []
        self.iTransLast = -1
        self.gRCTrans = array([[-1, 0], [0, 1], [1, 0], [0, -1]])  # NESW in RC

        self.debug_bool = False
        self.debug_move_bool = False
        self.wid_output = None
        self.draw_mode = "Draw"
        self.env_filename = "temp.pkl"
        self.set_env(env)
        self.selected_agent = None
        self.init_agents_static = None
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

    def interpolate_path(self, rcLast, rcCell):
        if np.array_equal(rcLast, rcCell):
            return []
        rcLast = array(rcLast)
        rcCell = array(rcCell)
        rcDelta = rcCell - rcLast

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

    def drag_path_element(self, rcCell):
        """Mouse motion event handler for drawing.
        """
        lrcStroke = self.lrcStroke

        # Store the row,col location of the click, if we have entered a new cell
        if len(lrcStroke) > 0:
            rcLast = lrcStroke[-1]
            if not np.array_equal(rcLast, rcCell):  # only save at transition
                lrcInterp = self.interpolate_path(rcLast, rcCell)
                lrcStroke.extend(lrcInterp)
                self.debug("lrcStroke ", len(lrcStroke), rcCell, "interp:", lrcInterp)

        else:
            # This is the first cell in a mouse stroke
            lrcStroke.append(rcCell)
            self.debug("lrcStroke ", len(lrcStroke), rcCell)

    def mod_path(self, bAddRemove):
        # disabled functionality (no longer required)
        if bAddRemove is False:
            return
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
            if self.env.rail.get_full_transitions(*lrcStroke[0]) == 0:
                self.mod_rail_2cells(lrcStroke, bAddRemove, iCellToMod=0)

        # Add transitions for groups of 3 cells
        # hence inbound and outbound transitions for middle cell
        while len(lrcStroke) >= 3:
            self.mod_rail_3cells(lrcStroke, bAddRemove=bAddRemove)

        # If final cell empty, insert deadend:
        if len(lrcStroke) == 2:
            if self.env.rail.get_full_transitions(*lrcStroke[1]) == 0:
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
        self.env.agents_static = []

        self.redraw()

    def clear_cell(self, cell_row_col):
        self.debug_cell(cell_row_col)
        self.env.rail.grid[cell_row_col[0], cell_row_col[1]] = 0
        self.redraw()

    def reset(self, replace_agents=False, nAgents=0):
        self.regenerate("complex", nAgents=nAgents)
        self.redraw()

    def restart_agents(self):
        self.env.agents = EnvAgent.list_from_static(self.env.agents_static)
        self.redraw()

    def set_filename(self, filename):
        self.env_filename = filename

    def load(self):
        if os.path.exists(self.env_filename):
            self.log("load file: ", self.env_filename)
            self.env.load(self.env_filename)
            if not self.regen_size_height == self.env.height or not self.regen_size_width == self.env.width:
                self.regen_size_height = self.env.height
                self.regen_size_width = self.env.width
                self.regenerate(None, 0, self.env)
                self.env.load(self.env_filename)

            self.env.restart_agents()
            self.env.reset(False, False)
            self.init_agents_static = None
            self.view.oRT.update_background()
            self.fix_env()
            self.set_env(self.env)
            self.redraw()
        else:
            self.log("File does not exist:", self.env_filename, " Working directory: ", os.getcwd())

    def save(self):
        self.log("save to ", self.env_filename, " working dir: ", os.getcwd())
        temp_store = self.env.agents
        # clear agents before save , because we want the "init" position of the agent to expert
        self.env.agents = []
        self.env.save(self.env_filename)
        # reset agents current (current position)
        self.env.agents = temp_store

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
        elif method == "Random Cell":
            fnMethod = random_rail_generator(cell_type_relative_proportion=[1] * 11)
        else:
            fnMethod = complex_rail_generator(nr_start_goal=nAgents, nr_extra=20, min_dist=12, seed=int(time.time()))

        if env is None:
            self.env = RailEnv(width=self.regen_size_width,
                               height=self.regen_size_height,
                               rail_generator=fnMethod,
                               number_of_agents=nAgents,
                               obs_builder_object=TreeObsForRailEnv(max_depth=2))
        else:
            self.env = env
        self.env.reset(regen_rail=True)
        self.fix_env()
        self.set_env(self.env)
        self.view.new_env()
        self.redraw()

    def set_regen_width(self, size):
        self.regen_size_width = size

    def set_regen_height(self, size):
        self.regen_size_height = size

    def find_agent_at(self, cell_row_col):
        for agent_idx, agent in enumerate(self.env.agents_static):
            if tuple(agent.position) == tuple(cell_row_col):
                return agent_idx
        return None

    def click_agent(self, cell_row_col):
        """ The user has clicked on a cell -
            - If there is an agent, select it
                - If that agent was already selected, then deselect it
            - If there is no agent selected, and no agent in the cell, create one
            - If there is an agent selected, and no agent in the cell, move the selected agent to the cell
        """

        # Has the user clicked on an existing agent?
        agent_idx = self.find_agent_at(cell_row_col)

        if agent_idx is None:
            # No
            if self.selected_agent is None:
                # Create a new agent and select it.
                agent_static = EnvAgentStatic(position=cell_row_col, direction=0, target=cell_row_col, moving=False)
                self.selected_agent = self.env.add_agent_static(agent_static)
                self.view.oRT.update_background()
            else:
                # Move the selected agent to this cell
                agent_static = self.env.agents_static[self.selected_agent]
                agent_static.position = cell_row_col
                agent_static.old_position = cell_row_col
                self.env.agents = []
        else:
            # Yes
            # Have they clicked on the agent already selected?
            if self.selected_agent is not None and agent_idx == self.selected_agent:
                # Yes - deselect the agent
                self.selected_agent = None
            else:
                # No - select the agent
                self.selected_agent = agent_idx

        self.init_agents_static = None
        self.redraw()

    def add_target(self, rcCell):
        if self.selected_agent is not None:
            self.env.agents_static[self.selected_agent].target = rcCell
            self.init_agents_static = None
            self.view.oRT.update_background()
            self.redraw()

    def fix_env(self):
        self.env.width = self.env.rail.width
        self.env.height = self.env.rail.height

    def log(self, *args, **kwargs):
        if self.view is None:
            print(*args, **kwargs)
        else:
            self.view.log(*args, **kwargs)

    def debug(self, *args, **kwargs):
        if self.debug_bool:
            self.log(*args, **kwargs)

    def debug_cell(self, rcCell):
        binTrans = self.env.rail.get_full_transitions(*rcCell)
        sbinTrans = format(binTrans, "#018b")[2:]
        self.debug("cell ",
                   rcCell,
                   "Transitions: ",
                   binTrans,
                   sbinTrans,
                   [sbinTrans[i:(i + 4)] for i in range(0, len(sbinTrans), 4)])
