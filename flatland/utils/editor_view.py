

import os
import time
from collections import deque

from typing import Optional

import ipywidgets
#import jpy_canvas
from ipycanvas import Canvas
import ipyevents as ipe

import numpy as np
from ipywidgets import IntSlider, VBox, HBox, Checkbox, Output, Text, RadioButtons, Tab
from numpy import array

import flatland.utils.rendertools as rt
from flatland.core.grid.grid4_utils import mirror
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator, empty_rail_generator

from flatland.utils.editor_interfaces import AbstractController, AbstractModel, AbstractView



class View(AbstractView):
    """ The Jupyter Editor View - creates and holds the widgets comprising the Editor.
    """

    def __init__(self, editor:AbstractModel, sGL="MPL", screen_width=800, screen_height=800):
        self.editor = self.model = editor
        self.sGL = sGL
        self.xyScreen = (screen_width, screen_height)
        self.controller: Optional[AbstractController] = None


    def display(self):
        self.wOutput.clear_output()
        return self.wMain

    def clear_output(self, oDummy):
        self.log("clear output", oDummy)
        self.wOutput.clear_output()

    def init_canvas(self):
        # update the rendertool with the env
        self.new_env()
        self.oRT.render_env(show=False, show_observations=False, show_predictions=False, show_rowcols=True)
        img = self.oRT.get_image()

        # NCW (new canvas widget)
        #self.wImage = jpy_canvas.Canvas(img)
        self.wImage = Canvas(width=img.shape[1], height=img.shape[0])

        # NCW 
        #self.yxSize = self.wImage.data.shape[:2]
        self.yxSize = img.shape[:2]

        # NCW - not sure if we need a "writableData" any more
        #self.writableData = np.copy(self.wImage.data)  # writable copy of image - wid_img.data is somehow readonly
        self.wImage.put_image_data(img)
        self.writableData = np.copy(img)
        

        # Register Canvas event handler

        # NCW: 
        #self.wImage.register_move(self.controller.on_mouse_move)
        #self.wImage.register_click(self.controller.on_click)
        
        oEvent = ipe.Event(source=self.wImage, watched_events=['mousemove', 'click'])

        self.yxBase = self.oRT.gl.yxBase
        self.nPixCell = self.oRT.gl.nPixCell

        oEvent.on_dom_event(self.controller.handle_event)

    def init_widgets(self):
        # Debug checkbox - enable logging in the Output widget
        self.wDebug = ipywidgets.Checkbox(description="Debug")
        self.wDebug.value = False # Change to True to make debugging easier - or click the checkbox
        self.wDebug.observe(self.controller.set_debug, names="value")

        # Separate checkbox for mouse move events - they are very verbose
        self.wDebug_move = Checkbox(description="Debug mouse move")
        self.wDebug_move.observe(self.controller.set_debug_move, names="value")

        # This is like a cell widget where loggin goes
        self.wOutput = Output()

        self.wClearOutput = ipywidgets.Button(description="Clear Output")
        self.wClearOutput.on_click(self.clear_output)

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
        self.regen_method = RadioButtons(description="Regen\nMethod", options=["Empty", "Sparse"])

        self.replace_agents = Checkbox(value=True, description="Replace Agents")

        self.wTab = Tab()
        
        #for i, title in enumerate(tab_contents):
        #    self.wTab.set_title(i, title)
        
        self.wTab.children = [
            VBox([self.regen_width, self.regen_height, self.regen_n_agents, self.regen_method]),
            VBox([self.wDebug, self.wDebug_move, self.replace_agents, self.wClearOutput])
        ]
        self.wTab.set_title(0, "Regen")
        self.wTab.set_title(1, "Debug")

        # abbreviated description of buttons and the methods they call
        ldButtons = [
            dict(name="Refresh", method=self.controller.refresh, tip="Redraw only"),
            dict(name="Rotate Agent", method=self.controller.rotate_agent, tip="Rotate selected agent"),
            dict(name="Restart Agents", method=self.controller.reset_agents,
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
        self.oRT = rt.RenderTool(self.editor.env, gl=self.sGL, show_debug=True,
            screen_height=self.xyScreen[1], screen_width=self.xyScreen[0])

    def redraw(self):
        """ Redraw the environment and agents.
            This will erase the current image and draw a new one.
            See also redisplay_image()
        """
        with self.wOutput:
            self.oRT.set_new_rail()
            self.model.env.reset_agents()
            for a in self.model.env.agents:
                if hasattr(a, 'old_position') is False:
                    a.old_position = a.position
                if hasattr(a, 'old_direction') is False:
                    a.old_direction = a.direction

            self.oRT.render_env(show_agents=True,
                                show_inactive_agents=True,
                                show=False,
                                selected_agent=self.model.selected_agent,
                                show_observations=False,
                                show_rowcols=True,
                                )
            img = self.oRT.get_image()

            #self.wImage.data = img
            #self.writableData = np.copy(self.wImage.data)
            self.writableData = np.copy(img)
            self.wImage.put_image_data(img)
            

            # the size should only be updated on regenerate at most
            #self.yxSize = self.wImage.data.shape[:2]
            return img

    def redisplay_image(self):
        """ Redisplay the writable image in the Canvas.
            Called during image editing, when minor changes are made directly to the image,
            between redraws.
        """
        #if self.writableData is not None:
            # This updates the image in the browser to be the new edited version
        #    self.wImage.data = self.writableData
        self.wImage.put_image_data(self.writableData)

    def drag_path_element(self, x, y):
        """ Add another x,y point to a drag gesture.
            Just draw a black square on the in-memory copy of the image.
            With ipyCanvas, we need to adjust the Event x,y coordinates to image x,y.
        """
        #
        yxRectEv = array(self.controller.getBoundingRectYX())
        yxPointEv = array([y, x])
        yxPointImg = np.clip(yxPointEv * self.yxSize / yxRectEv, 0, self.yxSize).astype(int)

        #if x > 10 and x < self.yxSize[1] and y > 10 and y < self.yxSize[0]:
        #self.writableData[(y - 2):(y + 2), (x - 2):(x + 2), :3] = 0
        self.writableData[yxPointImg[0]-2:yxPointImg[0]+2,  # y
                          yxPointImg[1]-2:yxPointImg[1]+2,  # x
                          :3] = 0                           # color
        self.log("drag_path_element: ", x, y)
        #else:
        #    self.log("Drag out of bounds: ", x, y)

    def xy_to_rc(self, x, y):
        """ Convert from x,y coordinates to row,col coordinates.
            This is used to convert mouse clicks to row,col coordinates.
        """
        yxRect = array(self.controller.getBoundingRectYX())
        yxPoint = array(((array([y, x]) - self.yxBase)))
        rcRect = array([self.model.env.height, self.model.env.width])
        
        # Scale factors for converting from pixels (y,x) to cells (r,c)
        #nY = np.floor((self.yxSize[0] - self.yxBase[0]) / self.model.env.height)
        #nX = np.floor((self.yxSize[1] - self.yxBase[1]) / self.model.env.width)

        rc_cell = np.floor(np.clip(yxPoint / yxRect * rcRect, [0,0], rcRect - 1)).astype(int)
        self.log("xy_to_rc: ", x, y, " -> ", rc_cell, type(rc_cell))
        # Row from y
        #rc_cell[0] = max(0, min(np.floor(yxPoint[0] / nY), self.model.env.height - 1))
        # Column from x
        #=rc_cell[1] = max(0, min(np.floor(yxPoint[1] / nX), self.model.env.width - 1))

        # Using numpy arrays for coords not currently supported downstream in the env, observations, etc
        return tuple(rc_cell)

    def log(self, *args, **kwargs):
        if self.wOutput:
            with self.wOutput:
                print(*args, **kwargs)
        else:
            print(*args, **kwargs)

