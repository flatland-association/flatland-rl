

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


class Controller(AbstractController):
    """
    Controller to handle incoming events from the ipywidgets
    Updates the editor/model.
    Calls the View directly for things which do not directly effect the model
    (this means the mouse drag path before it is interpreted as transitions)
    """

    def __init__(self, model:"AbstractModel", view:AbstractView):
        self.editor = self.model = model
        self.view = view
        self.q_events = deque()
        self.drawMode = "Draw"
        self.bMouseDown = False

        self.boundingRectWidth = None
        self.boundingRectHeight = None

    def set_model(self, model):
        self.model = model

    def _getCoords(self, event:dict):
        #x = event['canvasX']
        #y = event['canvasY']
        x = event['relativeX']
        y = event['relativeY']
        #self.debug("debug:", x, y)
        return x, y

    def getModKeys(self, event:dict):
        bShift = event["shiftKey"]
        bCtrl = event["ctrlKey"]
        bAlt = event["altKey"]
        return bShift, bCtrl, bAlt

    #def handle_event(self, event:dict):
    def handle_event(self, event: ipe.Event):
        if "boundingRectWidth" in event:
            self.boundingRectWidth = event["boundingRectWidth"]

        if "boundingRectHeight" in event:
            self.boundingRectHeight = event["boundingRectHeight"]

        if event['type'] == 'mousemove':
            nButtons = int(event["buttons"])
            if nButtons > 0:
                self.bMouseDown = True
                self.on_mouse_move(event)
            else:  # nButtons == 0, ie mouse is now up 
                if self.bMouseDown:  # mouse was down, now up
                    self.bMouseDown = False
                    self.log("testing 123")
                    self.log("mouse up event: ", event)
                    self.log("lrcStroke: ", self.model.lrcStroke)
                    self.log("testing 456")
                    self.on_mouse_move(event) # process a move event with no buttons pressed
        elif event['type'] == 'click':
            nStroke = self.model.get_len_stroke() 
            if nStroke == 0:
                self.on_click(event)
            else:
                self.log("ignoring click, stroke len:", nStroke)

            self.debug(event)
            #self.on_click(event)


    def on_click(self, event):
        self.debug("on_click: ", event)
        x, y = self._getCoords(event)
        rc_cell = self.view.xy_to_rc(x, y)

        bShift, bCtrl, bAlt = self.getModKeys(event)
        if bCtrl and not bShift and not bAlt:   # only ctrl -> click agent
            self.model.click_agent(rc_cell)
            self.model.clear_stroke()
        elif bShift and bCtrl:                  # ctrl+shift -> add_target
            self.model.add_target(rc_cell)
            self.model.clear_stroke()
        elif bAlt and not bShift and not bCtrl: # only alt -> clear cell
            self.model.clear_cell(rc_cell)
            self.model.clear_stroke()

        self.debug("click in cell", rc_cell)
        self.model.debug_cell(rc_cell)

        if self.model.selected_agent is not None:
            self.model.clear_stroke()

    def set_debug(self, event):
        self.model.set_debug(event["new"])

    def set_debug_move(self, event):
        self.model.set_debug_move(event["new"])

    def set_draw_mode(self, event):
        self.set_draw_mode = event["new"]

    def set_filename(self, event):
        self.model.set_filename(event["new"])

    def on_mouse_move(self, event: dict):
        """ Mouse motion event handler for drawing.
        """
        x, y = self._getCoords(event)
        q_events = self.q_events

        # only log mousedown drag events, unless debug_move is set
        nButtons = int(event["buttons"])
        #if self.model.debug_bool and (nButtons > 0 or self.model.debug_move_bool):
        # in fact we already filter so we only get drags and mouse up.
        self.debug_event(event)

        # If the mouse is held down, enqueue an event in our own queue
        # The intention was to avoid too many redraws.

        if nButtons > 0:
            q_events.append((time.time(), x, y))
            bShift, bCtrl, bAlt = self.getModKeys(event)
            
            # Reset the stroke, if ALT, CTRL or SHIFT pressed
            if bShift or bCtrl or bAlt:
                self.model.clear_stroke()
                while len(q_events) > 0:
                    t, x, y = q_events.popleft()
                return
            
        # NCW: this can't be right.  If the mouse is not held down, treat it as a mouseup, and draw the stroke.
        #else:
        #    self.model.clear_stroke()

        # JW: I think this clause causes all editing to fail once an agent is selected.
        # I also can't see why it's necessary.  So I've if-falsed it out.
        if False:
            if self.model.selected_agent is not None:
                self.model.clear_stroke()
                while len(q_events) > 0:
                    t, x, y = q_events.popleft()
                return

        # Process the low-level events in our queue:
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
                    rc_cell = self.view.xy_to_rc(x, y)
                    self.editor.drag_path_element(rc_cell)

                self.view.redisplay_image()
                #self.view.redraw()


        # if mouse up, process the stroke
        if nButtons == 0:
            self.model.mod_path(not event["shiftKey"])

    def refresh(self, event):
        self.debug("refresh")
        self.view.redraw()

    def clear(self, event):
        self.model.clear()

    def clear_stroke(self, msg:str=""):
        self.debug("controller clear_stroke: ", msg)
        self.model.clear_stroke()

    def reset(self, event):
        self.log("Reset - nAgents:", self.view.regen_n_agents.value)
        self.log("Reset - size:", self.model.regen_size_width)
        self.log("Reset - size:", self.model.regen_size_height)
        self.model.reset(regenerate_schedule=self.view.replace_agents.value,
                         nAgents=self.view.regen_n_agents.value)

    def rotate_agent(self, event):
        self.log("Rotate Agent:", self.model.selected_agent)
        if self.model.selected_agent is not None:
            for agent_idx, agent in enumerate(self.model.env.agents):
                if agent is None:
                    continue
                if agent_idx == self.model.selected_agent:
                    agent.initial_direction = (agent.initial_direction + 1) % 4
                    agent.direction = agent.initial_direction
                    agent.old_direction = agent.direction
        self.model.redraw()

    def reset_agents(self, event):
        self.log("Restart Agents - nAgents:", self.view.regen_n_agents.value)
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

    def debug_event(self, event):
        self.model.debug_event(event)

    def getBoundingRectYX(self):
        return self.boundingRectHeight, self.boundingRectWidth