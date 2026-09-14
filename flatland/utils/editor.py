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


from flatland.utils.editor_model import EditorModel
from flatland.utils.editor_view import View
from flatland.utils.editor_controller import Controller
from flatland.utils.editor_interfaces import AbstractView, AbstractController, AbstractModel

class EditorMVC:
    """ EditorMVC - a class to encompass and assemble the Jupyter Editor Model-View-Controller.
    """

    def __init__(self, env=None, sGL="PIL", env_filename="temp.pkl"):
        """ Create an Editor MVC assembly around a railenv, or create one if None.
        """
        if env is None:
            nAgents = 3
            n_cities = 2
            max_rails_between_cities = 2
            max_rails_in_city = 4
            seed = 0
            env = RailEnv(
                width=30,
                height=20,
                rail_generator=sparse_rail_generator(
                    max_num_cities=n_cities,
                    seed=seed,
                    grid_mode=True,
                    max_rails_between_cities=max_rails_between_cities,
                    max_rail_pairs_in_city=max_rails_in_city
                ),
                line_generator=sparse_line_generator(),
                number_of_agents=nAgents,
                obs_builder_object=TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv())
            )

        env.reset()

        self.editor = EditorModel(env, env_filename=env_filename)
        self.editor.view = self.view = View(self.editor, sGL=sGL)
        self.view.controller = self.editor.controller = self.controller = Controller(self.editor, self.view)
        self.view.init_canvas()
        self.view.init_widgets()  # has to be done after controller


