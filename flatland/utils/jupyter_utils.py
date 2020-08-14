

import PIL
from IPython import display
from ipycanvas import canvas
import time

from flatland.envs import malfunction_generators as malgen
from flatland.envs.agent_utils import EnvAgent
#from flatland.envs import sparse_rail_gen as spgen
from flatland.envs import rail_generators as rail_gen
from flatland.envs import agent_chains as ac
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.persistence import RailEnvPersister
from flatland.utils.rendertools import RenderTool
from flatland.utils import env_edit_utils as eeu


class Behaviour():
    def __init__(self, env):
        self.env = env

    def getActions(self):
        return {}
    

class AlwaysForward():
    pass    


class EnvCanvas():

    def __init__(self, env):
        self.env = env
        self.oRT = RenderTool(env, show_debug=True)
        self.render()
        self.oCan = canvas.Canvas(size=(600,300))
        self.oCan.put_image_data(self.oRT.get_image())

    def render(self):
        self.oRT.render_env(show_rowcols=True,  show_inactive_agents=True, show_observations=False)

    def show(self):
        self.render()
        self.oCan.put_image_data(self.oRT.get_image())
        display.display(self.oCan)


