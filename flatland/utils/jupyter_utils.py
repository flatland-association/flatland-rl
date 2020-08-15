

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
        self.nAg = len(env.agents)

    def getActions(self):
        return {}
    
class AlwaysForward(Behaviour):
    def getActions(self):
        return { i:RailEnvActions.MOVE_FORWARD for i in range(self.nAg) }


class EnvCanvas():

    def __init__(self, env, behaviour:Behaviour=None):
        self.env = env
        self.iStep = 0
        if behaviour is None:
            behaviour = AlwaysForward(env)
        self.behaviour = behaviour
        self.oRT = RenderTool(env, show_debug=True)

        self.oCan = canvas.Canvas(size=(600,300))
        self.render()

    def render(self):
        self.oRT.render_env(show_rowcols=True,  show_inactive_agents=True, show_observations=False)
        self.oCan.put_image_data(self.oRT.get_image())

    def step(self):
        dAction = self.behaviour.getActions()
        self.env.step(dAction)

    def show(self):
        self.render()
        display.display(self.oCan)


