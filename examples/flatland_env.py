import os
import math
import numpy as np
import gym
from gym.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from gym.utils import EzPickle
from pettingzoo.utils.conversions import parallel_wrapper_fn
from flatland.envs.rail_env import RailEnv
from mava.wrappers.flatland import infer_observation_space, normalize_observation
from functools import partial
from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv


def env(**kwargs):
    env = raw_env(**kwargs)
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, gym.Env):

    metadata = {'render.modes': ['human', "rgb_array"], 'name': "flatland_pettingzoo",
            'video.frames_per_second': 10,
            'semantics.autoreset': False }

    def __init__(self, environment = False, preprocessor = False, agent_info = False, use_renderer=False, *args, **kwargs):
        # EzPickle.__init__(self, *args, **kwargs)
        self._environment = environment
        self.use_renderer = use_renderer
        self.renderer = None
        if self.use_renderer:
            self.initialize_renderer()

        n_agents = self.num_agents
        self._agents = [get_agent_keys(i) for i in range(n_agents)]
        self._possible_agents = self.agents[:]
        self._reset_next_step = True

        # self.agent_name_mapping = dict(zip(self.agents, list(range(self.n_pistons))))
        self._agent_selector = agent_selector(self.agents)
        # self.agent_selection = self.agents
        # self.observation_spaces = dict(
        #     zip(self.agents, [gym.spaces.Box(low=0, high=1,shape = (1,1) , dtype=np.float32)] * n_agents))
        self.num_actions = 5

        self.action_spaces = {
            agent: gym.spaces.Discrete(self.num_actions) for agent in self.possible_agents
        }              
        # self.state_space = gym.spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8)
        # self.closed = False
        self.seed()
        # preprocessor must be for observation builders other than global obs
        # treeobs builders would use the default preprocessor if none is
        # supplied
        self.preprocessor = self._obtain_preprocessor(preprocessor)

        self._include_agent_info = agent_info

        # observation space:
        # flatland defines no observation space for an agent. Here we try
        # to define the observation space. All agents are identical and would
        # have the same observation space.
        # Infer observation space based on returned observation
        obs, _ = self._environment.reset(regenerate_rail = False, regenerate_schedule = False)
        obs = self.preprocessor(obs)
        self.observation_spaces = {
            i: infer_observation_space(ob) for i, ob in obs.items()
        }

    
    @property
    def environment(self) -> RailEnv:
        """Returns the wrapped environment."""
        return self._environment

    @property
    def dones(self):
        dones = self._environment.dones
        # remove_all = dones.pop("__all__", None)
        return {get_agent_keys(key): value for key, value in dones.items()}    
    
    @property
    def obs_builder(self):    
        return self._environment.obs_builder    

    @property
    def width(self):    
        return self._environment.width  

    @property
    def height(self):    
        return self._environment.height  

    @property
    def agents_data(self):
        """Rail Env Agents data."""
        return self._environment.agents

    @property
    def num_agents(self) -> int:
        """Returns the number of trains/agents in the flatland environment"""
        return int(self._environment.number_of_agents)

    # def __getattr__(self, name):
    #     """Expose any other attributes of the underlying environment."""
    #     return getattr(self._environment, name)
   
    @property
    def agents(self):
        return self._agents

    @property
    def possible_agents(self):
        return self._possible_agents

    def env_done(self):
        return self._environment.dones["__all__"] or not self.agents
    
    def observe(self,agent):
        return self.obs.get(agent)
    
    def seed(self, seed: int = None) -> None:
        self._environment._seed(seed)

    def state(self):
        '''
        Returns an observation of the global environment
        '''
        return None

   
    def reset(self, *args, **kwargs):
        self._reset_next_step = False
        self._agents = self.possible_agents[:]
        if self.use_renderer:
            if self.renderer: #TODO: Errors with RLLib with renderer as None.
                self.renderer.reset()
        obs, info = self._environment.reset(*args, **kwargs)
        observations = self._collate_obs_and_info(obs, info)
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.action_dict = {i:0 for i in self.possible_agents}

        return observations

    def step(self, action):

        if self.env_done():
            self._agents = []
            return self.last()
        
        agent = self.agent_selection
        self.action_dict[get_agent_handle(agent)] = action
        if self._reset_next_step:
            return self.reset()

        if self.dones[agent]:
            self.agents.remove(agent)
            if not self.env_done():
                self.agent_selection = self._agent_selector.next()
            return self.last()

        if self._agent_selector.is_last():
            observations, rewards, dones, infos = self._environment.step(self.action_dict)
            self.rewards = {get_agent_keys(key): value for key, value in rewards.items()}
            if observations:
                observations = self._collate_obs_and_info(observations, infos)
           
            # self._agents = [agent
            #                 for agent in self.agents
            #                 if not self._environment.dones[get_agent_handle(agent)]
            #                 ]
    
        else:
            self._clear_rewards()
        
        # self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
        
        self.agent_selection = self._agent_selector.next()

        return self.last()

        # if self._agent_selector.is_last():
        #     self._agent_selector.reinit(self.agents)

    # collate agent info and observation into a tuple, making the agents obervation to
    # be a tuple of the observation from the env and the agent info
    def _collate_obs_and_info(self, observes, info):
        observations = {}
        infos = {}
        observes = self.preprocessor(observes)
        for agent, obs in observes.items():
            all_infos = {k: info[k][get_agent_handle(agent)] for k in info.keys()}
            agent_info = np.array(
                list(all_infos.values()), dtype=np.float32
            )
            infos[agent] = all_infos
            obs = (obs, agent_info) if self._include_agent_info else obs
            observations[agent] = obs

        self.infos = infos
        self.obs = observations
        return observations   


    def render(self, mode='human'):
        """
        This methods provides the option to render the
        environment's behavior to a window which should be
        readable to the human eye if mode is set to 'human'.
        """
        if not self.use_renderer:
            return

        if not self.renderer:
            self.initialize_renderer(mode=mode)

        return self.update_renderer(mode=mode)

    def initialize_renderer(self, mode="human"):
        # Initiate the renderer
        from flatland.utils.rendertools import RenderTool, AgentRenderVariant
        self.renderer = RenderTool(self.environment, gl="PGL",  # gl="TKPILSVG",
                                       agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
                                       show_debug=False,
                                       screen_height=600,  # Adjust these parameters to fit your resolution
                                       screen_width=800)  # Adjust these parameters to fit your resolution
        self.renderer.show = False

    def update_renderer(self, mode='human'):
        image = self.renderer.render_env(show=False, show_observations=False, show_predictions=False,
                                             return_image=True)
        return image[:,:,:3]

    def set_renderer(self, renderer):
        self.use_renderer = renderer
        if self.use_renderer:
            self.initialize_renderer(mode=self.use_renderer)

    def close(self):
        # self._environment.close()
        if self.renderer:
            try:
                if self.renderer.show:
                    self.renderer.close_window()
            except Exception as e:
                print("Could Not close window due to:",e)
            self.renderer = None

    def _obtain_preprocessor(
        self, preprocessor):
        """Obtains the actual preprocessor to be used based on the supplied
        preprocessor and the env's obs_builder object"""
        if not isinstance(self.obs_builder, GlobalObsForRailEnv):
            _preprocessor = preprocessor if preprocessor else lambda x: x
            if isinstance(self.obs_builder, TreeObsForRailEnv):
                _preprocessor = (
                    partial(
                        normalize_observation, tree_depth=self.obs_builder.max_depth
                    )
                    if not preprocessor
                    else preprocessor
                )
            assert _preprocessor is not None
        else:
            def _preprocessor(x):
                    return x

        def returned_preprocessor(obs):
            temp_obs = {}
            for agent_id, ob in obs.items():
                temp_obs[get_agent_keys(agent_id)] = _preprocessor(ob)
            return temp_obs

        return returned_preprocessor

# Utility functions   
def convert_np_type(dtype, value):
    return np.dtype(dtype).type(value) 

def get_agent_handle(id):
    """Obtain an agents handle given its id"""
    return int(id)

def get_agent_keys(id):
    """Obtain an agents handle given its id"""
    return str(id)