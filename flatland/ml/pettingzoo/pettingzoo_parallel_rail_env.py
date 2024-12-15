from typing import Optional

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import AgentID, ObsType, ActionType

from flatland.envs.rail_env import RailEnv


class PettingZooParallelEnvWrapper(ParallelEnv):
    metadata = {'render.modes': ['human', "rgb_array"], 'name': "flatland_pettingzoo",
                'video.frames_per_second': 10,
                'semantics.autoreset': False}

    def __init__(self, wrap: RailEnv, render_mode: Optional[str] = None):
        self.wrap = wrap
        self.agents: list[AgentID] = self.wrap.get_agent_handles()
        self.possible_agents: list[AgentID] = self.wrap.get_agent_handles()
        self.observation_spaces: dict[AgentID, gym.spaces.Space] = {
            # TODO bad smell: must be gymobservationbuilder
            handle: self.wrap.obs_builder.get_observation_space(handle)
            for handle in self.wrap.get_agent_handles()
        }
        self.action_spaces: dict[AgentID, gym.spaces.Space] = {
            i: gym.spaces.Discrete(5)
            for i in range(self.wrap.number_of_agents)
        }
        self.render_mode = render_mode

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        """Resets the environment.

        And returns a dictionary of observations (keyed by the agent name)
        """
        if options is None:
            options = {}
        return self.wrap.reset(random_seed=seed, **options)

    def step(
        self, actions: dict[AgentID, ActionType]
    ):
        """Receives a dictionary of actions keyed by the agent name.

        Returns the observation dictionary, reward dictionary, terminated dictionary, truncated dictionary
        and info dictionary, where each dictionary is keyed by the agent.
        """
        observations, rewards, terminations, infos = self.wrap.step(action_dict=actions)
        truncations = {"__all__": False}
        infos = {
            i:
                {
                    'action_required': infos['action_required'][i],
                    'malfunction': infos['malfunction'][i],
                    'speed': infos['speed'][i],
                    'state': infos['state'][i]
                } for i in self.wrap.get_agent_handles()
        }
        return observations, rewards, terminations, truncations, infos

    def render(self) -> None | np.ndarray | str | list:
        """Displays a rendered frame from the environment, if supported.

        Alternate render modes in the default environments are `'rgb_array'`
        which returns a numpy array and is supported by all environments outside
        of classic, and `'ansi'` which returns the strings printed
        (specific to classic environments).
        """
        return self.wrap.render()

    def close(self):
        """Closes the rendering window."""
        pass

    def state(self) -> np.ndarray:
        """Returns the state.

        State returns a global view of the environment appropriate for
        centralized training decentralized execution methods like QMIX
        """
        raise NotImplementedError(
            "state() method has not been implemented in the environment {}.".format(
                self.metadata.get("name", self.__class__.__name__)
            )
        )

    def observation_space(self, agent: AgentID) -> gym.spaces.Space:
        """Takes in agent and returns the observation space for that agent.

        MUST return the same value for the same agent name
        """
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID) -> gym.spaces.Space:
        """Takes in agent and returns the action space for that agent.

        MUST return the same value for the same agent name
        """
        return self.action_spaces[agent]
