"""
Flatland wrapper for ray RLlib multi-agent environment (https://docs.ray.io/en/latest/rllib/multi-agent-envs.html).
"""
import copy
from typing import Tuple, Set, Optional

import gymnasium as gym
from gymnasium import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.utils import override
from ray.rllib.utils.typing import MultiAgentDict, AgentID

from flatland.core.env import Environment
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.utils.rendertools import RenderTool


class RayMultiAgentWrapper(MultiAgentEnv):
    """
    Agents are identified by AgentIDs (string).
    See `ray.rllib.MultiAgentEnv`.
    Otherwise, there are flapping errors in runs: "IndexError: list index out of range" -> "ray.rllib.utils.error.MultiAgentEnvError: Agent 0 already had its `SingleAgentEpisode.is_done` set to True, but still received data in a following step! obs=None act=None rew=0 info=None extra_model_outputs=None."
    """

    def __init__(self, wrap: RailEnv, render_mode: Optional[str] = None):
        assert hasattr(wrap.obs_builder, "get_observation_space"), f"{type(wrap.obs_builder)} is not gym-compatible, missing get_observation_space"
        self.wrap: RailEnv = wrap
        self.render_mode = render_mode
        self.env_renderer = None
        if render_mode is not None:
            self.env_renderer = RenderTool(wrap)

        self.action_space: gym.spaces.Dict = spaces.Dict({
            str(i): gym.spaces.Discrete(wrap.action_space[0])
            for i in range(self.wrap.number_of_agents)
        })

        self.observation_space: gym.spaces.Dict = gym.spaces.Dict({
            str(handle): self.wrap.obs_builder.get_observation_space(handle)
            for handle in self.wrap.get_agent_handles()
        })

        # Provide full (preferred format) observation- and action-spaces as Dicts
        # mapping agent IDs to the individual agents' spaces.
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True

        super().__init__()

        self.agents = [str(i) for i in self.wrap.get_agent_handles()]

    @override(Environment)
    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:

        prev_dones = copy.deepcopy(self.wrap.dones)

        action_dict = {k: RailEnvActions(v) for k, v in action_dict.items()}
        obs, rewards, terminateds, infos = self.wrap.step(action_dict=action_dict)
        infos = {
            str(i):
                {
                    'action_required': infos['action_required'][i],
                    'malfunction': infos['malfunction'][i],
                    'speed': infos['speed'][i],
                    'state': infos['state'][i]
                } for i in self.wrap.get_agent_handles()
        }
        rewards = {str(i): rewards[i] for i in self.wrap.get_agent_handles()}

        # convert np.ndarray to MultiAgentDict
        obs = {str(i): obs[i] for i in self.wrap.get_agent_handles()}

        # report obs/done/info only once per agent per episode,
        # see https://github.com/ray-project/ray/issues/10761
        terminateds = copy.deepcopy(terminateds)
        terminateds = {str(i): terminateds[i] for i in self.wrap.get_agent_handles()}
        terminateds["__all__"] = all(terminateds.values())
        for i in self.wrap.get_agent_handles():
            if prev_dones[i] is True:
                del obs[str(i)]
                del terminateds[str(i)]
                del infos[str(i)]
        truncateds = {"__all__": False}

        if self.render_mode is not None:
            # We render the initial step and show the observed cells as colored boxes
            self.env_renderer.render_env(show=True, frames=True, show_observations=True, show_predictions=False)
        return obs, rewards, terminateds, truncateds, infos

    @override(Environment)
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        if options is None:
            options = {}
        obs, infos = self.wrap.reset(random_seed=seed, **options)

        # convert np.ndarray to MultiAgentDict
        obs = {str(i): obs[i] for i in self.wrap.get_agent_handles()}

        infos = {
            str(i):
                {
                    'action_required': infos['action_required'][i],
                    'malfunction': infos['malfunction'][i],
                    'speed': infos['speed'][i],
                    'state': infos['state'][i]
                } for i in self.wrap.get_agent_handles()
        }
        return obs, infos

    @override(MultiAgentEnv)
    def get_agent_ids(self) -> Set[AgentID]:
        return set({str(i) for i in self.wrap.get_agent_handles()})
