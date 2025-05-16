from typing import Optional, Any, List

import numpy as np
import torch
from ray.rllib import MultiAgentEnv
from ray.rllib.core import Columns
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils.numpy import convert_to_numpy, softmax

from flatland.core.policy import Policy
from flatland.env_generation.env_generator import env_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.ml.ray.ray_multi_agent_rail_env import RayMultiAgentWrapper


def ray_multi_agent_env_wrapper(wrap: RailEnv, render_mode: Optional[str] = None) -> MultiAgentEnv:
    """
    Wrap `RailEnv` as `ray.rllib.MultiAgentEnv`. Make sure the observation builds are

    Parameters
    ----------
    wrap: RailEnv
        the rail env
    render_mode
        passed to `RayMultiAgentWrapper.step()`

    Returns
    -------

    """
    return RayMultiAgentWrapper(wrap, render_mode)


def ray_env_generator(render_mode: Optional[str] = None, **kwargs) -> MultiAgentEnv:
    """
    Create and reset `RailEnv` wrapped as `ray.rllib.MultiAgentEnv`.

    Parameters
    ----------
    render_mode
        passed to `RayMultiAgentWrapper.step()`
    kwargs
        passed to `env_generator`

    Returns
    -------

    """
    rail_env, _, _ = env_generator(**kwargs)
    # https://discuss.ray.io/t/multi-agent-where-does-the-first-structure-comes-from/7010/8
    env = ray_multi_agent_env_wrapper(wrap=rail_env, render_mode=render_mode)
    return env


def ray_policy_wrapper(rl_module: RLModule) -> Policy:
    class _RayCheckpointPolicy(Policy):

        def act_many(self, handles: List[int], observations: List[Any], **kwargs):
            obss = np.stack(observations)

            actions = rl_module.forward_inference({"obs": torch.from_numpy(obss).unsqueeze(0).float()})
            if Columns.ACTIONS in actions:
                if isinstance(actions[Columns.ACTIONS], dict):
                    action_dict = {h: a[0] for h, a in actions[Columns.ACTIONS].items()}
                else:
                    action_dict = dict(zip(handles, convert_to_numpy(actions[Columns.ACTIONS][0])))
            else:
                logits = convert_to_numpy(actions[Columns.ACTION_DIST_INPUTS])
                action_dict = {str(h): np.random.choice(len(RailEnvActions), p=softmax(l)) for h, l in enumerate(logits[0])}

            return action_dict

    return _RayCheckpointPolicy()
