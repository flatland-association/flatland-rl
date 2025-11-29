from typing import Optional, Any, List

import numpy as np
import torch
from ray.rllib.algorithms import Algorithm
from ray.rllib.core import Columns
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils.numpy import convert_to_numpy, softmax

from flatland.core.policy import Policy
from flatland.env_generation.env_generator import env_generator
from flatland.envs.RailEnvPolicy import RailEnvPolicy
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.ml.ray.ray_multi_agent_rail_env import RayMultiAgentWrapper


def ray_multi_agent_env_wrapper(wrap: RailEnv, render_mode: Optional[str] = None) -> RayMultiAgentWrapper:
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


def ray_env_generator(render_mode: Optional[str] = None, **kwargs) -> RayMultiAgentWrapper:
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


# TODO is this action applying safe?
def ray_policy_wrapper(rl_module: RLModule) -> RailEnvPolicy:
    class _RayCheckpointPolicy(RailEnvPolicy):

        def act_many(self, handles: List[int], observations: List[Any], **kwargs):
            obss = np.stack(observations)
            # TODO why no batch dim?
            # obss = torch.from_numpy(obss).unsqueeze(0).float()
            obss = torch.from_numpy(obss).float()
            actions = rl_module.forward_inference({"obs": obss})
            if Columns.ACTIONS in actions:
                if isinstance(actions[Columns.ACTIONS], dict):
                    action_dict = {h: a[0] for h, a in actions[Columns.ACTIONS].items()}
                else:
                    # TODO why no batch dim?
                    action_dict = dict(zip(handles, convert_to_numpy(actions[Columns.ACTIONS])))
            else:
                logits = convert_to_numpy(actions[Columns.ACTION_DIST_INPUTS])

                # TODO why no batch dim?
                action_dict = {str(h): np.random.choice(len(RailEnvActions), p=softmax(l)) for h, l in enumerate(logits)}

            return action_dict

    return _RayCheckpointPolicy()


# TODO is Algorithm the appropriate abstraction here?
def ray_policy_wrapper_from_rllib_checkpoint(checkpoint_path: str, algo: Algorithm, module_id: str) -> RailEnvPolicy:
    """
    Load RLlib checkpoint into Flatland RailEnvPolicy.

    https://docs.ray.io/en/latest/rllib/checkpoints.html

    Parameters
    ----------
    checkpoint_path : str
        path to the rllib checkpoint
    algo : Algorithm
    module_id : str
        module ID to be found under <checkpoint_path>/learner_group/learner/rl_module/<module_id>

    Returns
    -------
    Policy
    """

    algo.restore(checkpoint_path)
    rl_module = algo.get_module(module_id)
    return ray_policy_wrapper(rl_module)
