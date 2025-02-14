from typing import Optional

from ray.rllib import MultiAgentEnv

from flatland.env_generation.env_generator import env_generator
from flatland.envs.rail_env import RailEnv
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
    rail_env = env_generator(**kwargs)
    # install agents!
    rail_env.reset()
    # https://discuss.ray.io/t/multi-agent-where-does-the-first-structure-comes-from/7010/8
    env = ray_multi_agent_env_wrapper(wrap=rail_env, render_mode=render_mode)
    return env
