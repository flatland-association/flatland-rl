from typing import Optional

from flatland.env_generation.env_creator import env_creator
from flatland.envs.rail_env import RailEnv
from flatland.ml.ray.ray_multi_agent_rail_env import RayMultiAgentWrapper


def ray_multi_agent_env_wrapper(wrap: RailEnv, render_mode: Optional[str] = None) -> RayMultiAgentWrapper:
    return RayMultiAgentWrapper(wrap, render_mode)


def ray_env_creator(render_mode: Optional[str] = None, **kwargs) -> RayMultiAgentWrapper:
    rail_env = env_creator(**kwargs)

    # TODO wrapper for obs conversion?

    # install agents!
    rail_env.reset()
    # https://discuss.ray.io/t/multi-agent-where-does-the-first-structure-comes-from/7010/8
    env = ray_multi_agent_env_wrapper(wrap=rail_env, render_mode=render_mode)
    return env
