from typing import Dict

from flatland.envs.rail_env import AbstractRailEnv
from flatland.envs.rail_env_action import RailEnvActions


def RailEnvHeadingInfoWrapper(env: AbstractRailEnv) -> AbstractRailEnv:
    """
    Patches `env`'s class in place so its `reset()`/`step()` info dict also carries a `heading` entry:
    for every agent handle, the waypoint (position, direction) its train is heading to - its route's
    final waypoint, `agent.waypoints[-1][0]`. This is exactly the waypoint `ShortestPathPolicy` itself
    paths towards (the last entry of its own `_shortest_paths[handle]`), just read directly off the
    agent rather than off a running policy instance, so it's available even before any policy has
    acted.

    Same override-and-delegate shape as `RailEnvStateMachineWrapper` (patches `env.__class__` to mix in
    an override, rather than wrapping the instance in a separate object) - see that wrapper's own
    docstring for why. Idempotent: wrapping an already-wrapped env is a no-op. Returns the same
    instance (not a copy), for chaining convenience.

    Parameters
    ----------
    env : AbstractRailEnv
        The env to patch in place.
    """
    base_cls = type(env)
    if not issubclass(base_cls, _HeadingInfoMixin):
        base_cls = type(f"{base_cls.__name__}WithHeadingInfo", (_HeadingInfoMixin, base_cls), {})
        env.__class__ = base_cls
    return env


class _HeadingInfoMixin:
    """ Only ever applied to an instance via `RailEnvHeadingInfoWrapper` - never instantiated/subclassed directly. """

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        return obs, self._with_heading_info(info)

    def step(self, action_dict: Dict[int, RailEnvActions]):
        obs, rewards, dones, info = super().step(action_dict)
        return obs, rewards, dones, self._with_heading_info(info)

    def _with_heading_info(self, info: Dict) -> Dict:
        info['heading'] = {agent.handle: agent.waypoints[-1][0] for agent in self.agents}
        return info
