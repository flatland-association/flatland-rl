"""Test Utils."""
from typing import List, Tuple, Optional

from attr import attrs, attrib

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.rail_env import RailEnvActions


@attrs
class Replay(object):
    position = attrib(type=Tuple[int, int])
    direction = attrib(type=Grid4TransitionsEnum)
    action = attrib(type=RailEnvActions)
    malfunction = attrib(default=0, type=int)
    penalty = attrib(default=None, type=Optional[float])


@attrs
class ReplayConfig(object):
    replay = attrib(type=List[Replay])
    target = attrib(type=Tuple[int, int])
    speed = attrib(type=float)
