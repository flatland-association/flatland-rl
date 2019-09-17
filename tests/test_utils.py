"""Test Utils."""
from typing import List

from attr import attrs, attrib

from flatland.envs.rail_env import RailEnvActions


@attrs
class Replay(object):
    position = attrib()
    direction = attrib()
    action = attrib(type=RailEnvActions)
    malfunction = attrib(default=0, type=int)


@attrs
class ReplayConfig(object):
    replay = attrib(type=List[Replay])
    target = attrib()
    speed = attrib(type=float)
