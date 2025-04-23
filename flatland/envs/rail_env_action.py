from enum import IntEnum
from functools import lru_cache
from typing import NamedTuple

from flatland.core.grid.grid4 import Grid4TransitionsEnum


@lru_cache()
def _is_moving_action(value):
    return value in [RailEnvActions.MOVE_RIGHT, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_FORWARD]


class RailEnvActions(IntEnum):
    DO_NOTHING = 0  # implies change of direction in a dead-end!
    MOVE_LEFT = 1
    MOVE_FORWARD = 2
    MOVE_RIGHT = 3
    STOP_MOVING = 4

    @staticmethod
    def to_char(a: int):
        return {
            0: 'B',
            1: 'L',
            2: 'F',
            3: 'R',
            4: 'S',
        }[a]

    @classmethod
    @lru_cache()
    def is_action_valid(cls, action):
        return action in cls._value2member_map_

    def is_moving_action(self):
        return _is_moving_action(self.value)


RailEnvGridPos = NamedTuple('RailEnvGridPos', [('r', int), ('c', int)])
RailEnvNextAction = NamedTuple('RailEnvNextAction', [('action', RailEnvActions), ('next_position', RailEnvGridPos),
                                                     ('next_direction', Grid4TransitionsEnum)])
