from functools import lru_cache
from typing import NamedTuple, Union

import fastenum
import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum


class RailEnvActions(fastenum.Enum):
    DO_NOTHING = 0  # implies change of direction in a dead-end!
    MOVE_LEFT = 1
    MOVE_FORWARD = 2
    MOVE_RIGHT = 3
    STOP_MOVING = 4

    @staticmethod
    @lru_cache
    def from_value(value: Union["RailEnvActions", int, str]) -> "RailEnvActions":
        if isinstance(value, RailEnvActions):
            return value
        if isinstance(value, str):
            value = int(value)
        return {
            0: RailEnvActions.DO_NOTHING,
            1: RailEnvActions.MOVE_LEFT,
            2: RailEnvActions.MOVE_FORWARD,
            3: RailEnvActions.MOVE_RIGHT,
            4: RailEnvActions.STOP_MOVING,
        }[value]

    @staticmethod
    def to_char(a: int):
        return {
            0: 'B',
            1: 'L',
            2: 'F',
            3: 'R',
            4: 'S',
        }[a]

    @staticmethod
    @lru_cache()
    def is_action_valid(action):
        if isinstance(action, RailEnvActions):
            return True
        # https://stackoverflow.com/questions/40429917/in-python-how-would-you-check-if-a-number-is-one-of-the-integer-types
        return isinstance(action, (int, np.integer)) and 0 <= action <= 4

    @staticmethod
    @lru_cache()
    def is_moving_action(value: "RailEnvActions") -> bool:
        return value in [RailEnvActions.MOVE_RIGHT, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_FORWARD]


RailEnvGridPos = NamedTuple('RailEnvGridPos', [('r', int), ('c', int)])
RailEnvNextAction = NamedTuple('RailEnvNextAction', [('action', RailEnvActions), ('next_position', RailEnvGridPos),
                                                     ('next_direction', Grid4TransitionsEnum)])
