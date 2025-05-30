from ast import literal_eval
from functools import lru_cache
from typing import NamedTuple, Any

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
    def from_value(value: Any) -> "RailEnvActions":
        """
        Returns the action if valid (either int value or in RailEnvActions), returns RailEnvActions.DO_NOTHING otherwise.
        """
        if isinstance(value, RailEnvActions):
            return value

        if isinstance(value, str) and value.isdigit():
            value = literal_eval(value)
        return {
            0: RailEnvActions.DO_NOTHING,
            "DO_NOTHING": RailEnvActions.DO_NOTHING,
            "RailEnvActions.DO_NOTHING": RailEnvActions.DO_NOTHING,
            1: RailEnvActions.MOVE_LEFT,
            "MOVE_LEFT": RailEnvActions.MOVE_LEFT,
            "RailEnvActions.MOVE_LEFT": RailEnvActions.MOVE_LEFT,
            2: RailEnvActions.MOVE_FORWARD,
            "MOVE_FORWARD": RailEnvActions.MOVE_FORWARD,
            "RailEnvActions.MOVE_FORWARD": RailEnvActions.MOVE_FORWARD,
            3: RailEnvActions.MOVE_RIGHT,
            "MOVE_RIGHT": RailEnvActions.MOVE_RIGHT,
            "RailEnvActions.MOVE_RIGHT": RailEnvActions.MOVE_RIGHT,
            4: RailEnvActions.STOP_MOVING,
            "STOP_MOVING": RailEnvActions.STOP_MOVING,
            "RailEnvActions.STOP_MOVING": RailEnvActions.STOP_MOVING,
        }.get(value, RailEnvActions.DO_NOTHING)

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

    @staticmethod
    @lru_cache()
    def is_left_right_action(value: "RailEnvActions") -> bool:
        return value in [RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT]


RailEnvGridPos = NamedTuple('RailEnvGridPos', [('r', int), ('c', int)])
RailEnvNextAction = NamedTuple('RailEnvNextAction', [('action', RailEnvActions), ('next_position', RailEnvGridPos),
                                                     ('next_direction', Grid4TransitionsEnum)])
