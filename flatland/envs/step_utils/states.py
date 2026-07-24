from enum import IntEnum
from functools import lru_cache


class TrainState(IntEnum):
    WAITING = 0
    READY_TO_DEPART = 1
    MALFUNCTION_OFF_MAP = 2
    MOVING = 3
    STOPPED = 4
    MALFUNCTION = 5
    DONE = 6

    @classmethod
    def check_valid_state(cls, state):
        return state in cls._value2member_map_

    @lru_cache
    def is_malfunction_state(self):
        return self.value in [self.MALFUNCTION, self.MALFUNCTION_OFF_MAP]

    @lru_cache
    def is_off_map_state(self):
        return self.value in [self.WAITING, self.READY_TO_DEPART, self.MALFUNCTION_OFF_MAP]

    @lru_cache
    def is_on_map_state(self):
        return self.value in [self.MOVING, self.STOPPED, self.MALFUNCTION]


class StateTransitionSignals:
    def __init__(self, in_malfunction: bool = False, earliest_departure_reached: bool = False, stop_action_given: bool = False,
                 movement_action_given: bool = False, target_reached: bool = False, movement_allowed: bool = False, new_speed_zero: bool = False):
        self.in_malfunction = in_malfunction
        self.earliest_departure_reached = earliest_departure_reached
        self.stop_action_given = stop_action_given
        self.movement_action_given = movement_action_given
        self.target_reached = target_reached
        self.movement_allowed = movement_allowed
        self.new_speed_zero = new_speed_zero

    def __repr__(self):
        return (f"StateTransitionSignals(in_malfunction={self.in_malfunction}, earliest_departure_reached={self.earliest_departure_reached}, "
                f"stop_action_given={self.stop_action_given}, movement_action_given={self.movement_action_given}, "
                f"target_reached={self.target_reached}, movement_allowed={self.movement_allowed}, new_speed_zero={self.new_speed_zero})")

    def __eq__(self, other):
        return isinstance(other, StateTransitionSignals) and (
            self.in_malfunction, self.earliest_departure_reached, self.stop_action_given, self.movement_action_given,
            self.target_reached, self.movement_allowed, self.new_speed_zero
        ) == (
            other.in_malfunction, other.earliest_departure_reached, other.stop_action_given, other.movement_action_given,
            other.target_reached, other.movement_allowed, other.new_speed_zero
        )

    def __getstate__(self):
        return {
            "in_malfunction": self.in_malfunction,
            "earliest_departure_reached": self.earliest_departure_reached,
            "stop_action_given": self.stop_action_given,
            "movement_action_given": self.movement_action_given,
            "target_reached": self.target_reached,
            "movement_allowed": self.movement_allowed,
            "new_speed_zero": self.new_speed_zero,
        }

    def __setstate__(self, state):
        # tolerate missing/renamed/extra keys: old pickled fixtures predate several field renames in this
        # class (e.g. valid_movement_action_given -> movement_action_given, movement_conflict ->
        # movement_allowed, and the now-removed malfunction_counter_complete) - a plain Python object's
        # default __dict__.update(state) silently absorbed this drift (unknown keys just became inert extra
        # __dict__ entries, missing keys left as whatever the pre-existing instance already had); mirror that
        # tolerance here rather than requiring every key to be present and known.
        self.in_malfunction = state.get("in_malfunction", False)
        self.earliest_departure_reached = state.get("earliest_departure_reached", False)
        self.stop_action_given = state.get("stop_action_given", False)
        self.movement_action_given = state.get("movement_action_given", False)
        self.target_reached = state.get("target_reached", False)
        self.movement_allowed = state.get("movement_allowed", False)
        self.new_speed_zero = state.get("new_speed_zero", False)
