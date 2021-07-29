from enum import IntEnum

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

    @staticmethod
    def is_malfunction_state(state):
        return state in [2, 5] # TODO: Can this be done with names instead?
    


