from enum import IntEnum

class TrainState(IntEnum):
    WAITING = 0
    READY_TO_DEPART = 1
    MOVING = 1
    STOPPED = 2
    MALFUNCTION = 3
    DONE = 4