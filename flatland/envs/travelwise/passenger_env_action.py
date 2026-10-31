from enum import IntEnum
from typing import Optional, Tuple


class PassengerAction(IntEnum):
    """
    A passenger's action in `PassengerEnv`:

    - `WAIT`: keep waiting, or keep riding;
    - `HOP_ON`: board the vehicle given with the action;
    - `HOP_OFF`: alight from the vehicle ridden, at its current entry point.
    """
    WAIT = 0
    HOP_ON = 1
    HOP_OFF = 2


# A passenger's action with the vehicle handle, given only with `HOP_ON`, e.g. `(PassengerAction.HOP_ON, 3)` or
# `(PassengerAction.WAIT, None)`.
PassengerActionT = Tuple[PassengerAction, Optional[int]]
