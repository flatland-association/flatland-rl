from dataclasses import dataclass
from enum import IntEnum


class TrainState(IntEnum):
    # WAITING: No actions allowed here, when current timestep is behind earliest departure.
    WAITING = 0

    # READY_TO_DEPART: Train is ready to move and will start at initial_position when moving action is provided.
    READY_TO_DEPART = 1

    # MALFUNCTION_OFF_MAP: When a randomized malfunction occurs in an off map state, any moving actions provided here are stored and used when malfunction completes, unless stop action is provided.
    MALFUNCTION_OFF_MAP = 2

    # MOVING: Indicates the train is moving, if speed is 1.0, the train changes position every timestep.
    MOVING = 3

    # STOPPED: Indicates the train is stopped, this can occur when the user provides a stop action, or the train tries to move into a cell that is occupied or to a cell which does not have a track.
    STOPPED = 4

    # MALFUNCTION: When a randomized malfunction occurs in an on map state, any moving actions provided here are stored and used when malfunction completes, unless stop action is provided. No movement can occur during a malfunction state.
    MALFUNCTION = 5

    # DONE: This is a terminal state which is activated when the target is reached.
    DONE = 6

    @classmethod
    def check_valid_state(cls, state):
        return state in cls._value2member_map_

    def is_malfunction_state(self):
        return self.value in [self.MALFUNCTION, self.MALFUNCTION_OFF_MAP]

    def is_off_map_state(self):
        return self.value in [self.WAITING, self.READY_TO_DEPART, self.MALFUNCTION_OFF_MAP]

    def is_on_map_state(self):
        return self.value in [self.MOVING, self.STOPPED, self.MALFUNCTION]


@dataclass(repr=True)
class StateTransitionSignals:
    # Malfunction states start when in_malfunction is set to true
    in_malfunction: bool = False

    # Malfunction counter complete - Malfunction state ends this timestep and actions are allowed from next timestep
    malfunction_counter_complete: bool = False

    # Earliest departure reached - Train is allowed to move now
    earliest_departure_reached: bool = False

    # Stop Action Given - User provided a stop action. Action preprocessing can also change a moving action to a stop action if the train tries to move into a invalid or occupied position.
    stop_action_given: bool = False

    # Movement action is provided and no movement conflict (see below).
    valid_movement_action_given: bool = False

    # Target position is reached
    target_reached: bool = False

    # Movement conflict: desired movement (if any) not allowed (i.e. MotionCheck NOK or in malfunction).
    # Desired movement allowed if not in malfunction and desired motion is allowed (if any).
    # Note that a train can move into an occupied cell if that cell is being emptied by the train that was previously occupied it.
    movement_conflict: bool = False
