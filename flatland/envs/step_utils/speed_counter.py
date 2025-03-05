from flatland.core.grid.grid_utils import IntVector2D
from flatland.envs.step_utils.states import TrainState

EPSILON = 0.01
SEGMENT_LENGTH = 1


class SpeedCounter:
    def __init__(self, speed: float, max_speed: float = None):
        self._speed = 1 / (round(1 / speed))
        self._distance = 0.0
        self._is_cell_entry = True
        if max_speed is not None:
            self._max_speed = max_speed
        else:
            # old constant speed behaviour
            self._max_speed = self._speed
        assert self._max_speed <= 1.0
        assert self._speed <= self._max_speed
        self.reset()

    def step(self, state: TrainState, old_position: IntVector2D, speed: float = None):
        """
        Step the speed counter.

        Parameters
        ----------
        state : TrainState
            Distance incremented only in MOVING state.
        old_position : IntVector2D
            Distance incremented only if already in grid (when we enter the grid, we enter at position zero).
        speed : float
            Set new speed effective immediately.
        """
        if speed is not None:
            self._speed = max(min(speed, self._max_speed), 0.0)
        # TODO bad code smell: the following condition should not be part of SpeedCounter?
        # Can't start counting when adding train to the map
        if state == TrainState.MOVING and old_position is not None and self._speed > 0:
            self._distance += self._speed

            # If trains cannot move to the next cell, they are in state stopped, so it's safe to apply modulo to reflect the distance travelled in the new cell!
            self._distance = self._distance % SEGMENT_LENGTH
            if self._distance < self._speed:
                self._is_cell_entry = True
            else:
                self._is_cell_entry = False

    def __repr__(self):
        return f"speed: {self.speed} \
                 max_speed: {self.max_speed} \
                 distance: {self.distance} \
                 is_cell_entry: {self.is_cell_entry}"

    def reset(self):
        self._distance = 0
        self._is_cell_entry = True

    # TODO why do we need is_cell_entry this at all? Only two usages one of which seems wrong.
    @property
    def is_cell_entry(self):
        """
        Have just entered the cell in the previous step?
        """
        return self._is_cell_entry

    def is_cell_exit(self, speed: float):
        """
        With the given speed, do we exit cell at next time step?
        """
        speed = max(min(speed, self._max_speed), 0.0)
        return self._distance + speed >= 1.0 - EPSILON

    @property
    def speed(self):
        return self._speed

    @property
    def max_speed(self):
        return self._max_speed

    @property
    def distance(self):
        """
        Distance travelled in current cell.
        """
        return self._distance

    def __getstate__(self):
        return {
            "speed": self._speed,
            "max_speed": self._max_speed,
            "distance": self._distance,
            "is_cell_entry": self._is_cell_entry,
        }

    def __setstate__(self, load_dict):
        if "_speed" in load_dict:
            self._speed = load_dict['_speed']
        else:
            self._speed = load_dict["speed"]
        if "counter" in load_dict:
            # old pickles have constant speed
            self._distance = load_dict['counter'] * self._speed
            self._is_cell_entry = load_dict['counter'] == 0
        else:
            self._distance = load_dict['distance']
        if "is_cell_entry" in load_dict:
            self._is_cell_entry = load_dict['distance']
        if "max_speed" in load_dict:
            self._max_speed = load_dict["max_speed"]
        else:
            # old pickles have constant speed
            self._max_speed = self._speed

    def __eq__(self, other):
        if not isinstance(other, SpeedCounter):
            return False
        return self._speed == other._speed and self._distance == other._distance and self._max_speed == other._max_speed
