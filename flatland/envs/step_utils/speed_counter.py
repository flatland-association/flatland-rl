from flatland.core.grid.grid_utils import IntVector2D
from flatland.envs.step_utils.states import TrainState


class SpeedCounter:
    def __init__(self, speed):
        self._speed = speed
        self._distance = 0.0
        self._is_cell_entry = True
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
            self._speed = speed
        # TODO bad code smell: this logic should not be part of SpeedCounter?
        # Can't start counting when adding train to the map
        if state == TrainState.MOVING and old_position is not None:
            self._distance += self._speed
            # travelling cells in any direction has distance 1
            # trains are in state stopped if they cannot move to the next cell
            self._distance = self._distance % 1
            if self._distance < self._speed:
                self._is_cell_entry = True
            else:
                self._is_cell_entry = False

    def __repr__(self):
        return f"speed: {self.speed} \
                 distance: {self.distance} \
                 is_cell_entry: {self.is_cell_entry} \
                 is_cell_exit: {self.is_cell_exit}"

    def reset(self):
        self._distance = 0
        self._is_cell_entry = True

    # TODO why do we need this at all?
    @property
    def is_cell_entry(self):
        """
        Have just entered the cell in the previous step?
        """
        return self._is_cell_entry

    @property
    def is_cell_exit(self):
        """
        With current speed, do we exit cell at next time step?
        """
        return self._distance + self._speed >= 1.0

    @property
    def speed(self):
        return self._speed

    @property
    def distance(self):
        """
        Distance travelled in current cell.
        """
        return self._distance

    def __getstate__(self):
        return {
            "speed": self._speed,
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

    def __eq__(self, other):
        return self._speed == other._speed and self._distance == other._distance
