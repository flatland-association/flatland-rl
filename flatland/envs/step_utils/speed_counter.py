import numpy as np
from flatland.envs.step_utils.states import TrainState

class SpeedCounter:
    def __init__(self, speed):
        self.speed = speed
        self.max_count = int(1/speed) - 1

    def update_counter(self, state, old_position):
        # When coming onto the map, do no update speed counter
        if state == TrainState.MOVING and old_position is not None:
            self.counter += 1
            self.counter = self.counter % (self.max_count + 1)

    def __repr__(self):
        return f"speed: {self.speed} \
                 max_count: {self.max_count} \
                 is_cell_entry: {self.is_cell_entry} \
                 is_cell_exit: {self.is_cell_exit} \
                 counter: {self.counter}"

    def reset_counter(self):
        self.counter = 0

    @property
    def is_cell_entry(self):
        return self.counter == 0

    @property
    def is_cell_exit(self):
        return self.counter == self.max_count

