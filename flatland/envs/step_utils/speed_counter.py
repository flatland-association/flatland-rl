from functools import lru_cache


@lru_cache()
def _calc_max_count(speed):
    return int(1.0 / speed) - 1


class SpeedCounter:
    def __init__(self, speed):
        self._speed = speed
        self.counter = None
        self.reset_counter()

    def update_counter(self):
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

    @property
    def speed(self):
        return self._speed

    @property
    def max_count(self):
        return _calc_max_count(self._speed)

    def to_dict(self):
        return {"speed": self._speed,
                "counter": self.counter}

    def from_dict(self, load_dict):
        self._speed = load_dict['speed']
        self.counter = load_dict['counter']

    def __eq__(self, other):
        return self._speed == other._speed and self.counter == other.counter
