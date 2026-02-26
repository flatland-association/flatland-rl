from decimal import Decimal
from fractions import Fraction
from typing import Optional

import numpy as np

SEGMENT_LENGTH: Fraction = Fraction(1)


def _pseudo_fractional(v: Optional[float], atol=1.e-2) -> Optional[Fraction]:
    """
    Convert float to fractional with special consideration of inverses of integers.
    E.g. with tolerance `atol=1.e-2`, `float(0.33)` is converted to `Fraction(1,3)`.

    Parameters
    ----------
    v : Optional[float]
    d    the float to be converted to fractional; if the float is the inverse of an integer by tolerance, then the corresponding fraction is returned
    atol : float
        the tolerance to determine inverse of integers

    Returns
    -------
    Fraction
    """
    if v is None:
        return None
    elif isinstance(v, Fraction):
        return v
    elif isinstance(v, Decimal):
        return Fraction.from_decimal(v)
    elif isinstance(v, int):
        return Fraction(v)
    elif isinstance(v, float):
        if np.isclose(v % 1, 0.0):
            return Fraction(0, 1) + int(v // 1)
        elif np.isclose(1 / round(1 / (v % 1)), v % 1, atol=atol):
            return Fraction(1, round(1 / (v % 1))) + int(v // 1)
        elif v < 0 and np.isclose(1 / round(1 / ((-v) % 1)), (-v) % 1, atol=atol):
            return - Fraction(1, round(1 / ((-v) % 1))) + int((-v) // 1)
        elif np.isclose(float(Decimal(str(v))), v):
            return Fraction.from_decimal(Decimal(str(v)))
        else:
            return Fraction.from_float(v)
    raise ValueError(f"Cannot convert {v} to Fraction.")


class SpeedCounter:
    def __init__(self, speed: float, max_speed: float = None):
        self._speed: Fraction = _pseudo_fractional(speed)
        self._distance: Fraction = Fraction(0)
        self._is_cell_entry = True
        self._max_speed: Fraction
        if max_speed is not None:
            self._max_speed = _pseudo_fractional(max_speed)
        else:
            # old constant speed behaviour
            self._max_speed = self._speed
        assert self._max_speed <= 1.0
        assert self._speed <= self._max_speed
        assert self._speed >= 0.0
        self.reset()

    def step(self, speed: Fraction = None):
        """
        Step the speed counter.

        Parameters
        ----------
        speed : Fraction
            Set new speed effective immediately.
        """

        if speed is not None:
            self._speed = max(min(_pseudo_fractional(speed), self._max_speed), Fraction(0))
        assert isinstance(self._speed, Fraction)
        assert self._speed >= 0.0
        assert self.speed <= 1.0

        self._distance += self._speed

        # If trains cannot move to the next cell, they are in state stopped, so it's safe to apply modulo to reflect the distance travelled in the new cell!
        while self.distance >= SEGMENT_LENGTH:
            self._distance = self._distance - SEGMENT_LENGTH
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

    @property
    def is_cell_entry(self):
        """
        Have just entered the cell in the previous step?
        """
        return self._is_cell_entry

    def is_cell_exit(self, speed: Fraction):
        """
        With the given speed, do we exit cell at next time step?
        """
        speed = max(min(speed, self._max_speed), Fraction(0))
        return self._distance + speed >= SEGMENT_LENGTH

    @property
    def speed(self) -> Fraction:
        return self._speed

    @property
    def max_speed(self) -> Fraction:
        return self._max_speed

    @property
    def distance(self) -> Fraction:
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
            # backwards compatibility
            self._speed = _pseudo_fractional(load_dict['_speed'])
        else:
            self._speed = _pseudo_fractional(load_dict["speed"])
        if "counter" in load_dict:
            # old pickles have constant speed
            self._distance = _pseudo_fractional(load_dict['counter'] * self._speed)
            self._is_cell_entry = load_dict['counter'] == 0
        else:
            self._distance = _pseudo_fractional(load_dict['distance'])
        if "is_cell_entry" in load_dict:
            self._is_cell_entry = load_dict['is_cell_entry']
        if "max_speed" in load_dict:
            self._max_speed = _pseudo_fractional(load_dict["max_speed"])
        else:
            # old pickles have constant speed
            self._max_speed = _pseudo_fractional(self._speed)

    def __eq__(self, other):
        if not isinstance(other, SpeedCounter):
            return False
        return self._speed == other._speed and self._distance == other._distance and self._max_speed == other._max_speed
