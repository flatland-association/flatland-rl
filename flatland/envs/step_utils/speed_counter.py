from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

SEGMENT_LENGTH: Fraction = Fraction(1)


@lru_cache()
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


@lru_cache()
def _cap_speed(agent_max_speed: Fraction, new_speed: Fraction) -> Fraction:
    v = max(Fraction(0), min(agent_max_speed, new_speed))
    assert isinstance(v, Fraction)
    assert v >= 0.0
    assert v <= 1.0
    return v


@lru_cache()
def _cached_cell_exit(_distance, speed: Fraction) -> bool:
    return _distance + speed >= SEGMENT_LENGTH


@lru_cache()
def cached_cell_exit(max_speed: Fraction, speed: Fraction, distance: Fraction) -> bool:
    speed = _cap_speed(max_speed, speed)
    return _cached_cell_exit(distance, speed)


@lru_cache()
def _distance_update(distance: Fraction, speed: Fraction,
                     crossing_completed: bool = True) -> Tuple[Fraction, bool]:
    distance += speed

    if crossing_completed:
        # check assumption
        assert distance >= SEGMENT_LENGTH
        distance = distance % SEGMENT_LENGTH
        return distance, distance < speed

    # move at most segment end
    return min(distance, SEGMENT_LENGTH), False


class SpeedCounter:
    def __init__(self, speed: float, max_speed: float = None):
        self._speed: Fraction = _pseudo_fractional(speed)
        self._distance: Optional[Fraction] = None
        self._is_cell_entry = False
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

    def step(self, speed: Optional[Fraction], crossing_completed: bool) -> None:
        """
        Step the speed counter:
        - the distance traveled this step is computed from the pre-step speed.
        - the speed is updated to the new speed (modulo capping by max speed).

        Parameters
        ----------
        speed : Optional[Fraction]
            The new speed, effective from the next step, or None while off map (leaving the map,
            or staying off map).
        crossing_completed : bool
            Whether the transition into the next cell actually completed.
        """
        if speed is None:
            # design: distance is None when off map
            self._distance = None
            self._is_cell_entry = False
            return
        if self._distance is None:
            # design: distance is None when off map -- entering the map: bootstrap distance to 0
            # instead of advancing from a pre-step speed that does not reflect being on the map yet.
            self._distance = Fraction(0)
            self._is_cell_entry = True
        else:
            self._distance, self._is_cell_entry = _distance_update(self._distance, self._speed, crossing_completed)
        self._speed = _cap_speed(self._max_speed, _pseudo_fractional(speed))

    def stop(self) -> None:
        """
        Freeze speed at 0 without touching distance.

        Use this instead of step() whenever the agent's on-map is malfunction or force stop.
        """
        self._speed = Fraction(0)
        self._is_cell_entry = False

    def __repr__(self):
        return f"speed: {self.speed} \
                 max_speed: {self.max_speed} \
                 distance: {self.distance} \
                 is_cell_entry: {self.is_cell_entry}"

    def reset(self):
        self.step(None, False)

    @property
    def is_cell_entry(self):
        """
        Have just entered the cell in the previous step?
        """
        return self._is_cell_entry

    def is_cell_exit(self) -> bool:
        """
        At the current speed, do we exit the cell at the next time step?
        """
        if self._distance is None:
            # design: distance is None when off map
            return True
        return cached_cell_exit(self._max_speed, self._speed, self._distance)

    @property
    def speed(self) -> Fraction:
        return self._speed

    @property
    def max_speed(self) -> Fraction:
        return self._max_speed

    @property
    def distance(self) -> Optional[Fraction]:
        """
        Distance traveled in current cell. None while off map (until step() is called with a
        non-None speed).
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
