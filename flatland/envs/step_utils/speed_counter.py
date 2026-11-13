from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
from typing import Optional

import numpy as np

from flatland.core.transition_map import DEFAULT_EDGE_LENGTH, EdgeLength, MAX_SPEED, validated_edge_length

# shared zero-Fraction constant - Fraction is immutable, so every "return/set to zero speed" call site
# can safely reuse the same instance instead of constructing a fresh Fraction(0) - profiled at ~12x the
# cost of referencing an already-built constant.
ZERO_FRACTION: Fraction = Fraction(0)


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
    v = max(ZERO_FRACTION, min(agent_max_speed, new_speed))
    assert isinstance(v, Fraction)
    assert v >= 0.0
    assert v <= MAX_SPEED
    return v


@lru_cache()
def cached_cell_exit(max_speed: Fraction, speed: Optional[Fraction], distance: Optional[Fraction],
                     edge_length: Optional[EdgeLength]) -> bool:
    """
    Would this speed/distance cross the boundary of the current edge (of length `edge_length`) next step?
    `distance is None` (off map) is
    trivially `True` (see `SpeedCounter.set()`) - checked first, before `_cap_speed`, which can't handle
    `speed=None`. All on-map call sites already only reach this after excluding off_map themselves, so
    this convention is never actually read bare off map (see design_by_contract.md's `cell_exit` row).
    """
    if distance is None:
        return True
    speed = _cap_speed(max_speed, speed)
    if speed == 0:
        return False
    return distance + speed >= edge_length


class SpeedCounter:
    """
    Tracks an agent's speed and within-edge distance as Fractions, together with the length of the edge
    `(current entry point, next entry point)` the agent is currently on. speed/distance/edge_length are
    None while off map (see set()) and become concrete Fractions once the agent enters the map.

    On map, distance is in [0, edge_length): an agent never sits on the exit of its edge - a crossing wraps
    into the next edge, also into a target edge (target reached, not removed), and a denied crossing keeps the
    pre-step distance.

    Definition: speed is the distance travelled per step - each step adds the pre-step speed to the distance. Its
    inverse `1 / speed` is the number of steps it takes to pass through an edge of length 1 (`edge_length / speed`
    steps for an edge of length `edge_length`).

    Four static, lru_cache'd formulas compute the expected post-step speed/distance from explicit
    pre-step inputs - used by RailEnv.step() to compute candidate speed/distance (which set() is then
    called with directly), and again by its post-step invariant checks
    (_check_speed_distance_speedup_postconditions) to verify the actual post-step values match. They are static
    rather than instance methods since they compute a *candidate* value from explicit pre-step inputs,
    not the counter's own (post-step) state; acceleration_delta/braking_delta are env-level parameters,
    not SpeedCounter state, so are passed in explicitly rather than read off self. All four return None
    off map, i.e. when `pre_speed` (speed formulas) or `distance` (distance formulas) is None.

    | Method | Formula | Situation |
    |---|---|---|
    | `speed_after_acceleration` | `min(pre_speed + acceleration_delta, max_speed)` | `MOVE_FORWARD` / start from rest |
    | `speed_after_braking` | `max(pre_speed + braking_delta, 0)` | `STOP_MOVING` |
    | `distance_after_crossing` | `(distance + pre_speed - edge_length) % MAX_SPEED` if `distance + pre_speed >= edge_length`, else `distance + pre_speed` | keep moving, or cross (also into a target edge) |
    | `distance_without_crossing` | `min(distance + pre_speed, edge_length)` | not used by `RailEnv.step()` |

    A denied crossing (invalid action or resource_check) uses none of them: the agent doesn't move, its
    distance stays at the pre-step value and its speed is set to 0 (see RailEnv.step()'s (13)).
    """

    def __init__(self, max_speed: float, speed: Optional[float] = None):
        self._max_speed: Fraction = _pseudo_fractional(max_speed)
        assert self._max_speed <= MAX_SPEED
        # design: speed is None until the agent enters the map (see step())
        self._speed: Optional[Fraction] = _pseudo_fractional(speed) if speed is not None else None
        if self._speed is not None:
            assert self._speed <= self._max_speed
            assert self._speed >= 0.0
        self._distance: Optional[Fraction] = None
        self._edge_length: Optional[EdgeLength] = None
        self._is_cell_entry = False

    def set(self, speed: Optional[Fraction], distance: Optional[Fraction], edge_length: Optional[EdgeLength],
            crossed: bool = False) -> None:
        """
        Directly set speed/distance to an already-computed value - e.g. RailEnv.step()'s own
        candidate_speed/candidate_distance (accepted), or the pre-step distance with speed Fraction(0)
        when the candidate was rejected. Unlike the old step(), this does not itself derive distance from
        a crossing decision - the caller has already made that decision and passes it as `crossed`;
        set() only tracks the resulting state and derives is_cell_entry from it.

        Parameters
        ----------
        speed : Optional[Fraction]
            The new speed, or None while off map (leaving the map, or staying off map).
        distance : Optional[Fraction]
            The new within-cell distance, or None while off map. Both speed and distance are None
            together, or both concrete Fractions together - never mixed (see the class docstring).
        edge_length : Optional[EdgeLength]
            The length of the edge the agent is on after this update, i.e.
            `rail.edge_length(current_entry_point, next_entry_point)` - ignored while off map (distance None).
        crossed : bool
            Whether the agent crossed into a new edge with this update (on map before and after).
        """
        # design: is_cell_entry is "just entered a new cell" - true exactly when the agent just
        # bootstrapped onto the map (previous distance None, new distance concrete), or the caller says it
        # crossed. It can't be derived from the distances alone: at speed == 1 (e.g., max_speed=1) on a
        # unit-length edge, distance_after_crossing wraps exactly back to the pre-step value (0 -> 0), and a
        # denied crossing keeps the pre-step distance (with speed 0), just like an agent at rest.
        self._is_cell_entry = distance is not None and (self._distance is None or crossed)
        self._distance = distance
        self._edge_length = edge_length if distance is not None else None
        # design: speed and distance are None together while off map - force speed None whenever
        # distance is None, regardless of what was passed, so a caller's off-map placeholder speed
        # (e.g. _candidate_speed's own "stay off map" branch, which always returns Fraction(0) rather
        # than None - see its docstring) can never leave the two inconsistent.
        # speed is always already a Fraction at every real call site (candidate_speed, Fraction(0)) -
        # _pseudo_fractional's float-tolerance conversion never actually fires here, but its lru_cache
        # wrapper still costs ~6x a plain isinstance check for a no-op cache hit (profiled) - skip it.
        self._speed = (_cap_speed(self._max_speed, speed if isinstance(speed, Fraction) else _pseudo_fractional(speed))
                       if (speed is not None and distance is not None) else None)

    def stop(self) -> None:
        """
        Freeze speed at 0 without touching distance.

        Use this instead of step() whenever the agent's on-map is malfunction or force stop.
        """
        self._speed = ZERO_FRACTION
        self._is_cell_entry = False

    @staticmethod
    @lru_cache()
    def speed_after_acceleration(pre_speed: Optional[Fraction], max_speed: Fraction,
                                 acceleration_delta: Fraction) -> Optional[Fraction]:
        """
        Expected speed after a MOVE_FORWARD action, or a movement action starting an agent moving from
        rest: pre-step speed plus acceleration_delta, capped at max_speed. Off map (`pre_speed` is None), returns None.

        Physical motivation: a real train's engine has a limited power/torque budget, so it cannot jump
        straight to its running speed - it accelerates gradually, gaining at most acceleration_delta of
        speed per step, whether it is starting from a standstill (STOPPED/READY_TO_DEPART -
        MOVING, or a just-recovered MALFUNCTION) or already rolling and simply told to keep going faster
        (MOVE_FORWARD while MOVING). max_speed models the train's own top speed (or a speed restriction
        on this track section) that acceleration can never exceed.
        """
        if pre_speed is None:
            return None
        return min(pre_speed + acceleration_delta, max_speed)

    @staticmethod
    @lru_cache()
    def speed_after_braking(pre_speed: Optional[Fraction], braking_delta: Fraction) -> Optional[Fraction]:
        """
        Expected speed after a STOP_MOVING action: pre-step speed plus (negative) braking_delta,
        floored at 0. Off map (`pre_speed` is None), returns None.

        Physical motivation: braking is the operator's own controlled deceleration, not an instant
        halt - braking_delta models the brake's fixed deceleration rate per step, the mirror image of
        acceleration_delta. This is the operator's deliberate choice (distinct from being force-stopped
        by an invalid action or a resource_check denial, see RailEnv.step()'s (13)) - a train
        already at or very near the cell boundary when STOP_MOVING is given can still complete an
        already-in-flight crossing on the very step it comes to a stop (see distance_after_crossing).
        """
        if pre_speed is None:
            return None
        return max(pre_speed + braking_delta, 0)

    @staticmethod
    @lru_cache()
    def distance_after_crossing(distance: Optional[Fraction], pre_speed: Optional[Fraction],
                                edge_length: Optional[EdgeLength]) -> Optional[Fraction]:
        """
        Expected post-step distance of an agent that keeps moving: pre-step distance plus pre-step speed - just
        `distance + pre_speed` while it stays below `edge_length`, otherwise the agent crosses into the next edge and
        keeps the leftover `distance + pre_speed - edge_length`. Off map (`distance` is None, and then so are
        `pre_speed` and `edge_length`), returns None.

        Physical motivation: the train's momentum this step carries it across the boundary and some
        distance into the next cell - the wrap (`(distance + pre_speed - edge_length) % MAX_SPEED`) is exactly that leftover distance,
        i.e. how far into the new cell the train's momentum actually reaches. This is the only one of
        the four formulas that ever transitions the agent's current_entry_point/next_entry_point into a
        new cell (see RailEnv.step()'s (12)/(13)); the granting action can be MOVE_FORWARD/MOVE_LEFT/
        MOVE_RIGHT while MOVING (pre-step speed > 0), or an explicit STOP_MOVING if the
        crossing was already in flight before the brake takes effect (see speed_after_braking above) -
        the action itself never matters once resource_check has granted the crossing.

        The leftover is in [0, `MAX_SPEED`), so below the new edge's length (every edge is at least `MIN_EDGE_LENGTH`
        = `MAX_SPEED` long): on map, `distance` is below `edge_length` (see the class docstring) and `pre_speed` is
        at most `MAX_SPEED`, so the modulo changes no value an agent on its way reaches. Physically, having crossed,
        the train is inside the new edge, never on the boundary it just crossed - also when crossing into a target
        edge (target reached, not removed).
        """
        if distance is None:
            return None
        leftover = distance + pre_speed - edge_length
        if leftover < 0:
            return distance + pre_speed
        return leftover % MAX_SPEED

    @staticmethod
    @lru_cache()
    def distance_without_crossing(distance: Optional[Fraction], pre_speed: Optional[Fraction],
                                  edge_length: Optional[EdgeLength]) -> Optional[Fraction]:
        """
        Pre-step distance plus pre-step speed, capped at the current edge's boundary (`edge_length`) rather than
        wrapping into a new edge. Off map (`distance` is None, and then so are `pre_speed` and `edge_length`),
        returns None.

        Not used by RailEnv.step(): an agent whose target is reached and not removed (remove_agents_at_target=False)
        crosses into its target edge like any other crossing (distance_after_crossing), and its distance stays there
        once it is DONE (see RailEnv._candidate_distance).

        Not used for a denied crossing (invalid action or resource_check): there the agent doesn't move at
        all - its distance stays at the pre-step value (see RailEnv.step()'s (13)), so a restart after the
        denial covers the remaining distance to the boundary, and no distance is dropped when it crosses.
        """
        if distance is None:
            return None
        return min(distance + pre_speed, edge_length)

    def __repr__(self):
        return f"speed: {self.speed} \
                 max_speed: {self.max_speed} \
                 distance: {self.distance} \
                 edge_length: {self.edge_length} \
                 is_cell_entry: {self.is_cell_entry}"

    def reset(self):
        self.set(None, None, None)

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
        return cached_cell_exit(self._max_speed, self._speed, self._distance, self._edge_length)

    @property
    def speed(self) -> Optional[Fraction]:
        """
        Current speed. None while off map (until step() is called with a non-None speed).
        """
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

    @property
    def edge_length(self) -> Optional[EdgeLength]:
        """
        Length of the edge (current entry point -> next entry point) the agent is on. None while off map.
        """
        return self._edge_length

    def __getstate__(self):
        return {
            "speed": self._speed,
            "max_speed": self._max_speed,
            "distance": self._distance,
            "edge_length": self._edge_length,
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
        if "edge_length" in load_dict:
            self._edge_length = validated_edge_length(load_dict["edge_length"]) if load_dict["edge_length"] is not None else None
        else:
            # old pickles have unit-length edges only
            self._edge_length = DEFAULT_EDGE_LENGTH if self._distance is not None else None

    def __eq__(self, other):
        if not isinstance(other, SpeedCounter):
            return False
        return (self._speed == other._speed and self._distance == other._distance and self._max_speed == other._max_speed
                and self._edge_length == other._edge_length)
