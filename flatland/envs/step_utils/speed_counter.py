from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
from typing import Optional

import numpy as np

# length of an edge `(u, v)` (current entry point -> next entry point) whose length is not explicitly given -
# see `AbstractRailEnv.edge_length()`.
DEFAULT_EDGE_LENGTH: Fraction = Fraction(1)
# smallest admissible edge length: speed is capped at 1 (see `_cap_speed`), so with every edge at least this
# long an agent crosses at most one edge boundary per step, and the leftover distance after a crossing always
# fits into the newly entered edge.
MIN_EDGE_LENGTH: Fraction = Fraction(1)
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


def validated_edge_length(length) -> Fraction:
    """
    Convert an edge length to `Fraction` (see `_pseudo_fractional`) and check it is at least `MIN_EDGE_LENGTH`.

    Raises
    ------
    ValueError
        if the length is shorter than `MIN_EDGE_LENGTH`.
    """
    length = _pseudo_fractional(length)
    if length is None or length < MIN_EDGE_LENGTH:
        raise ValueError(f"Edge length must be at least {MIN_EDGE_LENGTH}, found {length}.")
    return length


@lru_cache()
def _cap_speed(agent_max_speed: Fraction, new_speed: Fraction) -> Fraction:
    v = max(ZERO_FRACTION, min(agent_max_speed, new_speed))
    assert isinstance(v, Fraction)
    assert v >= 0.0
    assert v <= 1.0
    return v


@lru_cache()
def cached_cell_exit(max_speed: Fraction, speed: Optional[Fraction], distance: Optional[Fraction],
                     edge_length: Optional[Fraction]) -> bool:
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

    Four static, lru_cache'd formulas compute the expected post-step speed/distance from explicit
    pre-step inputs - used by RailEnv.step() to compute candidate speed/distance (which set() is then
    called with directly), and again by its post-step invariant checks
    (_check_speed_distance_speedup_postconditions) to verify the actual post-step values match. They are static
    rather than instance methods since they compute a *candidate* value from explicit pre-step inputs,
    not the counter's own (post-step) state; acceleration_delta/braking_delta are env-level parameters,
    not SpeedCounter state, so are passed in explicitly rather than read off self. All four are None
    (off map) in, None out.

    | Method | Formula | Situation |
    |---|---|---|
    | `speed_after_acceleration` | `min(pre_speed + acceleration_delta, max_speed)` | `MOVE_FORWARD` / start from rest |
    | `speed_after_braking` | `max(pre_speed + braking_delta, 0)` | `STOP_MOVING` |
    | `distance_after_crossing` | `(distance + pre_speed - edge_length) % 1` if `distance + pre_speed >= edge_length`, else `distance + pre_speed` | cell boundary crossed |
    | `distance_without_crossing` | `min(distance + pre_speed, edge_length)` | cell boundary not crossed |
    """

    def __init__(self, max_speed: float, speed: Optional[float] = None):
        self._max_speed: Fraction = _pseudo_fractional(max_speed)
        assert self._max_speed <= 1.0
        # design: speed is None until the agent enters the map (see step())
        self._speed: Optional[Fraction] = _pseudo_fractional(speed) if speed is not None else None
        if self._speed is not None:
            assert self._speed <= self._max_speed
            assert self._speed >= 0.0
        self._distance: Optional[Fraction] = None
        self._edge_length: Optional[Fraction] = None
        self._is_cell_entry = False

    def set(self, speed: Optional[Fraction], distance: Optional[Fraction],
            edge_length: Optional[Fraction] = DEFAULT_EDGE_LENGTH) -> None:
        """
        Directly set speed/distance to an already-computed value - e.g. RailEnv.step()'s own
        candidate_speed/candidate_distance (accepted), or the discarded-candidate formulas
        (distance_without_crossing/Fraction(0)) when the candidate was rejected. Unlike the old step(),
        this does not itself derive distance from a crossing decision - the caller has already made
        that decision; set() only tracks the resulting state and derives is_cell_entry from it.

        Parameters
        ----------
        speed : Optional[Fraction]
            The new speed, or None while off map (leaving the map, or staying off map).
        distance : Optional[Fraction]
            The new within-cell distance, or None while off map. Both speed and distance are None
            together, or both concrete Fractions together - never mixed (see the class docstring).
        edge_length : Optional[Fraction]
            The length of the edge the agent is on after this update (ignored while off map, i.e.
            distance None).
        """
        # design: is_cell_entry is "just entered a new cell" - true exactly when the agent just
        # bootstrapped onto the map (previous distance None, new distance concrete), or the pre-step
        # distance/speed pair implies a crossing was attempted (old_distance + old_speed reaches or
        # exceeds the old edge's length) *and* the new distance shows it landed inside the fresh cell rather
        # than banked at the exit (new_distance < new edge's length - if the crossing was denied, old and new
        # edge are the same). A plain "new < old" comparison isn't
        # enough on its own: at speed == 1 (e.g. max_speed=1) on a unit-length edge, distance_after_crossing's
        # modulo wraps exactly back to the pre-step value (0 -> 0), so "new < old" and even "new <= old"
        # both misclassify it - the latter also misfires on a stopped/banked agent whose distance is
        # unchanged for an unrelated reason (denied at the boundary, or genuinely at rest). Old_speed
        # here is this call's *pre*-step self._speed - it isn't overwritten until after this block.
        self._is_cell_entry = (
            (self._distance is None and distance is not None)
            or (
                self._distance is not None and distance is not None and self._speed is not None
                and self._distance + self._speed >= self._edge_length
                and distance < edge_length
            )
        )
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
        rest: pre-step speed plus acceleration_delta, capped at max_speed. None (off map) in, None out.

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
        floored at 0. None (off map) in, None out.

        Physical motivation: braking is the operator's own controlled deceleration, not an instant
        halt - braking_delta models the brake's fixed deceleration rate per step, the mirror image of
        acceleration_delta. This is the operator's deliberate choice (distinct from being force-stopped
        by an invalid action or a resource_check denial, see distance_without_crossing below) - a train
        already at or very near the cell boundary when STOP_MOVING is given can still complete an
        already-in-flight crossing on the very step it comes to a stop (see distance_after_crossing).
        """
        if pre_speed is None:
            return None
        return max(pre_speed + braking_delta, 0)

    @staticmethod
    @lru_cache()
    def distance_after_crossing(distance: Optional[Fraction], pre_speed: Optional[Fraction],
                                edge_length: Optional[Fraction]) -> Optional[Fraction]:
        """
        Expected distance when this step's cell-boundary crossing completed: pre-step distance plus
        pre-step speed, wrapped into the newly-entered cell. Also used mid-edge (no boundary reached,
        `distance + pre_speed < edge_length`), where it is just `distance + pre_speed`. None (off map) in,
        None out.

        Physical motivation: the train's momentum this step carries it across the boundary of the current
        edge (of length `edge_length`) and some distance into the next one - the leftover
        `distance + pre_speed - edge_length` is exactly how far into the new edge the train's momentum
        actually reaches. This is the only one of
        the four formulas that ever transitions the agent's current_entry_point/next_entry_point into a
        new cell (see RailEnv.step()'s (12)/(13)); the granting action can be MOVE_FORWARD/MOVE_LEFT/
        MOVE_RIGHT while MOVING (pre-step speed > 0), or an explicit STOP_MOVING if the
        crossing was already in flight before the brake takes effect (see speed_after_braking above) -
        the action itself never matters once resource_check has granted the crossing.

        After a crossing, the result is always in [0, 1) and hence strictly inside the new edge (every edge
        is at least `MIN_EDGE_LENGTH` = 1 long) - physically, having actually crossed into the new edge, the
        train is now somewhere inside it, never still sitting exactly on its exit boundary (that "sitting on
        the boundary" case is distance_without_crossing below, where the crossing does *not* happen). The
        leftover is at most 1 (distance <= edge_length, pre_speed <= 1); it equals 1 only for an agent banked
        at the boundary crossing at full speed 1, where the `% 1` wraps it to 0 (as for unit-length edges).
        """
        if distance is None:
            return None
        leftover = distance + pre_speed - edge_length
        if leftover < 0:
            return distance + pre_speed
        return leftover % 1

    @staticmethod
    @lru_cache()
    def distance_without_crossing(distance: Optional[Fraction], pre_speed: Optional[Fraction],
                                  edge_length: Optional[Fraction]) -> Optional[Fraction]:
        """
        Expected distance when this step's cell-boundary crossing did not complete (denied by an
        invalid action or resource_check, or parked at/beyond a just-reached target): pre-step distance
        plus pre-step speed, capped at the current edge's boundary (`edge_length`) rather than wrapping into a
        new cell. None (off map) in, None out.

        Physical motivation: the train's momentum this step still carries it right up to the cell
        boundary - it just isn't credited with crossing it, so it's parked exactly there instead of
        past it. This is real physical distance actually covered this step (pre_speed is always > 0 in
        every case below - a MOVING agent's pre-step speed can never be 0).
        This formula is deliberately the *same* for an invalid action and a resource_check denial - both
        are the environment overriding the operator's request against its will, physically identical
        (a real, momentum-carrying train braked to a stop right at the boundary), so both get the same
        consequence. A granted STOP_MOVING never lands here on its own: if resource_check grants a
        in-flight crossing (is_cell_exit() already true pre-step), the crossing completes via
        distance_after_crossing above regardless of the action - STOP_MOVING cannot itself hold a
        crossing back once resource_check has approved it.

        The result equals exactly edge_length (never something less) in each of the following cases -
        all three only ever apply once `is_cell_exit()` was already true pre-step (i.e. distance + speed reaches edge_length):
        - an invalid action denies the crossing attempt at the cell boundary (e.g. going straight
          through a symmetric switch, which only allows turning) - the operator's own mistaken
          instruction, physically a real train braking hard right at the switch rather than derailing;
        - resource_check denies the crossing to another agent (a real motion conflict - the track ahead
          is physically occupied, so the train brakes to a stop just short of it, same as an emergency
          stop signal);
        - the agent reaches its target this step (remove_agents_at_target=False) - parked at the target,
          capped since there is no further cell to wrap into.
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
        self.set(None, None)

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
    def edge_length(self) -> Optional[Fraction]:
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
            self._edge_length = _pseudo_fractional(load_dict["edge_length"])
        else:
            # old pickles have unit-length edges only
            self._edge_length = DEFAULT_EDGE_LENGTH if self._distance is not None else None

    def __eq__(self, other):
        if not isinstance(other, SpeedCounter):
            return False
        return (self._speed == other._speed and self._distance == other._distance and self._max_speed == other._max_speed
                and self._edge_length == other._edge_length)
