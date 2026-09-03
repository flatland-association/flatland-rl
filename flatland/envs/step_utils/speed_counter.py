from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
from typing import Optional

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


class SpeedCounter:
    """
    Tracks an agent's speed and within-cell distance as Fractions of SEGMENT_LENGTH. speed/distance are
    None while off map (see set()) and become concrete Fractions once the agent enters the map.

    Four static, lru_cache'd formulas compute the expected post-step speed/distance from explicit
    pre-step inputs - used by RailEnv.step() to compute candidate speed/distance (which set() is then
    called with directly), and again by its post-step invariant checks
    (_check_post_speed_distance_invariants) to verify the actual post-step values match. They are static
    rather than instance methods since they compute a *candidate* value from explicit pre-step inputs,
    not the counter's own (post-step) state; acceleration_delta/braking_delta are env-level parameters,
    not SpeedCounter state, so are passed in explicitly rather than read off self. All four are None
    (off map) in, None out.

    | Method | Formula | Situation |
    |---|---|---|
    | `speed_after_acceleration` | `min(pre_speed + acceleration_delta, max_speed)` | `MOVE_FORWARD` / start from rest |
    | `speed_after_braking` | `max(pre_speed + braking_delta, 0)` | `STOP_MOVING` |
    | `distance_after_crossing` | `(pre_offset + pre_speed) % SEGMENT_LENGTH` | cell boundary crossed |
    | `distance_without_crossing` | `min(pre_offset + pre_speed, SEGMENT_LENGTH)` | cell boundary not crossed |
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
        self._is_cell_entry = False

    def set(self, speed: Optional[Fraction], distance: Optional[Fraction]) -> None:
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
        """
        # design: is_cell_entry is "just entered a new cell" - true exactly when distance dropped
        # relative to its previous value (a wrap into a new cell), or the agent just bootstrapped onto
        # the map (previous distance None, new distance concrete). distance never decreases within the
        # same cell (both distance_after_crossing's modulo and distance_without_crossing's cap only ever
        # hold or grow it), so "new < old" is unambiguously a crossing, not mid-cell noise. Unlike the
        # old step()'s new_distance < speed (which reduces to old_distance < SEGMENT_LENGTH under
        # crossing_completed, wrongly False for an agent banked exactly at the boundary that resumes and
        # genuinely crosses), this needs no separate crossing_completed flag at all.
        self._is_cell_entry = (
            (self._distance is None and distance is not None)
            or (self._distance is not None and distance is not None and distance < self._distance)
        )
        self._distance = distance
        # design: speed and distance are None together while off map - force speed None whenever
        # distance is None, regardless of what was passed, so a caller's off-map placeholder speed
        # (e.g. _candidate_speed's own "stay off map" branch, which always returns Fraction(0) rather
        # than None - see its docstring) can never leave the two inconsistent.
        self._speed = (_cap_speed(self._max_speed, _pseudo_fractional(speed))
                       if (speed is not None and distance is not None) else None)

    def stop(self) -> None:
        """
        Freeze speed at 0 without touching distance.

        Use this instead of step() whenever the agent's on-map is malfunction or force stop.
        """
        self._speed = Fraction(0)
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
        speed per step, whether it is starting from a genuine standstill (STOPPED/READY_TO_DEPART -
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
    def distance_after_crossing(pre_offset: Optional[Fraction], pre_speed: Optional[Fraction]) -> Optional[Fraction]:
        """
        Expected distance when this step's cell-boundary crossing completed: pre-step distance plus
        pre-step speed, wrapped into the newly-entered cell. None (off map) in, None out.

        Physical motivation: the train's momentum this step carries it across the boundary and some
        distance into the next cell - the wrap (`% SEGMENT_LENGTH`) is exactly that leftover distance,
        i.e. how far into the new cell the train's momentum actually reaches. This is the only one of
        the four formulas that ever transitions the agent's current_entry_point/next_entry_point into a
        new cell (see RailEnv.step()'s (10a)/(10b)); the granting action can be MOVE_FORWARD/MOVE_LEFT/
        MOVE_RIGHT while genuinely MOVING (pre-step speed > 0), or an explicit STOP_MOVING if the
        crossing was already in flight before the brake takes effect (see speed_after_braking above) -
        the action itself never matters once resource_check has granted the crossing.

        Never returns exactly SEGMENT_LENGTH: a modulo result is always in [0, SEGMENT_LENGTH) by
        construction - physically, having actually crossed into the new cell, the train is now somewhere
        inside it, never still sitting exactly on the boundary it just crossed (that "sitting on the
        boundary" case is distance_without_crossing below, where the crossing does *not* happen).
        """
        if pre_offset is None:
            return None
        return (pre_offset + pre_speed) % SEGMENT_LENGTH

    @staticmethod
    @lru_cache()
    def distance_without_crossing(pre_offset: Optional[Fraction], pre_speed: Optional[Fraction]) -> Optional[Fraction]:
        """
        Expected distance when this step's cell-boundary crossing did not complete (denied by an
        invalid action or resource_check, or parked at/beyond a just-reached target): pre-step distance
        plus pre-step speed, capped at the cell boundary rather than wrapping into a new cell. None (off
        map) in, None out.

        Physical motivation: the train's momentum this step still carries it right up to the cell
        boundary - it just isn't credited with crossing it, so it's parked exactly there instead of
        past it. This is real physical distance actually covered this step (pre_speed is always > 0 in
        every case below - a MOVING agent's pre-step speed can never be 0), not a frozen/no-op value;
        contrast with an agent whose pre-step speed genuinely was 0 this step (STOPPED, or STOPPED/
        MALFUNCTION promoted to MOVING this step), whose distance is asserted unchanged at its pre-step
        value by a completely different invariant branch (see (D1) in
        _check_post_speed_distance_invariants), not by this formula.

        The result equals exactly SEGMENT_LENGTH (never something less) in each of the following cases -
        all three only ever apply once `is_cell_exit()` was already true pre-step (i.e.
        `pre_offset + pre_speed >= SEGMENT_LENGTH`, which is exactly the min()'s other operand):
        - an invalid action denies the crossing attempt at the cell boundary (e.g. going straight
          through a symmetric switch, which only allows turning) - the operator's own mistaken
          instruction, physically a real train braking hard right at the switch rather than derailing;
        - resource_check denies the crossing to another agent (a real motion conflict - the track ahead
          is physically occupied, so the train brakes to a stop just short of it, same as an emergency
          stop signal);
        - the agent reaches its target this step (remove_agents_at_target=False) - parked at the target,
          capped since there is no further cell to wrap into.
        The target-reached case always lands at exactly SEGMENT_LENGTH too, never below: RailEnv.step()'s
        (3b.5) only ever turns a step into a genuine crossing attempt (the candidate entry point becoming
        a *new* cell, which could then turn out to be the target) once `is_cell_exit()` already holds -
        there is no path into this branch with `pre_offset + pre_speed < SEGMENT_LENGTH`. What varies
        between target-reached occurrences is only how much excess (`pre_offset + pre_speed -
        SEGMENT_LENGTH`) gets silently discarded by the cap - anywhere from zero (an exact-fitting
        arrival) up to a full SEGMENT_LENGTH (e.g. a `max_speed` that doesn't evenly divide SEGMENT_LENGTH,
        or an agent restarting from a previously banked-at-boundary position). See
        tests/test_flatland_rail_agent_status.py::test_distance_without_crossing_reaches_segment_length_on_target_single_agent
        and its _banked_restart sibling for worked, env-level examples of each - the effect only shows up
        end-to-end because RailEnv.step()'s (10b) never reaches its ordinary MOVING crossing branch for a
        target-reaching step (agent.state is already DONE there - see (10a)'s update_if_reached(), called
        first), falling through to the DONE-but-not-removed fallback that calls this formula instead.

        This formula is deliberately the *same* for an invalid action and a resource_check denial - both
        are the environment overriding the operator's request against its will, physically identical
        (a real, momentum-carrying train braked to a stop right at the boundary), so both get the same
        consequence. A granted STOP_MOVING never lands here on its own: if resource_check grants a
        genuinely in-flight crossing (is_cell_exit() already true pre-step), the crossing completes via
        distance_after_crossing above regardless of the action - STOP_MOVING cannot itself hold a
        crossing back once resource_check has approved it.

        See rail_env.py's `movement_allowed` design note (loop 2, right before the state machine step)
        for a related but distinct policy choice this formula does *not* itself decide: whether a
        STOPPED/MALFUNCTION agent is even *allowed* to promote to MOVING while its target is still
        occupied at the moment of promotion. The current design grants that promotion optimistically
        regardless (self-loop, no penalty yet) and only pays the price - a resource_check denial via
        this very formula, plus whatever penalty a Rewards implementation attaches to it - once the
        agent's own pre-step speed is genuinely positive and it makes a real attempt. E.g. in a platoon:
        a follower given a movement action while its leader's cell is still occupied is promoted to
        MOVING for free (no penalty this step, since no resource contention is ever checked for a
        self-looping agent); if the leader has vacated that cell by the time the follower's real attempt
        happens, no penalty ever accrues for that promotion at all. Only if the leader is *still* there
        at the moment of the real attempt does the follower get force-stopped back to STOPPED with a
        collision penalty via this formula - i.e. the same promotion event can end up either free or
        penalized, depending purely on timing relative to the leader's own progress, not on whether the
        target looked free at the moment the operator's movement action was given.
        """
        if pre_offset is None:
            return None
        return min(pre_offset + pre_speed, SEGMENT_LENGTH)

    def __repr__(self):
        return f"speed: {self.speed} \
                 max_speed: {self.max_speed} \
                 distance: {self.distance} \
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
        if self._distance is None:
            # design: distance is None when off map
            return True
        return cached_cell_exit(self._max_speed, self._speed, self._distance)

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
