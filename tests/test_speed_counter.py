import pickle
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from flatland.envs.step_utils.speed_counter import SpeedCounter, _pseudo_fractional, _cap_speed, \
    cached_cell_exit, _cached_cell_exit


# design: distance update with pre-step speed.
def test_step_counter_speed025():
    sc = SpeedCounter(max_speed=0.25, speed=0.25)
    # design: distance is None when off map
    assert sc.is_cell_entry == False  # off map: no cell entered (yet) at all
    assert sc.is_cell_exit() == True
    assert sc.distance is None
    assert np.isclose(float(sc.speed), 0.25)

    sc.set(sc.speed, Fraction(0))
    assert sc.is_cell_entry == True  # bootstrap onto the map: just entered its first cell
    assert sc.is_cell_exit() == False
    assert sc.distance == 0
    assert np.isclose(float(sc.speed), 0.25)

    sc.set(sc.speed, Fraction(1, 4))
    assert sc.is_cell_entry == False  # mid-cell advance (0 -> 0.25): still the same cell
    assert sc.is_cell_exit() == False
    assert sc.distance == 0.25
    assert np.isclose(float(sc.speed), 0.25)

    sc.set(sc.speed, Fraction(1, 2))
    assert sc.is_cell_entry == False  # mid-cell advance (0.25 -> 0.5): still the same cell
    assert sc.is_cell_exit() == False
    assert sc.distance == 0.5
    assert np.isclose(float(sc.speed), 0.25)

    sc.set(sc.speed, Fraction(3, 4))
    assert sc.is_cell_entry == False  # mid-cell advance (0.5 -> 0.75): still the same cell, now at is_cell_exit()
    assert sc.is_cell_exit() == True
    assert sc.distance == 0.75
    assert np.isclose(float(sc.speed), 0.25)

    sc.set(sc.speed, Fraction(0))
    assert sc.is_cell_entry == True  # wraps into the next cell (0.75 -> 0): genuine crossing
    assert sc.is_cell_exit() == False
    assert sc.distance == 0
    assert np.isclose(float(sc.speed), 0.25)


# design: distance update with pre-step speed.
def test_step_counter_speed05():
    sc = SpeedCounter(max_speed=0.5, speed=0.5)
    # design: distance is None when off map
    assert sc.is_cell_entry == False  # off map: no cell entered (yet) at all
    assert sc.is_cell_exit() == True
    assert sc.distance is None
    assert np.isclose(float(sc.speed), 0.5)

    sc.set(sc.speed, Fraction(0))
    assert sc.is_cell_entry == True  # bootstrap onto the map: just entered its first cell
    assert sc.is_cell_exit() == False
    assert sc.distance == 0
    assert np.isclose(float(sc.speed), 0.5)

    sc.set(sc.speed, Fraction(1, 2))
    assert sc.is_cell_entry == False  # mid-cell advance (0 -> 0.5): still the same cell, now at is_cell_exit()
    assert sc.is_cell_exit() == True
    assert sc.distance == 0.5
    assert np.isclose(float(sc.speed), 0.5)

    sc.set(sc.speed, Fraction(0))
    assert sc.is_cell_entry == True  # wraps into the next cell (0.5 -> 0): genuine crossing
    assert sc.is_cell_exit() == False
    assert sc.distance == 0.0
    assert np.isclose(float(sc.speed), 0.5)


# design: distance update with pre-step speed.
def test_step_counter_speed025_05():
    sc = SpeedCounter(speed=0.25, max_speed=1.0)
    # design: distance is None when off map
    assert sc.is_cell_entry == False  # off map: no cell entered (yet) at all
    assert sc.is_cell_exit() == True
    assert sc.distance is None
    assert np.isclose(float(sc.speed), 0.25)

    sc.set(sc.speed, Fraction(0))
    assert sc.is_cell_entry == True  # bootstrap onto the map: just entered its first cell
    assert sc.is_cell_exit() == False
    assert sc.distance == 0
    assert np.isclose(float(sc.speed), 0.25)

    sc.set(sc.speed, Fraction(1, 4))
    assert sc.is_cell_entry == False  # mid-cell advance (0 -> 0.25): still the same cell
    assert sc.is_cell_exit() == False
    assert sc.distance == 0.25
    assert np.isclose(float(sc.speed), 0.25)

    sc.set(0.5, Fraction(1, 2))
    assert sc.is_cell_entry == False  # mid-cell advance with a new (higher) speed (0.25 -> 0.5): still the same cell
    assert sc.is_cell_exit() == True
    assert sc.distance == 0.5
    assert np.isclose(float(sc.speed), 0.5)

    sc.set(sc.speed, Fraction(0))
    assert sc.is_cell_entry == True  # wraps into the next cell (0.5 -> 0): genuine crossing
    assert sc.is_cell_exit() == False
    assert sc.distance == 0
    assert np.isclose(float(sc.speed), 0.5)

    sc.set(0.25, Fraction(1, 2))
    assert sc.is_cell_entry == False  # mid-cell advance with a new (lower) speed (0 -> 0.5): still the same cell
    assert sc.is_cell_exit() == False
    assert sc.distance == 0.5
    assert np.isclose(float(sc.speed), 0.25)


# design: distance update with pre-step speed.
def test_step_counter_speed025_03():
    sc = SpeedCounter(speed=0.25, max_speed=0.3)
    # design: distance is None when off map
    assert sc.is_cell_entry == False  # off map: no cell entered (yet) at all
    assert sc.is_cell_exit() == True
    assert sc.distance is None
    assert np.isclose(float(sc.speed), 0.25)

    sc.set(sc.speed, Fraction(0))
    assert sc.is_cell_entry == True  # bootstrap onto the map: just entered its first cell
    assert sc.is_cell_exit() == False
    assert sc.distance == 0
    assert np.isclose(float(sc.speed), 0.25)

    sc.set(sc.speed, Fraction(1, 4))
    assert sc.is_cell_entry == False  # mid-cell advance (0 -> 0.25): still the same cell
    assert sc.is_cell_exit() == False
    assert sc.distance == 0.25
    assert np.isclose(float(sc.speed), 0.25)

    sc.set(Fraction(1, 2), Fraction(1, 2))
    assert sc.is_cell_entry == False  # mid-cell advance with a new (capped-to-max_speed) speed (0.25 -> 0.5): still the same cell
    assert sc.is_cell_exit() == False
    assert np.isclose(float(sc.distance), 0.5)
    assert np.isclose(float(sc.speed), 0.3)

    sc.set(sc.speed, Fraction(4, 5))
    assert sc.is_cell_entry == False  # mid-cell advance (0.5 -> 0.8): still the same cell, now at is_cell_exit()
    assert sc.is_cell_exit() == True
    assert np.isclose(float(sc.distance), 0.8)
    assert np.isclose(float(sc.speed), 0.3)

    sc.set(sc.speed, Fraction(1, 10))
    assert sc.is_cell_entry == True  # wraps into the next cell (0.8 -> 0.1): genuine crossing
    assert sc.is_cell_exit() == False
    assert np.isclose(float(sc.distance), 0.1)
    assert np.isclose(float(sc.speed), 0.3)

    sc.set(-0.5, Fraction(2, 5))
    # invalidate cell_entry despite speed 0
    assert sc.is_cell_entry == False  # braking to a stop mid-cell (0.1 -> 0.4, capped at speed=0): still the same cell
    assert sc.is_cell_exit() == False
    assert np.isclose(float(sc.distance), 0.4)
    assert np.isclose(float(sc.speed), 0.0)


def test_clone_speed_counter_speed1():
    """Test that a SpeedCounter stays consistent when restored from a pickled state."""
    sc = SpeedCounter(speed=1, max_speed=1)
    assert pickle.loads(pickle.dumps(sc)) == sc


def test_clone_speed_counter_fractional_speed():
    """Test that a SpeedCounter stays consistent when restored from a pickled state."""
    sc = SpeedCounter(speed=1 / 5, max_speed=1 / 3)
    assert pickle.loads(pickle.dumps(sc)) == sc
    sc.set(sc.speed, Fraction(0))
    # design: distance is None when off map
    assert sc.is_cell_entry  # bootstrap onto the map: just entered its first cell
    assert sc.distance == 0
    assert pickle.loads(pickle.dumps(sc)) == sc
    sc.set(1 / 10, Fraction(1, 5))
    assert not sc.is_cell_entry  # mid-cell advance (0 -> 0.2): still the same cell
    # design: distance update with pre-step speed.
    assert np.isclose(float(sc.distance), 0.2)
    assert pickle.loads(pickle.dumps(sc)) == sc


def test_no_fractional():
    # Step 1: 0.59999999999999997780
    # Step 2: 0.19999999999999995559
    # Step 3: 0.79999999999999993339
    # Step 4: 0.39999999999999991118
    # Step 5: 0.99999999999999988898
    d = 0.0
    s = 0.6
    for i in range(1, 6):
        d = (d + s) % 1.0
        print(f'Step {i}: {d:.20f}')
    assert d != 1.0
    assert np.isclose(d, 1.0)


def test_pseudo_fractional():
    # Step 1: 0.60000000000000000000
    # Step 2: 0.20000000000000000000
    # Step 3: 0.80000000000000000000
    # Step 4: 0.40000000000000000000
    # Step 5: 1.00000000000000000000
    d = Fraction(0, 1)
    s = Fraction(6, 10)
    for i in range(1, 6):
        d = (d + s)
        while d > 1:
            d -= Fraction(1, 1)
        try:
            print(f'Step {i}: {d:.20f}')
        except TypeError:
            # support for Python < 3.12, float-style-formatting introduced only in 3.12 https://docs.python.org/3.12/library/fractions.html
            print(f'Step {i}: {d}')
    assert d == 1.0


def test__pseudo_fractional():
    assert _pseudo_fractional(0) == Fraction(0)
    assert _pseudo_fractional(0.0) == Fraction(0.0)
    assert _pseudo_fractional(1) == Fraction(1)
    assert _pseudo_fractional(1.0) == Fraction(1)
    assert _pseudo_fractional(1 / 3) == Fraction(1, 3)
    assert _pseudo_fractional(0.33) == Fraction(1, 3)
    assert _pseudo_fractional(0.333) == Fraction(1, 3)
    assert _pseudo_fractional(4 / 3) == Fraction(4, 3)
    assert _pseudo_fractional(0.55) == Fraction(55, 100)
    assert _pseudo_fractional(-1 / 3) == - Fraction(1, 3)
    assert _pseudo_fractional(-0.33) == - Fraction(1, 3)
    assert _pseudo_fractional(-0.55) == - Fraction(55, 100)


def test__pseudo_fractional_none():
    assert _pseudo_fractional(None) is None


def test__pseudo_fractional_fraction_passthrough():
    assert _pseudo_fractional(Fraction(2, 3)) == Fraction(2, 3)


def test__pseudo_fractional_decimal():
    # N.B. _pseudo_fractional is @lru_cache()'d, and Decimal("0.25") == 0.25 (cross-type numeric
    # equality/hash) - without clearing the cache first, an earlier test's _pseudo_fractional(0.25) call
    # would make this a cache HIT, silently skipping the isinstance(v, Decimal) branch entirely.
    _pseudo_fractional.cache_clear()
    assert _pseudo_fractional(Decimal("0.25")) == Fraction(1, 4)


def test__pseudo_fractional_invalid_type_raises():
    with pytest.raises(ValueError):
        _pseudo_fractional("not a number")


# N.B. the final `else: Fraction.from_float(v)` branch of `_pseudo_fractional` could not be reached by any
# finite float in a 200k-sample brute-force search: `str(v)` always round-trips exactly back to `v` via
# `Decimal`, so the preceding `np.isclose(float(Decimal(str(v))), v)` branch is a tautology for every finite,
# non-NaN float - the final `else` appears to be unreachable dead code, not tested here for that reason.


def test_cached_cap_speed_clamps_negative_to_zero():
    assert _cap_speed(Fraction(1, 2), Fraction(-1, 4)) == Fraction(0)


def test_cached_cap_speed_clamps_above_max_speed():
    assert _cap_speed(Fraction(1, 2), Fraction(3, 4)) == Fraction(1, 2)


def test_cached_cap_speed_passthrough_within_range():
    assert _cap_speed(Fraction(1, 2), Fraction(1, 4)) == Fraction(1, 4)


def test_cached_cell_exit_true():
    assert _cached_cell_exit(Fraction(3, 4), Fraction(1, 2)) == True


def test_cached_cell_exit_false():
    assert _cached_cell_exit(Fraction(1, 4), Fraction(1, 4)) == False


def test_cached_cell_exit_caps_speed_at_max_speed():
    # naive (uncapped) distance(0.5) + speed(0.9) = 1.4 >= 1 -> would be True, but max_speed=0.3 caps the
    # effective speed to 0.3 first: 0.5 + 0.3 = 0.8 < 1 -> False.
    assert cached_cell_exit(Fraction(3, 10), Fraction(9, 10), Fraction(1, 2)) == False


def test_step_crossing_not_completed_caps_at_boundary():
    """A MOVING agent whose transition into the next cell is blocked by a resource conflict this step:
    distance must be capped at the cell boundary, not wrapped into the next cell as if it had moved."""
    sc = SpeedCounter(max_speed=0.5, speed=0.5)
    sc.set(sc.speed, Fraction(0))  # design: distance is None when off map
    sc.set(sc.speed, Fraction(1, 2))  # distance -> 1/2 (from the pre-step speed 1/2); speed stays 1/2 for the next call
    sc.set(speed=0.5, distance=Fraction(1, 1))
    assert sc.distance == Fraction(1, 1)
    assert not sc.is_cell_entry  # denied at the boundary (1/2 -> 1, capped): still the same cell, not the next one
    assert sc.speed == Fraction(1, 2)


def test_step_crossing_not_completed_under_boundary_behaves_like_normal_step():
    sc = SpeedCounter(max_speed=0.25, speed=0.25)
    sc.set(sc.speed, Fraction(0))  # design: distance is None when off map
    sc.set(speed=0.25, distance=Fraction(1, 4))
    assert sc.distance == Fraction(1, 4)
    assert not sc.is_cell_entry  # mid-cell advance (0 -> 1/4), nowhere near the boundary: still the same cell


def test_stop_freezes_speed_without_touching_distance():
    """design: distance update with pre-step speed - stop() must leave already-accumulated in-cell
    distance untouched (e.g. a malfunction interrupting a MOVING agent mid-cell)."""
    sc = SpeedCounter(max_speed=0.5, speed=0.5)
    sc.set(sc.speed, Fraction(0))  # design: distance is None when off map
    sc.set(sc.speed, Fraction(1, 2))  # distance -> 1/2
    sc.stop()
    assert sc.speed == Fraction(0)
    assert sc.distance == Fraction(1, 2)
    assert not sc.is_cell_entry  # force-stopped mid-cell, position frozen in place: still the same cell


def test_set_is_cell_entry_true_for_genuine_crossing_from_banked_boundary():
    """A MOVING agent bootstraps at speed=1 onto the map, is denied/braked to a stop exactly at the next
    cell boundary (banking distance == SEGMENT_LENGTH), is later promoted back to MOVING while still
    banked (distance unchanged), then genuinely completes its crossing into the next cell. is_cell_entry
    correctly reports True for that last step - set() derives it from old vs. new distance, not from
    new_distance < speed (the old step()'s formula, which wrongly reported False at exactly this boundary
    - see Phase 0's xfail test this replaces)."""
    sc = SpeedCounter(max_speed=1.0, speed=1.0)
    sc.set(sc.speed, Fraction(0))  # bootstrap onto the map: distance -> 0, speed -> 1
    sc.set(speed=0, distance=Fraction(1))  # denied/braked exactly at the boundary: distance -> 1 (banked), speed -> 0
    assert sc.distance == Fraction(1)
    sc.set(speed=0.5, distance=Fraction(1))  # promoted back to MOVING while still banked: distance stays 1, speed -> 1/2
    assert sc.distance == Fraction(1)
    sc.set(speed=0.5, distance=Fraction(1, 2))  # genuine crossing completes from the banked boundary
    assert sc.distance == Fraction(1, 2)
    assert sc.is_cell_entry  # the train's cell really did just change


def test_reset_clears_distance_speed_and_cell_entry():
    sc = SpeedCounter(speed=0.5, max_speed=1.0)
    sc.set(sc.speed, Fraction(0))  # design: distance is None when off map
    sc.set(sc.speed, Fraction(1, 2))
    assert sc.distance != 0
    assert not sc.is_cell_entry  # mid-cell advance: still the same cell
    sc.reset()
    # design: speed and distance are None when off map - reset() puts the counter back off map
    assert sc.distance is None
    assert sc.speed is None
    assert not sc.is_cell_entry  # back off map: no cell entered
    # max_speed is untouched by reset()
    assert sc.max_speed == Fraction(1)


def test_repr_contains_state():
    sc = SpeedCounter(speed=0.5, max_speed=1.0)
    r = repr(sc)
    assert "speed: 1/2" in r
    # design: distance is None when off map
    assert "distance: None" in r
    assert "is_cell_entry: False" in r  # off map: no cell entered


def test_eq_against_non_speed_counter_returns_false():
    sc = SpeedCounter(max_speed=0.5, speed=0.5)
    assert sc != "not a speed counter"
    assert sc.__eq__(5) is False


def test_eq_between_speed_counters_with_different_state():
    assert SpeedCounter(max_speed=0.5, speed=0.5) != SpeedCounter(max_speed=0.25, speed=0.25)

    sc1 = SpeedCounter(max_speed=0.5, speed=0.5)
    sc2 = SpeedCounter(max_speed=0.5, speed=0.5)
    sc2.set(sc2.speed, Fraction(0))
    assert sc1 != sc2  # differs in distance now


def test_eq_between_independently_constructed_identical_speed_counters():
    assert SpeedCounter(speed=0.5, max_speed=1.0) == SpeedCounter(speed=0.5, max_speed=1.0)


def test_init_rejects_max_speed_above_one():
    with pytest.raises(AssertionError):
        SpeedCounter(max_speed=1.5)


def test_init_rejects_speed_above_max_speed():
    with pytest.raises(AssertionError):
        SpeedCounter(speed=0.5, max_speed=0.3)


def test_init_rejects_negative_speed():
    with pytest.raises(AssertionError):
        SpeedCounter(speed=-0.1, max_speed=1.0)


def test_setstate_backward_compat_underscore_speed_key():
    """Very old pickles stored the speed under `_speed` rather than `speed`, and had no `max_speed` key
    at all (constant-speed agents, so max_speed defaults to speed)."""
    sc = SpeedCounter.__new__(SpeedCounter)
    sc.__setstate__({"_speed": 0.5, "distance": 0.25, "is_cell_entry": False})
    assert sc.speed == Fraction(1, 2)
    assert sc.max_speed == Fraction(1, 2)
    assert sc.distance == Fraction(1, 4)
    assert not sc.is_cell_entry  # persisted state says: mid-cell, not a fresh cell entry - carried through as-is


def test_setstate_backward_compat_counter_key():
    """Even older pickles stored progress as an integer `counter` (cells at constant speed) instead of a
    `distance` fraction: distance = counter * speed, is_cell_entry = counter == 0."""
    sc = SpeedCounter.__new__(SpeedCounter)
    sc.__setstate__({"speed": Fraction(1, 4), "counter": 2})
    assert sc.speed == Fraction(1, 4)
    assert sc.max_speed == Fraction(1, 4)
    assert sc.distance == Fraction(1, 2)
    assert not sc.is_cell_entry  # persisted state says: counter=2 cells already traveled, not a fresh entry


def test_setstate_backward_compat_counter_key_zero():
    sc = SpeedCounter.__new__(SpeedCounter)
    sc.__setstate__({"speed": Fraction(1, 4), "counter": 0})
    assert sc.distance == Fraction(0)
    assert sc.is_cell_entry  # persisted state says: counter=0, no cells traveled yet - a fresh entry


def test_setstate_missing_is_cell_entry_and_counter_leaves_it_unset():
    """KNOWN BUG: a state dict with neither "counter" nor "is_cell_entry" never assigns `_is_cell_entry` at
    all, so accessing `.is_cell_entry` afterward raises AttributeError instead of falling back to a
    default. This documents the current (buggy) behaviour rather than a desired one."""
    sc = SpeedCounter.__new__(SpeedCounter)
    sc.__setstate__({"speed": Fraction(1, 2), "distance": Fraction(1, 4)})
    with pytest.raises(AttributeError):
        _ = sc.is_cell_entry
