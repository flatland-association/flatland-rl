import pickle
from fractions import Fraction

import numpy as np

from flatland.envs.step_utils.speed_counter import SpeedCounter, _pseudo_fractional


def test_step_counter_speed025():
    sc = SpeedCounter(speed=0.25)
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0
    assert np.isclose(float(sc.speed), 0.25)

    sc.step()
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0.25
    assert np.isclose(float(sc.speed), 0.25)

    sc.step()
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0.5
    assert np.isclose(float(sc.speed), 0.25)

    sc.step()
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.25) == True
    assert sc.distance == 0.75
    assert np.isclose(float(sc.speed), 0.25)

    sc.step()
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0
    assert np.isclose(float(sc.speed), 0.25)


def test_step_counter_speed05():
    sc = SpeedCounter(speed=0.5)
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0.5) == False
    assert sc.distance == 0
    assert np.isclose(float(sc.speed), 0.5)

    sc.step()
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.5) == True
    assert sc.distance == 0.5
    assert np.isclose(float(sc.speed), 0.5)

    sc.step()
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0.5) == False
    assert sc.distance == 0.0
    assert np.isclose(float(sc.speed), 0.5)


def test_step_counter_speed025_05():
    sc = SpeedCounter(speed=0.25, max_speed=1.0)
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0
    assert np.isclose(float(sc.speed), 0.25)

    sc.step()
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0.25
    assert np.isclose(float(sc.speed), 0.25)

    sc.step(speed=0.5)
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.5) == True
    assert sc.distance == 0.75
    assert np.isclose(float(sc.speed), 0.5)

    sc.step()
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0.5) == False
    assert sc.distance == 0.25
    assert np.isclose(float(sc.speed), 0.5)

    sc.step(speed=0.25)
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0.5
    assert np.isclose(float(sc.speed), 0.25)


def test_step_counter_speed025_03():
    sc = SpeedCounter(speed=0.25, max_speed=0.3)
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0
    assert np.isclose(float(sc.speed), 0.25)

    sc.step()
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0.25
    assert np.isclose(float(sc.speed), 0.25)

    sc.step(speed=Fraction(1, 2))
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.3) == False
    assert np.isclose(float(sc.distance), 0.55)
    assert np.isclose(float(sc.speed), 0.3)

    sc.step()
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.3) == True
    assert np.isclose(float(sc.distance), 0.85)
    assert np.isclose(float(sc.speed), 0.3)

    sc.step()
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0.3) == False
    assert np.isclose(float(sc.distance), 0.15)
    assert np.isclose(float(sc.speed), 0.3)

    sc.step(speed=-0.5)
    # invalidate cell_entry despite speed 0
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0) == False
    assert np.isclose(float(sc.distance), 0.15)
    assert np.isclose(float(sc.speed), 0.0)


def test_clone_speed_counter_speed1():
    """Test that a SpeedCounter stays consistent when restored from a pickled state."""
    sc = SpeedCounter(speed=1, max_speed=1)
    assert pickle.loads(pickle.dumps(sc)) == sc


def test_clone_speed_counter_fractional_speed():
    """Test that a SpeedCounter stays consistent when restored from a pickled state."""
    sc = SpeedCounter(speed=1 / 5, max_speed=1 / 3)
    assert pickle.loads(pickle.dumps(sc)) == sc
    sc.step(speed=1 / 10)
    assert not sc.is_cell_entry
    assert np.isclose(float(sc.distance), 0.1)
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
        print(f'Step {i}: {d:.20f}')
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
