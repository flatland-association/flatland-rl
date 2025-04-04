import pickle

import numpy as np

from flatland.envs.step_utils.speed_counter import SpeedCounter


def test_step_counter_speed025():
    sc = SpeedCounter(speed=0.25)
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0
    assert sc.speed == 0.25

    sc.step()
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0.25
    assert sc.speed == 0.25

    sc.step()
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0.5
    assert sc.speed == 0.25

    sc.step()
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.25) == True
    assert sc.distance == 0.75
    assert sc.speed == 0.25

    sc.step()
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0
    assert sc.speed == 0.25


def test_step_counter_speed05():
    sc = SpeedCounter(speed=0.5)
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0.5) == False
    assert sc.distance == 0
    assert sc.speed == 0.5

    sc.step()
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.5) == True
    assert sc.distance == 0.5
    assert sc.speed == 0.5

    sc.step()
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0.5) == False
    assert sc.distance == 0.0
    assert sc.speed == 0.5


def test_step_counter_speed025_05():
    sc = SpeedCounter(speed=0.25, max_speed=1.0)
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0
    assert sc.speed == 0.25

    sc.step()
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0.25
    assert sc.speed == 0.25

    sc.step(speed=0.5)
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.5) == True
    assert sc.distance == 0.75
    assert sc.speed == 0.5

    sc.step()
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0.5) == False
    assert sc.distance == 0.25
    assert sc.speed == 0.5

    sc.step(speed=0.25)
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0.5
    assert sc.speed == 0.25


def test_step_counter_speed025_03():
    sc = SpeedCounter(speed=0.25, max_speed=0.3)
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0
    assert sc.speed == 0.25

    sc.step()
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0.25
    assert sc.speed == 0.25

    sc.step(speed=0.5)
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.3) == False
    assert sc.distance == 0.55
    assert sc.speed == 0.3

    sc.step()
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.3) == True
    assert np.isclose(sc.distance, 0.85)
    assert sc.speed == 0.3

    sc.step()
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0.3) == False
    assert np.isclose(sc.distance, 0.15)
    assert sc.speed == 0.3

    sc.step(speed=-0.5)
    # invalidate cell_entry despite speed 0
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0) == False
    assert np.isclose(sc.distance, 0.15)
    assert sc.speed == 0.0

def test_clone_speed_counter_speed1():
    """Test that a SpeedCounter stays consistent when restored from a pickled state."""
    sc = SpeedCounter(speed=1, max_speed=1)
    assert pickle.loads(pickle.dumps(sc)) == sc

def test_clone_speed_counter_fractional_speed():
    """Test that a SpeedCounter stays consistent when restored from a pickled state."""
    sc = SpeedCounter(speed=1/5, max_speed=1/3)
    assert pickle.loads(pickle.dumps(sc)) == sc
    sc.step(speed=1/10)
    assert not sc.is_cell_entry
    assert sc.distance == 0.1
    assert pickle.loads(pickle.dumps(sc)) == sc
