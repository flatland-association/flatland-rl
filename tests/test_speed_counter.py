import numpy as np

from flatland.envs.step_utils.speed_counter import SpeedCounter
from flatland.envs.step_utils.states import TrainState


def test_step_counter_speed025():
    sc = SpeedCounter(speed=0.25)
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0
    assert sc.speed == 0.25

    sc.step(TrainState.MOVING, (0, 0))
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0.25
    assert sc.speed == 0.25

    sc.step(TrainState.MOVING, (0, 0))
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0.5
    assert sc.speed == 0.25

    sc.step(TrainState.MOVING, (0, 0))
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.25) == True
    assert sc.distance == 0.75
    assert sc.speed == 0.25

    sc.step(TrainState.MOVING, (1, 0))
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

    sc.step(TrainState.MOVING, (0, 0))
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.5) == True
    assert sc.distance == 0.5
    assert sc.speed == 0.5

    sc.step(TrainState.MOVING, (0, 0))
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

    sc.step(TrainState.MOVING, (0, 0))
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0.25
    assert sc.speed == 0.25

    sc.step(TrainState.MOVING, (0, 0), speed=0.5)
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.5) == True
    assert sc.distance == 0.75
    assert sc.speed == 0.5

    sc.step(TrainState.MOVING, (0, 0))
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0.5) == False
    assert sc.distance == 0.25
    assert sc.speed == 0.5

    sc.step(TrainState.MOVING, (0, 0), speed=0.25)
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

    sc.step(TrainState.MOVING, (0, 0))
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.25) == False
    assert sc.distance == 0.25
    assert sc.speed == 0.25

    sc.step(TrainState.MOVING, (0, 0), speed=0.5)
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.3) == False
    assert sc.distance == 0.55
    assert sc.speed == 0.3

    sc.step(TrainState.MOVING, (0, 0))
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit(0.3) == True
    assert np.isclose(sc.distance, 0.85)
    assert sc.speed == 0.3

    sc.step(TrainState.MOVING, (0, 0))
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0.3) == False
    assert np.isclose(sc.distance, 0.15)
    assert sc.speed == 0.3

    sc.step(TrainState.MOVING, (0, 0), speed=-0.5)
    # !!! must
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit(0) == False
    assert np.isclose(sc.distance, 0.15)
    assert sc.speed == 0.0
