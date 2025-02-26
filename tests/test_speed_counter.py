from flatland.envs.step_utils.speed_counter import SpeedCounter
from flatland.envs.step_utils.states import TrainState


def test_update_counter_speed025():
    sc = SpeedCounter(speed=0.25)
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit == False
    assert sc.distance == 0
    assert sc.speed == 0.25

    sc.step(TrainState.MOVING, (0, 0))
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit == False
    assert sc.distance == 0.25
    assert sc.speed == 0.25

    sc.step(TrainState.MOVING, (0, 0))
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit == False
    assert sc.distance == 0.5
    assert sc.speed == 0.25

    sc.step(TrainState.MOVING, (0, 0))
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit == True
    assert sc.distance == 0.75
    assert sc.speed == 0.25

    sc.step(TrainState.MOVING, (1, 0))
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit == False
    assert sc.distance == 0
    assert sc.speed == 0.25


def test_update_counter_speed05():
    sc = SpeedCounter(speed=0.5)
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit == False
    assert sc.distance == 0
    assert sc.speed == 0.5

    sc.step(TrainState.MOVING, (0, 0))
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit == True
    assert sc.distance == 0.5
    assert sc.speed == 0.5

    sc.step(TrainState.MOVING, (0, 0))
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit == False
    assert sc.distance == 0.0
    assert sc.speed == 0.5
