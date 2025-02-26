from flatland.envs.step_utils.speed_counter import SpeedCounter
from flatland.envs.step_utils.states import TrainState


def test_update_counter_speed025():
    sc = SpeedCounter(speed=0.25)

    assert sc.is_cell_entry == True
    assert sc.is_cell_exit == False
    assert sc.counter == 0
    assert sc.max_count == 4

    sc.update_counter(TrainState.MOVING, (0, 0))
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit == False
    assert sc.counter == 1
    assert sc.max_count == 4

    sc.update_counter(TrainState.MOVING, (0, 0))
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit == False
    assert sc.counter == 2
    assert sc.max_count == 4

    sc.update_counter(TrainState.MOVING, (0, 0))
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit == True
    assert sc.counter == 3
    assert sc.max_count == 4

    sc.update_counter(TrainState.MOVING, (1, 0))
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit == False
    assert sc.counter == 0
    assert sc.max_count == 4


def test_update_counter_speed05():
    sc = SpeedCounter(speed=0.5)
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit == False
    assert sc.counter == 0
    assert sc.max_count == 2

    sc.update_counter(TrainState.MOVING, (0, 0))
    assert sc.is_cell_entry == False
    assert sc.is_cell_exit == True
    assert sc.counter == 1
    assert sc.max_count == 2

    sc.update_counter(TrainState.MOVING, (0, 0))
    assert sc.is_cell_entry == True
    assert sc.is_cell_exit == False
    assert sc.counter == 0
    assert sc.max_count == 2
