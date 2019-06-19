from flatland.core.transition_map import GridTransitionMap
from flatland.core.transitions import Grid4Transitions, Grid8Transitions, Grid4TransitionsEnum


def test_grid4_set_transitions():
    grid4_map = GridTransitionMap(2, 2, Grid4Transitions([]))
    grid4_map.set_transition((0, 0), Grid4TransitionsEnum.EAST, 1)
    actual_transitions  = grid4_map.get_transitions((0,0))
    assert False


def test_grid8_set_transitions():
    grid8_map = GridTransitionMap(2, 2, Grid8Transitions([]))
