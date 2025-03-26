from functools import lru_cache
from typing import Any, Tuple

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.utils.decorators import enable_infrastructure_lru_cache


@lru_cache()
def process_illegal_action(action: Any) -> RailEnvActions:
    """
    Returns the action if valid (either int value or in RailEnvActions), returns RailEnvActions.DO_NOTHING otherwise.
    """
    if not RailEnvActions.is_action_valid(action):
        return RailEnvActions.DO_NOTHING
    else:
        return RailEnvActions(action)


@enable_infrastructure_lru_cache()
def preprocess_left_right_action(action: RailEnvActions, rail: RailGridTransitionMap, position: Tuple[int, int], direction: Grid4TransitionsEnum):
    """
    LEFT/RIGHT is converted to FORWARD if left/right is not available.
    """
    if action in [RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT] and not rail.check_valid_action(action, position, direction):
        # TODO revise design: this may accelerate!
        action = RailEnvActions.MOVE_FORWARD
    return action
