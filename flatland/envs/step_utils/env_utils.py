from dataclasses import dataclass
from typing import Tuple

from flatland.core.grid.grid4 import Grid4Transitions
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import StateTransitionSignals


@dataclass(repr=True)
class AgentTransitionData:
    """ Class for keeping track of temporary agent data for position update """
    speed: float
    new_position: Tuple[int, int]
    new_direction: Grid4Transitions
    new_speed: float
    new_position_level_free: float
    preprocessed_action: RailEnvActions
    agent_position_level_free: Tuple[int, int]
    state_transition_signal: StateTransitionSignals
