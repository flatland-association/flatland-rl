from dataclasses import dataclass
from typing import Tuple

from flatland.envs.step_utils.states import StateTransitionSignals


@dataclass(repr=True)
class AgentTransitionData:
    """ Class for keeping track of temporary agent data for position update """
    speed: float
    new_configuration: Tuple[Tuple[int, int], int]
    new_speed: float
    current_resource: Tuple[int, int]
    state_transition_signal: StateTransitionSignals
