from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple

from flatland.envs.step_utils.states import StateTransitionSignals


@dataclass(repr=True)
class AgentTransitionData:
    """ Class for keeping track of temporary agent data for position update """
    speed: Fraction
    candidate_speed: Fraction
    state_transition_signal: StateTransitionSignals
    candidate_entry_point: Tuple[Tuple[int, int], int] = None
    candidate_next_entry_point: Tuple[Tuple[int, int], int] = None
    candidate_distance: Optional[Fraction] = None
    resource_check: bool = False
    # duplicates state_machine.update_if_reached()'s own "entry_point in targets" check outside the state
    # machine, so rail_env.py's step() control flow (handle_done_state) doesn't need to read agent.state.
    just_reached_target: bool = False
