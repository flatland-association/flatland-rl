from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple

from flatland.envs.rail_env_action import RailEnvActions


@dataclass(repr=True)
class AgentTransitionData:
    """ Class for keeping track of temporary agent data for position update """
    speed: Fraction
    candidate_speed: Fraction
    candidate_entry_point: Tuple[Tuple[int, int], int] = None
    candidate_next_entry_point: Tuple[Tuple[int, int], int] = None
    candidate_distance: Optional[Fraction] = None
    resource_check: bool = False
    # whether this agent was already done *before* this step (pre-step value, like speed above) -
    # RailEnvStateMachineWrapper's post-step update_if_reached() gate needs this after step() returns, when
    # agent.target_entry_point may already reflect a DONE transition that happened *this* step.
    done: bool = False
    # whether this step's action led to a valid transition (only meaningful once the agent is at a cell
    # exit) - depends on this step's pre-step cell_exit/candidate_entry_point_independent (see
    # rail_env.py's collect phase), both already gone by the time anything downstream (rewards.py,
    # RailEnvStateMachineWrapper) could recompute it - unlike in_malfunction/earliest_departure_reached/
    # new_speed_zero/movement_allowed/stop_action_given/movement_action_given, all recomputable from
    # agent/candidate_speed/resource_check/action below. Read by rail_env.py's own movement_allowed calc
    # and by rewards.py.
    action_valid: bool = False
    # this step's action (as given, before any validity check) - stop_action_given (rewards.py) and
    # movement_action_given (RailEnvStateMachineWrapper) are both derived from this rather than stored
    # separately.
    action: RailEnvActions = RailEnvActions.DO_NOTHING
