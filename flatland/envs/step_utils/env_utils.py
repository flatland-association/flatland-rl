from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple

from flatland.envs.step_utils.states import StateTransitionSignals


@dataclass(repr=True)
class AgentTransitionData:
    """ Class for keeping track of temporary agent data for position update """
    speed: Fraction
    candidate_speed: Fraction
    state_transition_signal: StateTransitionSignals
    # design: actions applied at cell entry -- this step's attempted target: `agent.next_entry_point`
    # while attempting entry (`is_cell_exit()`), else self (`agent.current_entry_point`, no resource
    # contested). Distinct from `agent.next_entry_point`, which stays unchanged across retries until
    # an attempt actually succeeds.
    candidate_entry_point: Tuple[Tuple[int, int], int] = None
    # design: actions applied at cell entry -- the one-cell lookahead computed this step from
    # `agent.next_entry_point` (not from `current_entry_point`) using this step's action. Promoted to
    # become the new `agent.next_entry_point` only if `pending_entry_point` is actually entered this
    # step; otherwise discarded and recomputed fresh next call - never needs to survive past this step.
    candidate_next_entry_point: Tuple[Tuple[int, int], int] = None
    # design: actions applied at cell entry -- True iff a crossing was actually attempted this step
    # (is_cell_exit() reached) and denied because this step's action has no valid look-ahead from
    # the target being entered (entry point and next entry point must always advance together - see
    # the invariant in RailEnv.step()). Forces movement_allowed to False in loop 2, driving the state
    # machine to a stop, exactly like an ordinary invalid action would - distinct from simply not
    # attempting a crossing at all (most steps, mid-cell), which must not force a stop.
    crossing_denied: bool = False
    # whether this step's action leads to a valid transition - see RailEnv.step()'s (2) POSITION
    # UPDATE for the exact definition (always True while on-map; off-map, reflects whether the
    # transition from initial_entry_point is valid).
    action_valid: bool = False
    # whether MotionCheck granted this agent's contested resource this step - see RailEnv.step()'s
    # (8) FETCH CONFLICT RESOLUTION FOR AGENT for the exact definition. False while the agent is not
    # attempting a crossing this step (mid-cell).
    motion_check: bool = False
