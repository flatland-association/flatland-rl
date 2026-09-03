# `_candidate_speed` / `_candidate_distance` branch contract

Branch-by-branch comparison of `AbstractRailEnv._candidate_speed()` and `AbstractRailEnv._candidate_distance()`
(`flatland/envs/rail_env.py`). Both methods are pure, pre-step-only derivations of this step's optimistic
candidate speed/distance, called from the collect phase and cross-checked by the post-step invariant checks.
Their branch conditions are written to be self-contained (order-independent) and, for every case both methods
share, textually identical.

## Table 1 — case / formula / candidate value

| Case | Full formula | `_candidate_speed` | `_candidate_distance` |
|---|---|---|---|
| **Done** | `pre_done` | `0` | `distance_without_crossing(offset, speed)` |
| **Target reached** | `target_reached ∧ ¬pre_done` | `0` | `None` if `remove_agents_at_target` else `distance_without_crossing(offset, speed)` |
| **Malfunction** | `in_malfunction ∧ ¬pre_done ∧ ¬target_reached` | `0` | `pre_offset` |
| **Map entry** | `is_off_map ∧ candidate≠None ∧ ¬pre_done ∧ ¬target_reached ∧ ¬in_malfunction` | `cap_speed(max_speed, accel_delta)` | `0` |
| **Stay off map** | `is_off_map ∧ candidate=None ∧ ¬pre_done ∧ ¬target_reached ∧ ¬in_malfunction` | `0` | `None` |
| **Invalid action at cell exit** | `candidate_entry_point_independent_invalid ∧ is_cell_exit ∧ ¬pre_done ∧ ¬target_reached ∧ ¬in_malfunction ∧ ¬is_off_map` | `0` | `distance_without_crossing(offset, speed)` |
| **Stopped** (`pre_speed == 0`) | distance only | `pre_speed` (=`0`) *(via Default, if `DO_NOTHING`)*; `speed_after_acceleration(0, max_speed, accel_delta)` *(via Acceleration/start moving, if a moving action given)*; `speed_after_braking(0, braking_delta)` (=`0`) *(via Braking, if `STOP_MOVING`)* | `pre_offset` |
| **Acceleration or start moving** | speed only | `speed_after_acceleration(pre_speed, max_speed, accel_delta)` | `pre_offset` *(via Stopped, if `pre_speed==0` — the "start moving" sub-case)*; `distance_after_crossing(offset, speed)` *(via Default, if `pre_speed>0` — already moving, `MOVE_FORWARD`)* |
| **Braking** | speed only | `speed_after_braking(pre_speed, braking_delta)` | `pre_offset` *(via Stopped, if `pre_speed==0`)*; `distance_after_crossing(offset, speed)` *(via Default, if `pre_speed>0` — braking doesn't halt an already-in-flight boundary crossing)* |
| **Default** | both: `no_earlier_case_applies` | `pre_speed` | `distance_after_crossing(offset, speed)` |

`Stopped`/`Acceleration or start moving`/`Braking` are not a clean fusion between the two methods: distance
partitions this region by physical state (`pre_speed == 0` or not), speed partitions the same region by the
action given that step. Neither is a subset of the other - the two methods reach the same practical outcomes
via different, non-corresponding branches there.

## Table 2 — term definitions

| Term | Definition | Same in both? |
|---|---|---|
| `pre_done` | `agent.target_entry_point is not None` *(via PreStepSnapshot for the post-step checks)* | ✅ identical |
| `target_reached` | `candidate_entry_point in agent_targets` | ✅ identical |
| `in_malfunction` | `agent.malfunction_handler.in_malfunction` *(via PreStepSnapshot for the post-step checks)* | ✅ identical |
| `is_off_map` | `pre_current_entry_point is None`, where `pre_current_entry_point = agent.current_entry_point` *(via PreStepSnapshot for the post-step checks)* | ✅ identical |
| `candidate_entry_point is / is not None` | return value of `_candidate_entry_points(...)` - recomputed fresh each call, not stored/snapshotted | ✅ identical |
| `candidate_entry_point_independent_invalid` | `candidate_entry_point_independent is None`, where `candidate_entry_point_independent = self.rail.apply_action_independent(RailEnvActions.from_value(action_dict.get(agent.handle, RailEnvActions.DO_NOTHING)), agent.next_entry_point if agent.next_entry_point is not None else agent.initial_entry_point)` *(via PreStepSnapshot, both call paths)* | ✅ identical |
| `is_cell_exit` | `pre_offset is not None and (pre_offset + pre_speed >= SEGMENT_LENGTH)`, where `pre_offset = agent.speed_counter.distance`, `pre_speed = agent.speed_counter.speed` *(both via PreStepSnapshot for the post-step checks)* | ✅ identical |
| `invalid_action_at_cell_exit` | `candidate_entry_point_independent_invalid and is_cell_exit` | ✅ identical |
| `pre_speed == 0` | `agent.speed_counter.speed == 0` *(via PreStepSnapshot for the post-step checks)* | ✅ identical expression |
| `no_earlier_case_applies` | `¬pre_done ∧ ¬target_reached ∧ ¬in_malfunction ∧ ¬is_off_map ∧ ¬invalid_action_at_cell_exit` | ✅ same 5 exclusions |

One open naming question, not yet resolved: `candidate_entry_point is None` (used in the `map entry`/`stay off
map` branches) was considered for a `candidate_entry_point_invalid` rename to match
`candidate_entry_point_independent_invalid`, but rejected - it means "no departure happened this step" (which
also covers "not yet ready to depart" / "no movement action given"), not specifically "the action was invalid".
