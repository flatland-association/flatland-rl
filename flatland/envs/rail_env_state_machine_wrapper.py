from typing import Dict

from flatland.envs.rail_env import AbstractRailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import StateTransitionSignals, TrainState


def RailEnvStateMachineWrapper(env: AbstractRailEnv, skip_state_machine_update: bool = False) -> AbstractRailEnv:
    """
    Patches `env`'s class in place so its `step()` also updates `agent.state`/`agent.state_machine`.

    `AbstractRailEnv.step()` never reads/writes `agent.state`/`agent.state_machine` itself (position/
    speed/reward/done are unaffected either way, wrapped or not) - it ends with
    `return self._get_observations(), self.rewards_dict, self.dones, self.get_info_dict()`, and an
    obs_builder/predictor that does read `agent.state` (e.g. `TreeObsForRailEnv`,
    `ShortestPathPredictorForRailEnv`) needs it to already reflect *this* step's position by the time
    `_get_observations()` runs, not the previous step's. `_StateMachineUpdateMixin.step()` below achieves
    this without `AbstractRailEnv` needing any dedicated extension point: it temporarily replaces the
    instance's own `_get_observations` with a one-shot wrapper before delegating to `super().step()` -
    the one-shot wrapper restores the original `_get_observations`, runs the state machine update, then
    calls the (now restored) real `_get_observations()` - so by the time `AbstractRailEnv.step()`'s own
    return statement evaluates `self._get_observations()` (left of `self.get_info_dict()` in that same
    tuple expression, hence evaluated first), `agent.state` is already up to date for both. Without
    wrapping, `agent.state`/`agent.state_machine` are never touched at all - an obs_builder/predictor
    that depends on `agent.state` for correctness (not just `get_info_dict()`'s purely-informational
    `state`/`action_required` fields) needs the env wrapped to behave correctly.

    Idempotent: wrapping an already-wrapped env just updates `skip_state_machine_update` in place
    rather than double-wrapping. Returns the same instance (not a copy), for chaining convenience.

    Parameters
    ----------
    env : AbstractRailEnv
        The env to patch in place.
    skip_state_machine_update : bool
        Wrap but never actually update the state machine - lets a caller keep a uniform
        "always wrapped" call site while still toggling the behavior off.
    """
    base_cls = type(env)
    if not issubclass(base_cls, _StateMachineUpdateMixin):
        base_cls = type(f"{base_cls.__name__}WithStateMachine", (_StateMachineUpdateMixin, base_cls), {})
        env.__class__ = base_cls
    env.skip_state_machine_update = skip_state_machine_update
    return env


class _StateMachineUpdateMixin:
    """ Only ever applied to an instance via `RailEnvStateMachineWrapper` - never instantiated/subclassed directly. """

    def step(self, action_dict: Dict[int, RailEnvActions]):
        if not self.skip_state_machine_update:
            # One-shot: AbstractRailEnv.step() calls self._get_observations() exactly once, in its own
            # final `return self._get_observations(), ..., self.get_info_dict()` - evaluated left to
            # right, so before self.get_info_dict(). Restoring the real _get_observations before running
            # the update (rather than after) means _update_state_machine_after_step() itself is free to
            # call self._get_observations() without recursing into this wrapper.
            def _get_observations_after_state_machine_update():
                del self._get_observations
                self._update_state_machine_after_step()
                return self._get_observations()

            self._get_observations = _get_observations_after_state_machine_update
        return super().step(action_dict)

    def _update_state_machine_after_step(self):
        """
        All `agent.state`/`agent.state_machine` bookkeeping for the step that just ran, in one pass -
        called from within `AbstractRailEnv.step()` itself (via the one-shot `_get_observations` swap in
        `step()` above) once this step's position/speed/reward/done are finalized but before
        observations/info are built - see `TrainStateMachine`
        (`flatland/envs/step_utils/state_machine.py`): every transition method,
        `update_if_reached()`, and `state_position_sync_check()` read only their explicit arguments/
        `self.st_signals`, never another agent nor anything `step()` mutates later - so running all
        of it in a single pass here (rather than interleaved per-agent inside `step()`'s own per-agent
        loop, as it used to be) is behaviorally equivalent.

        The `StateTransitionSignals` fed into the state machine below are reconstructed here rather than
        carried directly: `stop_action_given`/`movement_action_given` are pure functions of
        `AgentTransitionData.action`; `in_malfunction` is a live agent attribute;
        `earliest_departure_reached`/`new_speed_zero`/`movement_allowed` are cheap recomputations from
        data `AgentTransitionData` already carries for other reasons. Only `action_valid` genuinely can't
        be reconstructed after the fact (see its own field comment on `AgentTransitionData`) and so is
        the one signal actually snapshotted, alongside `action` itself, in `step()`'s collect phase.

        The one exception is the issue #280 WAITING shortcut below, which needs this step's fresh
        `agent.malfunction_handler.in_malfunction` roll (from `step()`'s own (0a)/(0b), already
        applied by the time this hook runs) - so it runs first in this same pass, preserving both its
        own correctness and its original before-the-rest-of-the-state-machine relative order.
        """
        for agent in self.agents:
            in_malfunction = agent.malfunction_handler.in_malfunction
            # design (issue #280): an earliest_departure=0 agent never goes through a
            # state_machine.step() call before the very first step() runs - tweak state directly
            # here so it already sees READY_TO_DEPART instead of stale WAITING (symmetric with
            # MALFUNCTION_OFF_MAP's own straight-to-MOVING shortcut in _handle_malfunction_off_map).
            # State-machine-only - map entry itself is derived from earliest_departure/elapsed_steps
            # directly in _candidate_entry_points, never from agent.state.
            if (self._elapsed_steps == 1 and agent.earliest_departure == 0
                    and not in_malfunction and agent.state == TrainState.WAITING):
                agent.state_machine.set_state(TrainState.READY_TO_DEPART)

        for agent in self.agents:
            agent_transition_data = self.temp_transition_data[agent.handle]
            agent.state_machine.set_transition_signals(StateTransitionSignals(
                in_malfunction=agent.malfunction_handler.in_malfunction,
                # +1: earliest_departure_reached is deliberately signalled one step early (see
                # rail_env.py's own historical comment on this formula) so the WAITING ->
                # READY_TO_DEPART transition it drives completes in step N-1 - self._elapsed_steps
                # is not re-incremented between step()'s per-agent loop and this hook (both run within
                # the same step() call, after step()'s own single, top-of-function increment), so this
                # must match loop 1's original formula exactly, not compensate for any further increment.
                earliest_departure_reached=agent.earliest_departure <= self._elapsed_steps + 1,
                stop_action_given=agent_transition_data.action == RailEnvActions.STOP_MOVING,
                movement_action_given=RailEnvActions.is_moving_action(agent_transition_data.action),
                movement_allowed=agent_transition_data.action_valid and agent_transition_data.resource_check,
                new_speed_zero=agent_transition_data.candidate_speed == 0.0,
                action_valid=agent_transition_data.action_valid,
            ))
            agent.state_machine.step()
            # update_if_reached() is a state_machine-internal mutation only - rail_env.py's own
            # step() control flow (handle_done_state) re-derives the same "reached target" fact
            # independently, from agent.current_entry_point/agent.arrival_time, never from agent.state.
            if agent_transition_data.resource_check and not agent_transition_data.done:
                agent.state_machine.update_if_reached(agent_transition_data.candidate_entry_point, agent.targets)
            # Off map or on map state and position should match.
            if not self._fast_state_position_sync_check(agent.state, agent.current_entry_point, self.remove_agents_at_target):
                agent.state_machine.state_position_sync_check(agent.current_entry_point, agent.handle, self.remove_agents_at_target)

        if self.check_step_pre_post_conditions:
            self._check_malfunction_state_postcondition()  # only holds after env.step()!
