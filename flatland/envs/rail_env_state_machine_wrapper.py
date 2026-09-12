from flatland.envs.rail_env import AbstractRailEnv
from flatland.envs.step_utils.states import TrainState


def RailEnvStateMachineWrapper(env: AbstractRailEnv, skip_state_machine_update: bool = False) -> AbstractRailEnv:
    """
    Patches `env`'s class in place so its `step()` also updates `agent.state`/`agent.state_machine`,
    via `AbstractRailEnv._finalize_step_state_machine_hook()` - a hook `step()` calls internally after
    finalizing this step's position/speed/reward/done (which never read `agent.state`/`agent.state_machine`
    themselves, wrapped or not) but strictly before building this step's observations/info dict, so an
    obs_builder/predictor that does read `agent.state` (e.g. `TreeObsForRailEnv`,
    `ShortestPathPredictorForRailEnv`) sees it already consistent with this step's position - not a
    plain post-`step()` wrapper, which would leave the two one step out of phase (see
    `_update_state_machine_after_step`'s own docstring). Without wrapping, `agent.state`/
    `agent.state_machine` are never touched at all - an obs_builder/predictor that depends on
    `agent.state` for correctness (not just `get_info_dict()`'s purely-informational `state`/
    `action_required` fields) needs the env wrapped to behave correctly.

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

    def _finalize_step_state_machine_hook(self):
        if not self.skip_state_machine_update:
            self._update_state_machine_after_step()

    def _update_state_machine_after_step(self):
        """
        All `agent.state`/`agent.state_machine` bookkeeping for the step that just ran, in one pass -
        called by `AbstractRailEnv.step()` itself (via `_finalize_step_state_machine_hook()`) once
        this step's position/speed/reward/done are finalized but before observations/info are built -
        see `TrainStateMachine` (`flatland/envs/step_utils/state_machine.py`): every transition method,
        `update_if_reached()`, and `state_position_sync_check()` read only their explicit arguments/
        `self.st_signals`, never another agent nor anything `step()` mutates later - so running all
        of it in a single pass here (rather than interleaved per-agent inside `step()`'s own per-agent
        loop, as it used to be) is behaviorally equivalent.

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
            agent.state_machine.set_transition_signals(agent_transition_data.state_transition_signal)
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
