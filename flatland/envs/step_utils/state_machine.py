import cython

from flatland.envs.step_utils.states import TrainState, StateTransitionSignals

# hoisted once at import time so `calculate_next_state`'s hot dispatch can compare a `cython.int` local against
# these instead of repeatedly looking up `TrainState.WAITING` etc. (a Python attribute access) per branch -
# derived from TrainState itself, so they can't drift out of sync with it.
_WAITING = TrainState.WAITING.value
_READY_TO_DEPART = TrainState.READY_TO_DEPART.value
_MALFUNCTION_OFF_MAP = TrainState.MALFUNCTION_OFF_MAP.value
_MOVING = TrainState.MOVING.value
_STOPPED = TrainState.STOPPED.value
_MALFUNCTION = TrainState.MALFUNCTION.value
_DONE = TrainState.DONE.value


class TrainStateMachine:
    def __init__(self, initial_state=TrainState.WAITING):
        self._initial_state = initial_state
        self._state = initial_state
        self.st_signals = StateTransitionSignals()
        self.next_state = None
        self.previous_state = None

    def _handle_waiting(self):
        """" Waiting state goes to ready to depart when earliest departure is reached"""
        if self.st_signals.in_malfunction:
            self.next_state = TrainState.MALFUNCTION_OFF_MAP
        elif self.st_signals.earliest_departure_reached:
            self.next_state = TrainState.READY_TO_DEPART
        else:
            self.next_state = TrainState.WAITING

    def _handle_ready_to_depart(self):
        """ Can only go to MOVING if a valid action is provided """
        if self.st_signals.in_malfunction:
            self.next_state = TrainState.MALFUNCTION_OFF_MAP
        elif self.st_signals.movement_action_given and self.st_signals.movement_allowed:
            self.next_state = TrainState.MOVING
        else:
            self.next_state = TrainState.READY_TO_DEPART

    def _handle_malfunction_off_map(self):
        """
        Off-map counterpart to MALFUNCTION. Malfunctions are rolled for every agent every step
        regardless of on/off-map status (see MalfunctionEffectsGenerator.on_episode_step_start,
        which iterates all agents unconditionally) - design: if malfunctions were only rolled for
        agents already on-map (MOVING/STOPPED/MALFUNCTION), the malfunction_rate an operator
        configures would understate the rate actually experienced once on-map, since off-map
        steps - WAITING for earliest_departure, or already READY_TO_DEPART but not yet moving
        (no movement action given, or the entry cell contested by another agent) - would never
        count against it. MALFUNCTION_OFF_MAP exists so an off-map agent can absorb that same
        roll (with no position/speed/distance to freeze - see TrainState.is_off_map_state())
        instead of being exempt from it.
        """
        if not self.st_signals.in_malfunction:
            if self.st_signals.earliest_departure_reached:
                # design: disallow entering the map stopped
                if self.st_signals.movement_action_given and self.st_signals.movement_allowed:
                    self.next_state = TrainState.MOVING
                else:
                    self.next_state = TrainState.READY_TO_DEPART
            else:
                self.next_state = TrainState.WAITING
        else:
            self.next_state = TrainState.MALFUNCTION_OFF_MAP

    def _handle_moving(self):
        if self.st_signals.in_malfunction:
            self.next_state = TrainState.MALFUNCTION
        elif self.st_signals.target_reached:
            # this branch is never used as target reached is not handled by state_machine.step() but by state_machine.update_if_reached()!
            self.next_state = TrainState.DONE
        elif (self.st_signals.stop_action_given and self.st_signals.new_speed_zero) or not self.st_signals.movement_allowed:
            self.next_state = TrainState.STOPPED
        else:
            self.next_state = TrainState.MOVING

    def _handle_stopped(self):
        if self.st_signals.in_malfunction:
            self.next_state = TrainState.MALFUNCTION
        elif self.st_signals.movement_action_given and self.st_signals.movement_allowed:
            self.next_state = TrainState.MOVING
        else:
            self.next_state = TrainState.STOPPED

    def _handle_malfunction(self):
        if not self.st_signals.in_malfunction:
            if self.st_signals.movement_action_given and self.st_signals.movement_allowed:
                self.next_state = TrainState.MOVING
            else:
                self.next_state = TrainState.STOPPED
        else:
            self.next_state = TrainState.MALFUNCTION

    def _handle_done(self):
        """"
        Done state is terminal - unlike _handle_waiting/_handle_ready_to_depart above, this ignores
        st_signals.in_malfunction entirely: a DONE agent can still roll a malfunction (rolled
        unconditionally for every agent, see _handle_malfunction_off_map's docstring) and have
        malfunction_handler.in_malfunction read True, but it never transitions to
        MALFUNCTION/MALFUNCTION_OFF_MAP for it - the roll is silently absorbed with no state
        consequence, since DONE is terminal. See rail_env.py's _check_malfunction_state_invariant,
        which excludes DONE agents from its state/in_malfunction consistency check for this reason.

        This holds independently of remove_agents_at_target, which only decides what DONE's position
        looks like (current_entry_point cleared to None if True, left at the target cell if False -
        neither is_on_map_state() nor is_off_map_state() count DONE either way, see
        handle_done_state()) - not whether a DONE agent can still malfunction.
        """
        self.next_state = TrainState.DONE

    def calculate_next_state(self, current_state):
        state_value: cython.int = current_state.value

        # _Handle the current state
        if state_value == _WAITING:
            self._handle_waiting()

        elif state_value == _READY_TO_DEPART:
            self._handle_ready_to_depart()

        elif state_value == _MALFUNCTION_OFF_MAP:
            self._handle_malfunction_off_map()

        elif state_value == _MOVING:
            self._handle_moving()

        elif state_value == _STOPPED:
            self._handle_stopped()

        elif state_value == _MALFUNCTION:
            self._handle_malfunction()

        elif state_value == _DONE:
            self._handle_done()

        else:
            raise ValueError(f"Got unexpected state {current_state}")

    def step(self):
        """
        Steps the state machine to the next state.

        By the time this call returns, self.state is the settled outcome of this very step() call -
        callers reading agent.state right after env.step() returns are seeing "what happened this
        step", not a preview of the next one. MALFUNCTION/MALFUNCTION_OFF_MAP transitions honor this:
        rail_env.py's step() decrements malfunction_handler's down-counter and rolls any new
        malfunction (MalfunctionEffectsGenerator.on_episode_step_start) at the very start of the same
        env.step() call (see rail_env.py's _check_malfunction_state_invariant).

        This still leaves an asymmetry worth knowing about: the counter itself is genuinely external to
        this step()-then-reflect contract - it is mutated by an outside generator as an *input* to this
        step's transition, not produced as this step's *output* the way self.state is. A controller can
        read agent.malfunction_handler.malfunction_down_counter/in_malfunction directly (plain public
        attributes, no special access contract), and doing so exposes this step's already-updated
        counter value - but neither that counter nor self.state can tell the controller that the
        *next* env.step() call is about to end the malfunction, since the decrement/roll that decides
        that only happens at the start of that next call, not this one.

        RailEnv.action_required() (rail_env.py) is neither purely state-derived nor purely
        distance/speed-derived - self.state gates which of four cases applies, and only one of them
        actually consults distance/speed:
        - WAITING / MALFUNCTION_OFF_MAP (off map, not yet eligible to move): always False, regardless
          of anything else.
        - READY_TO_DEPART (off map, eligible to move): always True, regardless of anything else.
        - MOVING / STOPPED / MALFUNCTION (on map): collapses to SpeedCounter.is_cell_exit() alone
          (distance + speed >= SEGMENT_LENGTH) - identical for all three on-map states, with no special
          case for MALFUNCTION. This is where a "malfunctioning agents never need an action" assumption
          comes from: ordinarily a malfunction freezes distance mid-cell and speed at 0, so
          is_cell_exit() stays False (see test_action_required_false_during_malfunction in
          test_flatland_envs_rail_env.py). But it is not a state-level guarantee - an agent already
          banked exactly at a cell boundary (distance == SEGMENT_LENGTH, e.g. after a denied crossing)
          satisfies is_cell_exit() from distance alone even at speed 0, so action_required reads True
          there whether or not the agent also happens to be malfunctioning at that same position (see
          test_action_required_at_full_segment_length in the same file).
        - DONE (terminal - neither on map nor off map per TrainState.is_on_map_state()/
          is_off_map_state()): always False, even with remove_agents_at_target=False leaving the agent
          parked at a real, banked-at-boundary position (is_cell_exit() would read True there too, but
          is never consulted - DONE is excluded from the on-map branch above).

        Either way, action_required only ever reports this step's already-settled outcome, never a
        lookahead onto the next step's malfunction-counter mutation.

        TODO revise design: MALFUNCTION and MALFUNCTION_OFF_MAP are asymmetric here even when
        earliest_departure is already reached for both - both transition straight into MOVING on the
        very step their malfunction ends (given a movement action that same step, see
        _handle_malfunction/_handle_malfunction_off_map above), but action_required disagrees about the
        steps leading up to that while still malfunctioning: an on-map MALFUNCTION banked at a cell
        boundary (distance == SEGMENT_LENGTH from a denied crossing before the malfunction hit) reads
        action_required True for every remaining malfunctioning step, not just the one where the
        malfunction actually ends - misleadingly, since movement_action_given has no effect while
        in_malfunction is still True regardless of what action_required says. MALFUNCTION_OFF_MAP never
        has this problem: an off-map agent has no distance/speed to bank against SEGMENT_LENGTH, so it
        reads action_required False for the entire malfunction, only flipping True once the transition
        into MOVING/READY_TO_DEPART has already happened. Confirmed empirically: with earliest_departure
        already reached (0) going in, a MALFUNCTION_OFF_MAP agent reports action_required
        False/False/True across a 3-step malfunction (last step already MOVING), while an on-map
        MALFUNCTION agent banked at a boundary reports True/True/True across the same shape of
        malfunction (last step STOPPED, since MOVE_FORWARD is invalid at that particular switch - see
        test_action_required_at_full_segment_length's malfunction variant for the on-map side).

        Three revise-design options were considered:

        Position-based definitions used below (matching this codebase's convention of deriving on/off-map
        and done-ness from position rather than state, see CLAUDE.md's post-step invariant checks):
        on_map := agent.current_entry_point is not None; off_map := not on_map; not_done :=
        agent.target_entry_point is None (set exactly once, permanently, the first step agent.state
        becomes DONE - see handle_done_state() in rail_env.py - so it is a reliable done/not-done proxy
        for the whole episode without reading agent.state at all; the only gap is a same-iteration,
        intra-step window between update_if_reached() flipping state to DONE and handle_done_state()
        setting target_entry_point a few lines later - never observable across a step() call boundary,
        so irrelevant to action_required, which is only ever read from get_info_dict() after step()
        returns).

        - (a) Add one rule for the malfunctioning case only: while in_malfunction, action_required =
          earliest_departure_reached and (is_cell_exit or off_map) and not_done; otherwise
          state == TrainState.READY_TO_DEPART or (state.is_on_map_state() and is_cell_exit) - today's
          formula, unchanged. The `and not_done` guard is needed because a DONE agent can still have
          in_malfunction True (see _handle_done's docstring above) and earliest_departure_reached is
          trivially True long after departure - without it, the malfunction branch would wrongly
          evaluate to True for a terminal agent.
        - (b) (is_cell_exit and on_map and not_done) or (earliest_departure_reached and off_map and
          not_done). Resolves the asymmetry above the same way (a) does: MALFUNCTION_OFF_MAP starts
          reading action_required True once earliest_departure_reached, instead of being hardcoded False
          for the entire malfunction, exactly like READY_TO_DEPART already does. Verified equivalent to
          guarded (a) over every state-machine-reachable input (0 mismatches), but not as unconstrained
          boolean formulas: of all 112 raw (state, in_malfunction, earliest_departure_reached, is_cell_exit,
          on_map) combinations ignoring reachability, 24 disagree - every one of them an input the state
          machine can never actually produce (e.g. WAITING with earliest_departure_reached=True while not
          malfunctioning - synchronously transitions to READY_TO_DEPART the same step, per _handle_waiting,
          so never externally observable; or an on-map state with a None position, which violates the
          position/state sync invariant). (a) is the more defensive of the two here: its non-malfunctioning
          branch is keyed directly to `state == READY_TO_DEPART` (an enum identity check), while (b) is
          keyed to earliest_departure_reached/on_map (recomputed booleans) for every off-map state - if the
          state machine's own synchronization were ever violated elsewhere (a bug, or a bypass like
          `_set_state()`), (a) would stay correct while (b) could drift.
        - (c) action_required = is_cell_exit if (on_map and not_done) else False.
          Relative to today this only drops READY_TO_DEPART's unconditional True and fixes the DONE case
          - it does not fix the MALFUNCTION/MALFUNCTION_OFF_MAP asymmetry (MALFUNCTION_OFF_MAP stays
          hardcoded False, on-map MALFUNCTION stays exactly as is_cell_exit-quirky as today), but gives
          action_required simpler on-map-only semantics (position and speed/distance only, ignoring
          earliest_departure/malfunction/state entirely).
        """

        current_state = self._state

        # Clear next state
        self.clear_next_state()

        # Handle current state to get next_state
        self.calculate_next_state(current_state)

        # Set next state
        self.set_state(self.next_state)

    def clear_next_state(self):
        self.next_state = None

    def set_state(self, state):
        if not TrainState.check_valid_state(state):
            raise ValueError(f"Cannot set invalid state {state}")
        self.previous_state = self._state
        self._state = state

    def reset(self):
        self._state = self._initial_state
        self.previous_state = None
        self.st_signals = StateTransitionSignals()
        self.clear_next_state()

    def update_if_reached(self, entry_point, targets):
        # Need to do this hacky fix for now, state machine needed speed related states for proper handling
        self.st_signals.target_reached = entry_point in targets
        if self.st_signals.target_reached:
            self.next_state = TrainState.DONE
            self.set_state(self.next_state)

    @property
    def state(self):
        return self._state

    @property
    def state_transition_signals(self):
        return self.st_signals

    def set_transition_signals(self, state_transition_signals):
        self.st_signals = state_transition_signals

    def state_position_sync_check(self, entry_point, i_agent, remove_agents_at_target):
        """ Check for whether on map and off map states are matching with position being None """
        if self.state.is_on_map_state() and entry_point is None:
            raise ValueError("Agent ID {} Agent State {} is on map Agent Position {} if off map ".format(
                i_agent, str(self.state), str(entry_point)))
        elif self.state.is_off_map_state() and entry_point is not None:
            raise ValueError("Agent ID {} Agent State {} is off map Agent Position {} if on map ".format(
                i_agent, str(self.state), str(entry_point)))
        elif self.state == TrainState.DONE and remove_agents_at_target and entry_point is not None:
            raise ValueError("Agent ID {} Agent State {} is not None Agent Position {} if remove_agents_at_target".format(
                i_agent, str(self.state), str(entry_point)))

    def __repr__(self):
        return (
            f"TrainStateMachine(\n"
            f"\tstate={str(self.state)},\n"
            f"\tprevious_state={str(self.previous_state) if self.previous_state is not None else None},\n"
            f"\tst_signals={self.st_signals}\n"
            f")"
        )


    def to_dict(self):
        return {"state": self._state,
                "previous_state": self.previous_state}

    @staticmethod
    def from_dict(load_dict) -> "TrainStateMachine":
        sm = TrainStateMachine()
        sm.set_state(load_dict['state'])
        sm.previous_state = load_dict['previous_state']
        return sm

    def __eq__(self, other):
        return self._state == other._state and self.previous_state == other.previous_state

    def __getstate__(self):
        return {
            "_initial_state": self._initial_state,
            "_state": self._state,
            "st_signals": self.st_signals,
            "next_state": self.next_state,
            "previous_state": self.previous_state,
        }

    def __setstate__(self, state):
        self._initial_state = state["_initial_state"]
        self._state = state["_state"]
        self.st_signals = state["st_signals"]
        self.next_state = state["next_state"]
        self.previous_state = state["previous_state"]
