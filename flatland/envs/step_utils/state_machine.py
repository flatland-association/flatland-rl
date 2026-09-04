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
        consequence, since DONE is terminal. See rail_env.py's _check_malfunction_state_postcondition,
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
        env.step() call (see rail_env.py's _check_malfunction_state_postcondition).

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
          (speed > 0 and distance + speed >= SEGMENT_LENGTH - see design_by_contract.md) - identical
          for all three on-map states, with no special case for MALFUNCTION. Since is_cell_exit()
          requires speed > 0, it reads False for any on-map agent parked at speed 0, regardless of
          distance - including a STOPPED/MALFUNCTION agent banked exactly at a cell boundary (distance
          == SEGMENT_LENGTH, e.g. after a denied crossing). This makes "malfunctioning agents never
          need an action" a genuine state-level guarantee - in_malfunction always forces speed to 0
          (see _candidate_speed's own malfunction branch in rail_env.py), so is_cell_exit() is always
          False throughout any malfunction, on map or off - see
          test_action_required_false_during_malfunction and
          test_action_required_at_full_segment_length in test_flatland_envs_rail_env.py.
        - DONE (terminal - neither on map nor off map per TrainState.is_on_map_state()/
          is_off_map_state()): always False, even with remove_agents_at_target=False leaving the agent
          parked at a real position - is_cell_exit() is never consulted either way, since DONE is
          excluded from the on-map branch above.

        Either way, action_required only ever reports this step's already-settled outcome, never a
        lookahead onto the next step's malfunction-counter mutation.

        design: MALFUNCTION and MALFUNCTION_OFF_MAP used to be asymmetric here even when
        earliest_departure was already reached for both - both transition straight into MOVING on the
        very step their malfunction ends (given a movement action that same step, see
        _handle_malfunction/_handle_malfunction_off_map above), but action_required used to disagree
        about the steps leading up to that while still malfunctioning: an on-map MALFUNCTION agent
        banked at a cell boundary (distance == SEGMENT_LENGTH from a denied crossing before the
        malfunction hit) used to read action_required True for every remaining malfunctioning step -
        misleadingly, since movement_action_given has no effect while in_malfunction is still True
        regardless of what action_required says - while MALFUNCTION_OFF_MAP correctly read False for
        the entire malfunction (an off-map agent has no distance/speed to bank against SEGMENT_LENGTH).

        This asymmetry is now resolved - not by revising today's action_required formula above, which
        is unchanged - but by SpeedCounter.is_cell_exit()'s own speed > 0 guard (see
        design_by_contract.md): since in_malfunction always forces speed to 0, is_cell_exit() - and so
        action_required - now reads False throughout any on-map MALFUNCTION too, symmetric with
        MALFUNCTION_OFF_MAP, only flipping True again once the malfunction clears and a movement action
        promotes the agent back to MOVING. Confirmed empirically: with earliest_departure already
        reached (0) going in, both a MALFUNCTION_OFF_MAP agent and an on-map MALFUNCTION agent banked
        at a boundary now report action_required False/False/True across a 3-step malfunction (last
        step already MOVING again - see test_action_required_at_full_segment_length's malfunction
        variant for the on-map side).
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
