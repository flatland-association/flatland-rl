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
                # TODO https://github.com/flatland-association/flatland-rl/issues/280 revise design: should we not go to the READY_TO_DEPART first instead of directly to MOVING?
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
        """ Steps the state machine to the next state """

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
