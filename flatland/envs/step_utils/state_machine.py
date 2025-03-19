from flatland.envs.fast_methods import fast_position_equal
from flatland.envs.step_utils.states import TrainState, StateTransitionSignals


class TrainStateMachine:
    def __init__(self, initial_state=TrainState.WAITING):
        self._initial_state = initial_state
        self._state = initial_state
        self.st_signals = StateTransitionSignals()
        self.next_state = None
        self.previous_state = None

    def _handle_waiting(self):
        """" Waiting state goes to ready to depart when earliest departure is reached"""
        # TODO: Important - The malfunction handling is not like proper state machine
        #                   Both transition signals can happen at the same time
        #                   Atleast mention it in the diagram
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
        if not self.st_signals.in_malfunction:
            if self.st_signals.earliest_departure_reached:
                if self.st_signals.movement_action_given and self.st_signals.movement_allowed:
                    self.next_state = TrainState.MOVING
                elif self.st_signals.stop_action_given and self.st_signals.movement_allowed:
                    self.next_state = TrainState.STOPPED
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
        elif self.st_signals.stop_action_given or not self.st_signals.movement_allowed:
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
        """" Done state is terminal """
        self.next_state = TrainState.DONE

    def calculate_next_state(self, current_state):

        # _Handle the current state
        if current_state == TrainState.WAITING:
            self._handle_waiting()

        elif current_state == TrainState.READY_TO_DEPART:
            self._handle_ready_to_depart()

        elif current_state == TrainState.MALFUNCTION_OFF_MAP:
            self._handle_malfunction_off_map()

        elif current_state == TrainState.MOVING:
            self._handle_moving()

        elif current_state == TrainState.STOPPED:
            self._handle_stopped()

        elif current_state == TrainState.MALFUNCTION:
            self._handle_malfunction()

        elif current_state == TrainState.DONE:
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

    def update_if_reached(self, position, target):
        # Need to do this hacky fix for now, state machine needed speed related states for proper handling
        self.st_signals.target_reached = fast_position_equal(position, target)
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

    def state_position_sync_check(self, position, i_agent, remove_agents_at_target):
        """ Check for whether on map and off map states are matching with position being None """
        if self.state.is_on_map_state() and position is None:
            raise ValueError("Agent ID {} Agent State {} is on map Agent Position {} if off map ".format(
                i_agent, str(self.state), str(position)))
        elif self.state.is_off_map_state() and position is not None:
            raise ValueError("Agent ID {} Agent State {} is off map Agent Position {} if on map ".format(
                i_agent, str(self.state), str(position)))
        elif self.state == TrainState.DONE and remove_agents_at_target and position is not None:
            raise ValueError("Agent ID {} Agent State {} is not None Agent Position {} if remove_agents_at_target".format(
                i_agent, str(self.state), str(position)))

    def __repr__(self):
        return f"\n \
                 state: {str(self.state.name)}      previous_state {str(self.previous_state.name) if self.previous_state is not None else None} \n \
                 st_signals: {self.st_signals}"

    def to_dict(self):
        return {"state": self._state,
                "previous_state": self.previous_state}

    def from_dict(self, load_dict):
        self.set_state(load_dict['state'])
        self.previous_state = load_dict['previous_state']

    def __eq__(self, other):
        return self._state == other._state and self.previous_state == other.previous_state
