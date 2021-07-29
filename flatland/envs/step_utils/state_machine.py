from attr import s
from flatland.envs.step_utils.states import TrainState

class TrainStateMachine:
    def __init__(self, initial_state=TrainState.WAITING):
        self._initial_state = initial_state
        self._state = initial_state
        self.st_signals = {} # State Transition Signals
        self.next_state = None
    
    def _handle_waiting(self):
        """" Waiting state goes to ready to depart when earliest departure is reached"""
        # TODO: Important - The malfunction handling is not like proper state machine 
        #                   Both transition signals can happen at the same time
        #                   Atleast mention it in the diagram
        if self.st_signals['malfunction_onset']:  
            self.next_state = TrainState.MALFUNCTION_OFF_MAP
        elif self.st_signals['earliest_departure_reached']:
            self.next_state = TrainState.READY_TO_DEPART
        else:
            self.next_state = TrainState.WAITING

    def _handle_ready_to_depart(self):
        """ Can only go to MOVING if a valid action is provided """
        if self.st_signals['malfunction_onset']:  
            self.next_state = TrainState.MALFUNCTION_OFF_MAP
        elif self.st_signals['valid_movement_action_given']:
            self.next_state = TrainState.MOVING
        else:
            self.next_state = TrainState.READY_TO_DEPART
    
    def _handle_malfunction_off_map(self):
        if self.st_signals['malfunction_counter_complete']:
            if self.st_signals['earliest_departure_reached']:
                self.next_state = TrainState.READY_TO_DEPART
            else:
                self.next_state = TrainState.STOPPED
        else:
            self.next_state = TrainState.WAITING
    
    def _handle_moving(self):
        if self.st_signals['malfunction_onset']:
            self.next_state = TrainState.MALFUNCTION
        elif self.st_signals['target_reached']:
            self.next_state = TrainState.DONE
        elif self.st_signals['stop_action_given'] or self.st_signals['movement_conflict']:
            self.next_state = TrainState.STOPPED
        else:
            self.next_state = TrainState.MOVING
    
    def _handle_stopped(self):
        if self.st_signals['malfunction_onset']:
            self.next_state = TrainState.MALFUNCTION
        elif self.st_signals['valid_movement_action_given']:
            self.next_state = TrainState.MOVING
        else:
            self.next_state = TrainState.STOPPED
    
    def _handle_malfunction(self):
        if self.st_signals['malfunction_counter_complete'] and \
           self.st_signals['valid_movement_action_given']:
            self.next_state = TrainState.MOVING
        elif self.st_signals['malfunction_counter_complete'] and \
             (self.st_signals['stop_action_given'] or self.st_signals['movement_conflict']):
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
        self._state = state

    def reset(self):
        self._state = self._initial_state
        self.st_signals = {}
        self.clear_next_state()

    @property
    def state(self):
        return self._state
    
    @property
    def state_transition_signals(self):
        return self.st_signals
    
    def set_transition_signals(self, state_transition_signals):
        self.st_signals = state_transition_signals # TODO: Important: Check all keys are present and if not raise error


        
