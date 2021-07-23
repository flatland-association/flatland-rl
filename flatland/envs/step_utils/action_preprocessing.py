from flatland.envs.agent_utils import TrainState
from flatland.envs.rail_env import RailEnvActions

def process_illegal_action(action: RailEnvActions):
	# TODO - Dipam : This check is kind of weird, change this
	if action is None or action not in RailEnvActions._value2member_map_: 
		return RailEnvActions.DO_NOTHING
	else:
		return action

def process_do_nothing(state: TrainState):
	if state == TrainState.MOVING:
		action = RailEnvActions.MOVE_FORWARD
	else: 
		action = RailEnvActions.STOP_MOVING
	return action

def process_left_right(action, state, rail, position, direction):
	if not check_valid_action(action, state, rail, position, direction):
		action = RailEnvActions.MOVE_FORWARD
	return action

def check_valid_action(action, state, rail, position, direction):
	_, new_cell_valid, _, _, transition_valid = check_action_on_agent(action, state, rail, position, direction)
	action_is_valid = new_cell_valid and transition_valid
	return action_is_valid

def preprocess_action(action, state, rail, position, direction):
	"""
	Preprocesses actions to handle different situations of usage of action based on context
		- LEFT/RIGHT is converted to FORWARD if left/right is not available and train is moving
		- DO_NOTHING is converted to FORWARD if train is moving
		- DO_NOTHING is converted to STOP_MOVING if train is moving
	"""
	if state == TrainState.WAITING:
		action = RailEnvActions.DO_NOTHING

	action = process_illegal_action(action)

	if action == RailEnvActions.DO_NOTHING:
		action = process_do_nothing(state)
	elif action in [RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT]:
		action = process_left_right(action, state, rail, position, direction)
	
	if not check_valid_action(action, state, rail, position, direction):
		action = RailEnvActions.STOP_MOVING

	return action


# TODO - Placeholder - these will be renamed and moved out later

from flatland.envs.rail_env import fast_position_equal, fast_count_nonzero, fast_argmax, fast_clip, get_new_position

# TODO - Dipam - Improve these functions?
def check_action(action):
        """

        Parameters
        ----------
        agent : EnvAgent
        action : RailEnvActions

        Returns
        -------
        Tuple[Grid4TransitionsEnum,Tuple[int,int]]



        """
        transition_valid = None
        possible_transitions = rail.get_transitions(*position, direction)
        num_transitions = fast_count_nonzero(possible_transitions)

        new_direction = direction
        if action == RailEnvActions.MOVE_LEFT:
            new_direction = RailEnvActions.MOVE_FORWARD
            if num_transitions <= 1:
                transition_valid = False

        elif action == RailEnvActions.MOVE_RIGHT:
            new_direction = RailEnvActions.MOVE_FORWARD
            if num_transitions <= 1:
                transition_valid = False

        new_direction %= 4 # Dipam : Why?

        if action == RailEnvActions.MOVE_FORWARD and num_transitions == 1:
            # - dead-end, straight line or curved line;
            # new_direction will be the only valid transition
            # - take only available transition
            new_direction = fast_argmax(possible_transitions)
            transition_valid = True
        return new_direction, transition_valid


def check_action_on_agent(action, state, rail, position, direction):
        """
        Parameters
        ----------
        action : RailEnvActions
        agent : EnvAgent

        Returns
        -------
        bool
            Is it a legal move?
            1) transition allows the new_direction in the cell,
            2) the new cell is not empty (case 0),
            3) the cell is free, i.e., no agent is currently in that cell


        """
        # compute number of possible transitions in the current
        # cell used to check for invalid actions
        new_direction, transition_valid = check_action(agent, action)
        new_position = get_new_position(position, new_direction)

        new_cell_valid = (
            fast_position_equal(  # Check the new position is still in the grid
                new_position,
                fast_clip(new_position, [0, 0], [self.height - 1, self.width - 1]))
            and  # check the new position has some transitions (ie is not an empty cell)
            rail.get_full_transitions(*new_position) > 0)

        # If transition validity hasn't been checked yet.
        if transition_valid is None:
            transition_valid = rail.get_transition(
                (*position, direction),
                new_direction)

        # only call cell_free() if new cell is inside the scene
        if new_cell_valid:
            # Check the new position is not the same as any of the existing agent positions
            # (including itself, for simplicity, since it is moving)
            cell_free = self.cell_free(new_position)
        else:
            # if new cell is outside of scene -> cell_free is False
            cell_free = False
        return cell_free, new_cell_valid, new_direction, new_position, transition_valid

