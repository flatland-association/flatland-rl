"""
Definition of the RailEnv environment and related level-generation functions.

Generator functions are functions that take width, height and num_resets as arguments and return
a GridTransitionMap object.
"""
import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import TreeObsForRailEnv
from flatland.envs.generators import random_rail_generator

# from flatland.core.transitions import Grid8Transitions, RailEnvTransitions
# from flatland.core.transition_map import GridTransitionMap


class EnvAgentStatic(object):
    """ TODO: EnvAgentStatic - To store initial position, direction and target.
        This is like static data for the environment - it's where an agent starts,
        rather than where it is at the moment.
        The target should also be stored here.
    """
    def __init__(self, rcPos, iDir, rcTarget):
        self.rcPos = rcPos
        self.iDir = iDir
        self.rcTarget = rcTarget


class EnvAgent(object):
    """ TODO: EnvAgent - replace separate agent lists with a single list
        of agent objects.  The EnvAgent represent's the environment's view
        of the dynamic agent state.  So target is not part of it - target is
        static.
    """
    def __init__(self, rcPos, iDir):
        self.rcPos = rcPos
        self.iDir = iDir


class RailEnv(Environment):
    """
    RailEnv environment class.

    RailEnv is an environment inspired by a (simplified version of) a rail
    network, in which agents (trains) have to navigate to their target
    locations in the shortest time possible, while at the same time cooperating
    to avoid bottlenecks.

    The valid actions in the environment are:
        0: do nothing
        1: turn left and move to the next cell
        2: move to the next cell in front of the agent
        3: turn right and move to the next cell

    Moving forward in a dead-end cell makes the agent turn 180 degrees and step
    to the cell it came from.

    The actions of the agents are executed in order of their handle to prevent
    deadlocks and to allow them to learn relative priorities.

    TODO: WRITE ABOUT THE REWARD FUNCTION, and possibly allow for alpha and
    beta to be passed as parameters to __init__().
    """

    def __init__(self,
                 width,
                 height,
                 rail_generator=random_rail_generator(),
                 number_of_agents=1,
                 obs_builder_object=TreeObsForRailEnv(max_depth=2)):
        """
        Environment init.

        Parameters
        -------
        rail_generator : function
            The rail_generator function is a function that takes the width and
            height of a  rail map along with the number of times the env has
            been reset, and returns a GridTransitionMap object.
            Implemented functions are:
                random_rail_generator : generate a random rail of given size
                rail_from_GridTransitionMap_generator(rail_map) : generate a rail from
                                        a GridTransitionMap object
                rail_from_manual_specifications_generator(rail_spec) : generate a rail from
                                        a rail specifications array
                TODO: generate_rail_from_saved_list or from list of ndarray bitmaps ---
        width : int
            The width of the rail map. Potentially in the future,
            a range of widths to sample from.
        height : int
            The height of the rail map. Potentially in the future,
            a range of heights to sample from.
        number_of_agents : int
            Number of agents to spawn on the map. Potentially in the future,
            a range of number of agents to sample from.
        obs_builder_object: ObservationBuilder object
            ObservationBuilder-derived object that takes builds observation
            vectors for each agent.
        """

        self.rail_generator = rail_generator
        self.rail = None
        self.width = width
        self.height = height

        self.number_of_agents = number_of_agents

        self.obs_builder = obs_builder_object
        self.obs_builder._set_env(self)

        self.actions = [0] * self.number_of_agents
        self.rewards = [0] * self.number_of_agents
        self.done = False

        self.dones = {"__all__": False}
        self.obs_dict = {}
        self.rewards_dict = {}

        self.agents_handles = list(range(self.number_of_agents))

        # self.agents_position = []
        # self.agents_target = []
        # self.agents_direction = []
        self.num_resets = 0
        self.reset()
        self.num_resets = 0

        self.valid_positions = None

    def get_agent_handles(self):
        return self.agents_handles

    def fill_valid_positions(self):
        ''' Populate the valid_positions list for the current TransitionMap.
        '''
        self.valid_positions = valid_positions = []
        for r in range(self.height):
            for c in range(self.width):
                if self.rail.get_transitions((r, c)) > 0:
                    valid_positions.append((r, c))

    def check_agent_lists(self):
        ''' Check that the agent_handles, position and direction lists are all of length
            number_of_agents.
            (Suggest this is replaced with a single list of Agent objects :)
        '''
        for lAgents, name in zip(
                [self.agents_handles, self.agents_position, self.agents_direction],
                ["handles", "positions", "directions"]):
            assert self.number_of_agents == len(lAgents), "Inconsistent agent list:" + name

    def check_agent_locdirpath(self, iAgent):
        ''' Check that agent iAgent has a valid location and direction,
            with a path to its target.
            (Not currently used?)
        '''
        valid_movements = []
        for direction in range(4):
            position = self.agents_position[iAgent]
            moves = self.rail.get_transitions((position[0], position[1], direction))
            for move_index in range(4):
                if moves[move_index]:
                    valid_movements.append((direction, move_index))

        valid_starting_directions = []
        for m in valid_movements:
            new_position = self._new_position(self.agents_position[iAgent], m[1])
            if m[0] not in valid_starting_directions and \
                    self._path_exists(new_position, m[0], self.agents_target[iAgent]):
                valid_starting_directions.append(m[0])

        if len(valid_starting_directions) == 0:
            return False
        else:
            return True

    def pick_agent_direction(self, rcPos, rcTarget):
        """ Pick and return a valid direction index (0..3) for an agent starting at
            row,col rcPos with target rcTarget.
            Return None if no path exists.
            Picks random direction if more than one exists (uniformly).
        """
        valid_movements = []
        for direction in range(4):
            moves = self.rail.get_transitions((*rcPos, direction))
            for move_index in range(4):
                if moves[move_index]:
                    valid_movements.append((direction, move_index))
        # print("pos", rcPos, "targ", rcTarget, "valid movements", valid_movements)

        valid_starting_directions = []
        for m in valid_movements:
            new_position = self._new_position(rcPos, m[1])
            if m[0] not in valid_starting_directions and self._path_exists(new_position, m[0], rcTarget):
                valid_starting_directions.append(m[0])

        if len(valid_starting_directions) == 0:
            return None
        else:
            return valid_starting_directions[np.random.choice(len(valid_starting_directions), 1)[0]]

    def add_agent(self, rcPos=None, rcTarget=None, iDir=None):
        """ Add a new agent at position rcPos with target rcTarget and
            initial direction index iDir.
            Should also store this initial position etc as environment "meta-data"
            but this does not yet exist.
        """
        self.check_agent_lists()

        if rcPos is None:
            rcPos = np.random.choice(len(self.valid_positions))

        iAgent = self.number_of_agents

        if iDir is None:
            iDir = self.pick_agent_direction(rcPos, rcTarget)
        if iDir is None:
            print("Error picking agent direction at pos:", rcPos)
            return None

        self.agents_position.append(tuple(rcPos))  # ensure it's a tuple not a list
        self.agents_handles.append(max(self.agents_handles + [-1]) + 1)  # max(handles) + 1, starting at 0
        self.agents_direction.append(iDir)
        self.agents_target.append(rcPos)  # set the target to the origin initially
        self.number_of_agents += 1
        self.check_agent_lists()
        return iAgent

    def reset(self, regen_rail=True, replace_agents=True):
        if regen_rail or self.rail is None:
            # TODO: Import not only rail information but also start and goal positions
            self.rail = self.rail_generator(self.width, self.height, self.num_resets)
            self.fill_valid_positions()

        self.num_resets += 1

        self.dones = {"__all__": False}
        for handle in self.agents_handles:
            self.dones[handle] = False

        # Use a TreeObsForRailEnv to compute distance maps to each agent's target, to sample initial
        # agent's orientations that allow a valid solution.
        # TODO: Possibility ot fill valid positions from list of goals and start
        self.fill_valid_positions()

        if replace_agents:
            re_generate = True
            while re_generate:

                # self.agents_position = random.sample(valid_positions,
                #                                     self.number_of_agents)
                self.agents_position = [
                    self.valid_positions[i] for i in
                    np.random.choice(len(self.valid_positions), self.number_of_agents)]
                self.agents_target = [
                    self.valid_positions[i] for i in
                    np.random.choice(len(self.valid_positions), self.number_of_agents)]

                # agents_direction must be a direction for which a solution is
                # guaranteed.
                self.agents_direction = [0] * self.number_of_agents
                re_generate = False

                for i in range(self.number_of_agents):
                    direction = self.pick_agent_direction(self.agents_position[i], self.agents_target[i])
                    if direction is None:
                        re_generate = True
                        break
                    else:
                        self.agents_direction[i] = direction
                
        # Reset the state of the observation builder with the new environment
        self.obs_builder.reset()

        # Return the new observation vectors for each agent
        return self._get_observations()

    def step(self, action_dict):
        alpha = 1.0
        beta = 1.0

        invalid_action_penalty = -2
        step_penalty = -1 * alpha
        global_reward = 1 * beta

        # Reset the step rewards
        self.rewards_dict = dict()
        for handle in self.agents_handles:
            self.rewards_dict[handle] = 0

        if self.dones["__all__"]:
            return self._get_observations(), self.rewards_dict, self.dones, {}

        for i in range(len(self.agents_handles)):
            handle = self.agents_handles[i]
            transition_isValid = None

            if handle not in action_dict:
                continue

            if self.dones[handle]:
                continue
            action = action_dict[handle]

            if action < 0 or action > 3:
                print('ERROR: illegal action=', action,
                      'for agent with handle=', handle)
                return

            if action > 0:
                pos = self.agents_position[i]
                direction = self.agents_direction[i]

                # compute number of possible transitions in the current
                # cell used to check for invalid actions

                nbits = 0
                tmp = self.rail.get_transitions((pos[0], pos[1]))
                possible_transitions = self.rail.get_transitions((pos[0], pos[1], direction))
                # print(np.sum(self.rail.get_transitions((pos[0], pos[1],direction))),
                # self.rail.get_transitions((pos[0], pos[1],direction)),
                # self.rail.get_transitions((pos[0], pos[1])),
                # (pos[0], pos[1],direction))

                while tmp > 0:
                    nbits += (tmp & 1)
                    tmp = tmp >> 1
                movement = direction
                # print(nbits,np.sum(possible_transitions))
                if action == 1:
                    movement = direction - 1
                    if nbits <= 2 or np.sum(possible_transitions) <= 1:
                        transition_isValid = False

                elif action == 3:
                    movement = direction + 1
                    if nbits <= 2 or np.sum(possible_transitions) <= 1:
                        transition_isValid = False
                if movement < 0:
                    movement += 4
                if movement >= 4:
                    movement -= 4

                is_deadend = False
                if action == 2:
                    if nbits == 1:
                        # dead-end;  assuming the rail network is consistent,
                        # this should match the direction the agent has come
                        # from. But it's better to check in any case.
                        reverse_direction = 0
                        if direction == 0:
                            reverse_direction = 2
                        elif direction == 1:
                            reverse_direction = 3
                        elif direction == 2:
                            reverse_direction = 0
                        elif direction == 3:
                            reverse_direction = 1

                        valid_transition = self.rail.get_transition(
                            (pos[0], pos[1], direction),
                            reverse_direction)
                        if valid_transition:
                            direction = reverse_direction
                            movement = reverse_direction
                            is_deadend = True

                    if np.sum(possible_transitions) == 1:
                        # Take only available transition
                        movement = np.argmax(possible_transitions)

                new_position = self._new_position(pos, movement)
                # Is it a legal move?  1) transition allows the movement in the
                # cell,  2) the new cell is not empty (case 0),  3) the cell is
                # free, i.e., no agent is currently in that cell
                if (
                        new_position[1] >= self.width or
                        new_position[0] >= self.height or
                        new_position[0] < 0 or new_position[1] < 0):
                    new_cell_isValid = False

                elif self.rail.get_transitions((new_position[0], new_position[1])) > 0:
                    new_cell_isValid = True
                else:
                    new_cell_isValid = False

                # If transition validity hasn't been checked yet.
                if transition_isValid is None:
                    transition_isValid = self.rail.get_transition(
                        (pos[0], pos[1], direction),
                        movement) or is_deadend

                cell_isFree = True
                for j in range(self.number_of_agents):
                    if self.agents_position[j] == new_position:
                        cell_isFree = False
                        break

                if new_cell_isValid and transition_isValid and cell_isFree:
                    # move and change direction to face the movement that was
                    # performed
                    self.agents_position[i] = new_position
                    self.agents_direction[i] = movement
                else:
                    # the action was not valid, add penalty
                    self.rewards_dict[handle] += invalid_action_penalty

            # if agent is not in target position, add step penalty
            if self.agents_position[i][0] == self.agents_target[i][0] and \
                    self.agents_position[i][1] == self.agents_target[i][1]:
                self.dones[handle] = True
            else:
                self.rewards_dict[handle] += step_penalty

        # Check for end of episode + add global reward to all rewards!
        num_agents_in_target_position = 0
        for i in range(self.number_of_agents):
            if self.agents_position[i][0] == self.agents_target[i][0] and \
                    self.agents_position[i][1] == self.agents_target[i][1]:
                num_agents_in_target_position += 1

        if num_agents_in_target_position == self.number_of_agents:
            self.dones["__all__"] = True
            self.rewards_dict = [r + global_reward for r in self.rewards_dict]

        # Reset the step actions (in case some agent doesn't 'register_action'
        # on the next step)
        self.actions = [0] * self.number_of_agents
        return self._get_observations(), self.rewards_dict, self.dones, {}

    def _new_position(self, position, movement):
        if movement == 0:  # NORTH
            return (position[0] - 1, position[1])
        elif movement == 1:  # EAST
            return (position[0], position[1] + 1)
        elif movement == 2:  # SOUTH
            return (position[0] + 1, position[1])
        elif movement == 3:  # WEST
            return (position[0], position[1] - 1)

    def _path_exists(self, start, direction, end):
        # BFS - Check if a path exists between the 2 nodes

        visited = set()
        stack = [(start, direction)]
        while stack:
            node = stack.pop()
            if node[0][0] == end[0] and node[0][1] == end[1]:
                return 1
            if node not in visited:
                visited.add(node)
                moves = self.rail.get_transitions((node[0][0], node[0][1], node[1]))
                for move_index in range(4):
                    if moves[move_index]:
                        stack.append((self._new_position(node[0], move_index),
                                      move_index))

                # If cell is a dead-end, append previous node with reversed
                # orientation!
                nbits = 0
                tmp = self.rail.get_transitions((node[0][0], node[0][1]))
                while tmp > 0:
                    nbits += (tmp & 1)
                    tmp = tmp >> 1
                if nbits == 1:
                    stack.append((node[0], (node[1] + 2) % 4))

        return 0

    def _get_observations(self):
        self.obs_dict = {}
        for handle in self.agents_handles:
            self.obs_dict[handle] = self.obs_builder.get(handle)
        return self.obs_dict

    def render(self):
        # TODO:
        pass
