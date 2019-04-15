"""
The env module defines the base Environment class.
The base Environment class is adapted from rllib.env.MultiAgentEnv
(https://github.com/ray-project/ray).
"""
import random


class Environment:
    """
    Base interface for multi-agent environments in Flatland.

    Agents are identified by agent ids (handles).
    Examples:
        >>> obs = env.reset()
        >>> print(obs)
        {
            "train_0": [2.4, 1.6],
            "train_1": [3.4, -3.2],
        }
        >>> obs, rewards, dones, infos = env.step(
            action_dict={
                "train_0": 1, "train_1": 0})
        >>> print(rewards)
        {
            "train_0": 3,
            "train_1": -1,
        }
        >>> print(dones)
        {
            "train_0": False,    # train_0 is still running
            "train_1": True,     # train_1 is done
            "__all__": False,    # the env is not done
        }
        >>> print(infos)
        {
            "train_0": {},  # info for train_0
            "train_1": {},  # info for train_1
        }
    """

    def __init__(self):
        pass

    def reset(self):
        """
        Resets the env and returns observations from agents in the environment.

        Returns:
        obs : dict
            New observations for each agent.
        """
        raise NotImplementedError()

    def step(self, action_dict):
        """
        Environment step.

        Performs an environment step with simultaneous execution of actions for
        agents in action_dict. Returns observations for the agents.
        The returns are dicts mapping from agent_id strings to values.

        Parameters
        -------
        action_dict : dict
            Dictionary of actions to execute, indexed by agent id.

        Returns
        -------
        obs : dict
            New observations for each ready agent.
        rewards: dict
            Reward values for each ready agent.
        dones : dict
            Done values for each ready agent. The special key "__all__"
            (required) is used to indicate env termination.
        infos : dict
            Optional info values for each agent id.
        """
        raise NotImplementedError()

    def render(self):
        """
        Perform rendering of the environment.
        """
        raise NotImplementedError()

    def get_agent_handles(self):
        """
        Returns a list of agents' handles to be used as keys in the step()
        function.
        """
        raise NotImplementedError()


class RailEnv:
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

    def __init__(self, rail, number_of_agents=1):
        """
        Environment init.

        Parameters
        -------
        rail : numpy.ndarray of type numpy.uint16
            The transition matrix that defines the environment.
        number_of_agents : int
            Number of agents to spawn on the map.
        """

        self.rail = rail
        self.width = rail.width
        self.height = rail.height

        self.number_of_agents = number_of_agents

        self.actions = [0]*self.number_of_agents
        self.rewards = [0]*self.number_of_agents
        self.done = False

        self.agents_handles = list(range(self.number_of_agents))

    def get_agent_handles(self):
        return self.agents_handles

    def reset(self):
        self.dones = {"__all__": False}
        for handle in self.agents_handles:
            self.dones[handle] = False

        re_generate = True
        while re_generate:
            valid_positions = []
            for r in range(self.height):
                for c in range(self.width):
                    if self.rail.get_transitions((r, c)) > 0:
                        valid_positions.append((r, c))

            self.agents_position = random.sample(valid_positions,
                                                 self.number_of_agents)
            self.agents_target = random.sample(valid_positions,
                                               self.number_of_agents)

            # agents_direction must be a direction for which a solution is
            # guaranteed.
            self.agents_direction = [0]*self.number_of_agents
            re_generate = False
            for i in range(self.number_of_agents):
                valid_movements = []
                for direction in range(4):
                    position = self.agents_position[i]
                    moves = self.rail.get_transitions(
                            (position[0], position[1], direction))
                    for move_index in range(4):
                        if moves[move_index]:
                            valid_movements.append((direction, move_index))

                valid_starting_directions = []
                for m in valid_movements:
                    new_position = self._new_position(self.agents_position[i],
                                                      m[1])
                    if m[0] not in valid_starting_directions and \
                       self._path_exists(new_position, m[0],
                                         self.agents_target[i]):
                        valid_starting_directions.append(m[0])

                if len(valid_starting_directions) == 0:
                    re_generate = True
                else:
                    self.agents_direction[i] = random.sample(
                                               valid_starting_directions, 1)[0]

        obs_dict = {}
        for handle in self.agents_handles:
            obs_dict[handle] = self._get_observation_for_agent(handle)
        return obs_dict

    def step(self, action_dict):
        alpha = 1.0
        beta = 1.0

        invalid_action_penalty = -2
        step_penalty = -1 * alpha
        global_reward = 1 * beta

        # Reset the step rewards
        rewards_dict = {}
        for handle in self.agents_handles:
            rewards_dict[handle] = 0

        if self.dones["__all__"]:
            obs_dict = {}
            for handle in self.agents_handles:
                obs_dict[handle] = self._get_observation_for_agent(handle)
            return obs_dict, rewards_dict, self.dones, {}

        for i in range(len(self.agents_handles)):
            handle = self.agents_handles[i]

            if handle not in action_dict:
                continue

            action = action_dict[handle]

            if action < 0 or action > 3:
                print('ERROR: illegal action=', action,
                      'for agent with handle=', handle)
                return

            if action > 0:
                pos = self.agents_position[i]
                direction = self.agents_direction[i]

                movement = direction
                if action == 1:
                    movement = direction - 1
                elif action == 3:
                    movement = direction + 1

                if movement < 0:
                    movement += 4
                if movement >= 4:
                    movement -= 4

                is_deadend = False
                if action == 2:
                    # compute number of possible transitions in the current
                    # cell
                    nbits = 0
                    tmp = self.rail.get_transitions((pos[0], pos[1]))
                    while tmp > 0:
                        nbits += (tmp & 1)
                        tmp = tmp >> 1
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

                new_position = self._new_position(pos, movement)

                # Is it a legal move?  1) transition allows the movement in the
                # cell,  2) the new cell is not empty (case 0),  3) the cell is
                # free, i.e., no agent is currently in that cell
                if new_position[1] >= self.width or\
                   new_position[0] >= self.height or\
                   new_position[0] < 0 or new_position[1] < 0:
                    new_cell_isValid = False

                elif self.rail.get_transitions((new_position[0], new_position[1])) > 0:
                    new_cell_isValid = True
                else:
                    new_cell_isValid = False

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
                    rewards_dict[handle] += invalid_action_penalty

            # if agent is not in target position, add step penalty
            if self.agents_position[i][0] == self.agents_target[i][0] and \
               self.agents_position[i][1] == self.agents_target[i][1]:
                self.dones[handle] = True
            else:
                rewards_dict[handle] += step_penalty

        # Check for end of episode + add global reward to all rewards!
        num_agents_in_target_position = 0
        for i in range(self.number_of_agents):
            if self.agents_position[i][0] == self.agents_target[i][0] and \
               self.agents_position[i][1] == self.agents_target[i][1]:
                num_agents_in_target_position += 1

        if num_agents_in_target_position == self.number_of_agents:
            self.dones["__all__"] = True
            rewards_dict = [r+global_reward for r in rewards_dict]

        # Reset the step actions (in case some agent doesn't 'register_action'
        # on the next step)
        self.actions = [0]*self.number_of_agents

        obs_dict = {}
        for handle in self.agents_handles:
            obs_dict[handle] = self._get_observation_for_agent(handle)

        return obs_dict, rewards_dict, self.dones, {}

    def _new_position(self, position, movement):
        if movement == 0:    # NORTH
            return (position[0]-1, position[1])
        elif movement == 1:  # EAST
            return (position[0], position[1] + 1)
        elif movement == 2:  # SOUTH
            return (position[0]+1, position[1])
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

    def _get_observation_for_agent(self, handle):
        # TODO:
        return None

    def render(self):
        # TODO:
        pass
