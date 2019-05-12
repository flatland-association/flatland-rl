"""
Definition of the RailEnv environment and related level-generation functions.

Generator functions are functions that take width, height and num_resets as arguments and return
a GridTransitionMap object.
"""
import numpy as np
import pickle

from flatland.core.env import Environment
from flatland.core.env_observation_builder import TreeObsForRailEnv
from flatland.envs.generators import random_rail_generator
from flatland.envs.env_utils import get_new_position
from flatland.envs.agent_utils import EnvAgentStatic, EnvAgent

# from flatland.core.transitions import Grid8Transitions, RailEnvTransitions
# from flatland.core.transition_map import GridTransitionMap


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
            The rail_generator function is a function that takes the width,
            height and agents handles of a  rail environment, along with the number of times
            the env has been reset, and returns a GridTransitionMap object and a list of
            starting positions, targets, and initial orientations for agent handle.
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

        # use get_num_agents() instead
        # self.number_of_agents = number_of_agents

        self.obs_builder = obs_builder_object
        self.obs_builder._set_env(self)

        self.actions = [0] * number_of_agents
        self.rewards = [0] * number_of_agents
        self.done = False

        self.dones = dict.fromkeys(list(range(number_of_agents)) + ["__all__"], False)

        self.obs_dict = {}
        self.rewards_dict = {}

        # self.agents_handles = list(range(self.number_of_agents))

        # self.agents_position = []
        # self.agents_target = []
        # self.agents_direction = []
        self.agents = [None] * number_of_agents  # live agents
        self.agents_static = [None] * number_of_agents  # static agent information
        self.num_resets = 0
        self.reset()
        self.num_resets = 0   # yes, set it to zero again!

        self.valid_positions = None

    # no more agent_handles
    def get_agent_handles(self):
        return range(self.get_num_agents())

    def get_num_agents(self, static=True):
        if static:
            return len(self.agents_static)
        else:
            return len(self.agents)

    def add_agent_static(self, agent_static):
        """ Add static info for a single agent.
            Returns the index of the new agent.
        """
        self.agents_static.append(agent_static)
        return len(self.agents_static) - 1

    def restart_agents(self):
        """ Reset the agents to their starting positions defined in agents_static
        """
        self.agents = EnvAgent.list_from_static(self.agents_static)

    def reset(self, regen_rail=True, replace_agents=True):
        """ if regen_rail then regenerate the rails.
            if replace_agents then regenerate the agents static.
            Relies on the rail_generator returning agent_static lists (pos, dir, target)
        """
        tRailAgents = self.rail_generator(self.width, self.height, self.get_num_agents(), self.num_resets)

        if regen_rail or self.rail is None:
            self.rail = tRailAgents[0]

        if replace_agents:
            self.agents_static = EnvAgentStatic.from_lists(*tRailAgents[1:4])

        # Take the agent static info and put (live) agents at the start positions
        # self.agents = EnvAgent.list_from_static(self.agents_static[:len(self.agents_handles)])
        self.restart_agents()

        self.num_resets += 1

        # for handle in self.agents_handles:
        #    self.dones[handle] = False
        self.dones = dict.fromkeys(list(range(self.get_num_agents())) + ["__all__"], False)
        # perhaps dones should be part of each agent.
        
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
        # for handle in self.agents_handles:
        #    self.rewards_dict[handle] = 0
        for iAgent in range(self.get_num_agents()):
            self.rewards_dict[iAgent] = 0

        if self.dones["__all__"]:
            return self._get_observations(), self.rewards_dict, self.dones, {}

        # for i in range(len(self.agents_handles)):
        for iAgent in range(self.get_num_agents()):
            # handle = self.agents_handles[i]
            transition_isValid = None
            agent = self.agents[iAgent]

            if iAgent not in action_dict:  # no action has been supplied for this agent
                continue

            if self.dones[iAgent]:  # this agent has already completed...
                continue
            action = action_dict[iAgent]

            if action < 0 or action > 3:
                print('ERROR: illegal action=', action,
                      'for agent with index=', iAgent)
                return

            if action > 0:
                # pos = agent.position #  self.agents_position[i]
                # direction = agent.direction # self.agents_direction[i]

                # compute number of possible transitions in the current
                # cell used to check for invalid actions

                possible_transitions = self.rail.get_transitions((*agent.position, agent.direction))
                num_transitions = np.count_nonzero(possible_transitions)

                movement = agent.direction
                # print(nbits,np.sum(possible_transitions))
                if action == 1:
                    movement = agent.direction - 1
                    if num_transitions <= 1:
                        transition_isValid = False

                elif action == 3:
                    movement = agent.direction + 1
                    if num_transitions <= 1:
                        transition_isValid = False

                movement %= 4

                if action == 2:
                    if num_transitions == 1:
                        # - dead-end, straight line or curved line;
                        # movement will be the only valid transition
                        # - take only available transition
                        movement = np.argmax(possible_transitions)
                        transition_isValid = True

                new_position = get_new_position(agent.position, movement)
                # Is it a legal move?
                # 1) transition allows the movement in the cell,
                # 2) the new cell is not empty (case 0),
                # 3) the cell is free, i.e., no agent is currently in that cell
                
                # if (
                #        new_position[1] >= self.width or
                #        new_position[0] >= self.height or
                #        new_position[0] < 0 or new_position[1] < 0):
                #    new_cell_isValid = False

                # if self.rail.get_transitions(new_position) == 0:
                #     new_cell_isValid = False

                new_cell_isValid = (
                        np.array_equal(  # Check the new position is still in the grid
                            new_position,
                            np.clip(new_position, [0, 0], [self.height-1, self.width-1]))
                        and  # check the new position has some transitions (ie is not an empty cell)
                        self.rail.get_transitions(new_position) > 0)

                # If transition validity hasn't been checked yet.
                if transition_isValid is None:
                    transition_isValid = self.rail.get_transition(
                        (*agent.position, agent.direction),
                        movement)

                # cell_isFree = True
                # for j in range(self.number_of_agents):
                #    if self.agents_position[j] == new_position:
                #        cell_isFree = False
                #        break
                # Check the new position is not the same as any of the existing agent positions
                # (including itself, for simplicity, since it is moving)
                cell_isFree = not np.any(
                        np.equal(new_position, [agent2.position for agent2 in self.agents]).all(1))

                if all([new_cell_isValid, transition_isValid, cell_isFree]):
                    # move and change direction to face the movement that was
                    # performed
                    # self.agents_position[i] = new_position
                    # self.agents_direction[i] = movement
                    agent.position = new_position
                    agent.direction = movement
                else:
                    # the action was not valid, add penalty
                    self.rewards_dict[iAgent] += invalid_action_penalty

            # if agent is not in target position, add step penalty
            # if self.agents_position[i][0] == self.agents_target[i][0] and \
            #        self.agents_position[i][1] == self.agents_target[i][1]:
            #    self.dones[handle] = True
            if np.equal(agent.position, agent.target).all():
                self.dones[iAgent] = True
            else:
                self.rewards_dict[iAgent] += step_penalty

        # Check for end of episode + add global reward to all rewards!
        # num_agents_in_target_position = 0
        # for i in range(self.number_of_agents):
        #    if self.agents_position[i][0] == self.agents_target[i][0] and \
        #            self.agents_position[i][1] == self.agents_target[i][1]:
        #        num_agents_in_target_position += 1
        # if num_agents_in_target_position == self.number_of_agents:
        if np.all([np.array_equal(agent2.position, agent2.target) for agent2 in self.agents]):
            self.dones["__all__"] = True
            self.rewards_dict = [r + global_reward for r in self.rewards_dict]

        # Reset the step actions (in case some agent doesn't 'register_action'
        # on the next step)
        self.actions = [0] * self.get_num_agents()
        return self._get_observations(), self.rewards_dict, self.dones, {}

    def _get_observations(self):
        self.obs_dict = {}
        # for handle in self.agents_handles:
        for iAgent in range(self.get_num_agents()):
            self.obs_dict[iAgent] = self.obs_builder.get(iAgent)
        return self.obs_dict

    def render(self):
        # TODO:
        pass

    def save(self, sFilename):
        dSave = {
            "grid": self.rail.grid,
            "agents_static": self.agents_static
            }
        with open(sFilename, "wb") as fOut:
            pickle.dump(dSave, fOut)

    def load(self, sFilename):
        with open(sFilename, "rb") as fIn:
            dLoad = pickle.load(fIn)
            self.rail.grid = dLoad["grid"]
            self.height, self.width = self.rail.grid.shape
            self.agents_static = dLoad["agents_static"]
            self.agents = [None] * self.get_num_agents()
            self.dones = dict.fromkeys(list(range(self.get_num_agents())) + ["__all__"], False)
            
