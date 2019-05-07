"""
Definition of the RailEnv environment and related level-generation functions.

Generator functions are functions that take width, height and num_resets as arguments and return
a GridTransitionMap object.
"""
import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import TreeObsForRailEnv
from flatland.envs.generators import random_rail_generator
from flatland.envs.env_utils import get_new_position

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

    def reset(self, regen_rail=True, replace_agents=True):
        """
        TODO: replace_agents is ignored at the moment; agents will always be replaced.
        """
        if regen_rail or self.rail is None:
            self.rail, self.agents_position, self.agents_direction, self.agents_target = self.rail_generator(
                self.width,
                self.height,
                self.agents_handles,
                self.num_resets)

        self.num_resets += 1

        self.dones = {"__all__": False}
        for handle in self.agents_handles:
            self.dones[handle] = False

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

                possible_transitions = self.rail.get_transitions((pos[0], pos[1], direction))
                num_transitions = np.count_nonzero(possible_transitions)

                movement = direction
                # print(nbits,np.sum(possible_transitions))
                if action == 1:
                    movement = direction - 1
                    if num_transitions <= 1:
                        transition_isValid = False

                elif action == 3:
                    movement = direction + 1
                    if num_transitions <= 1:
                        transition_isValid = False

                if movement < 0:
                    movement += 4
                if movement >= 4:
                    movement -= 4

                if action == 2:
                    if num_transitions == 1:
                        # - dead-end, straight line or curved line;
                        # movement will be the only valid transition
                        # - take only available transition
                        movement = np.argmax(possible_transitions)
                        transition_isValid = True

                new_position = get_new_position(pos, movement)
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
                        movement)

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

    def _get_observations(self):
        self.obs_dict = {}
        for handle in self.agents_handles:
            self.obs_dict[handle] = self.obs_builder.get(handle)
        return self.obs_dict

    def render(self):
        # TODO:
        pass
