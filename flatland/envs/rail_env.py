"""
Definition of the RailEnv environment.
"""
# TODO:  _ this is a global method --> utils or remove later

from enum import IntEnum

import msgpack
import msgpack_numpy as m
import numpy as np

from flatland.core.env import Environment
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import EnvAgentStatic, EnvAgent
from flatland.envs.generators import random_rail_generator
from flatland.envs.observations import TreeObsForRailEnv

m.patch()


class RailEnvActions(IntEnum):
    DO_NOTHING = 0  # implies change of direction in a dead-end!
    MOVE_LEFT = 1
    MOVE_FORWARD = 2
    MOVE_RIGHT = 3
    STOP_MOVING = 4

    @staticmethod
    def to_char(a: int):
        return {
            0: 'B',
            1: 'L',
            2: 'F',
            3: 'R',
            4: 'S',
        }[a]


class RailEnv(Environment):
    """
    RailEnv environment class.

    RailEnv is an environment inspired by a (simplified version of) a rail
    network, in which agents (trains) have to navigate to their target
    locations in the shortest time possible, while at the same time cooperating
    to avoid bottlenecks.

    The valid actions in the environment are:

     -   0: do nothing (continue moving or stay still)
     -   1: turn left at switch and move to the next cell; if the agent was not moving, movement is started
     -   2: move to the next cell in front of the agent; if the agent was not moving, movement is started
     -   3: turn right at switch and move to the next cell; if the agent was not moving, movement is started
     -   4: stop moving

    Moving forward in a dead-end cell makes the agent turn 180 degrees and step
    to the cell it came from.


    The actions of the agents are executed in order of their handle to prevent
    deadlocks and to allow them to learn relative priorities.

    Reward Function:

    It costs each agent a step_penalty for every time-step taken in the environment. Independent of the movement
    of the agent. Currently all other penalties such as penalty for stopping, starting and invalid actions are set to 0.

    alpha = 1
    beta = 1
    Reward function parameters:

    - invalid_action_penalty = 0
    - step_penalty = -alpha
    - global_reward = beta
    - stop_penalty = 0  # penalty for stopping a moving agent
    - start_penalty = 0  # penalty for starting a stopped agent

    """

    def __init__(self,
                 width,
                 height,
                 rail_generator=random_rail_generator(),
                 number_of_agents=1,
                 obs_builder_object=TreeObsForRailEnv(max_depth=2),
                 max_episode_steps=None
                 ):
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
                rail_from_grid_transition_map(rail_map) : generate a rail from
                                        a GridTransitionMap object
                rail_from_manual_sp ecifications_generator(rail_spec) : generate a rail from
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
        max_episode_steps : int or None

        file_name: you can load a pickle file.
        """

        self.rail_generator = rail_generator
        self.rail = None
        self.width = width
        self.height = height

        self.rewards = [0] * number_of_agents
        self.done = False
        self.obs_builder = obs_builder_object
        self.obs_builder._set_env(self)

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

        self.dones = dict.fromkeys(list(range(number_of_agents)) + ["__all__"], False)

        self.obs_dict = {}
        self.rewards_dict = {}
        self.dev_obs_dict = {}
        self.dev_pred_dict = {}

        self.agents = [None] * number_of_agents  # live agents
        self.agents_static = [None] * number_of_agents  # static agent information
        self.num_resets = 0

        self.action_space = [1]
        self.observation_space = self.obs_builder.observation_space  # updated on resets?

        self.reset()
        self.num_resets = 0  # yes, set it to zero again!

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

        # Check if generator provided a distance map TODO: Make this check safer!
        if len(tRailAgents) > 5:
            self.obs_builder.distance_map = tRailAgents[-1]

        if regen_rail or self.rail is None:
            self.rail = tRailAgents[0]
            self.height, self.width = self.rail.grid.shape

        if replace_agents:
            self.agents_static = EnvAgentStatic.from_lists(*tRailAgents[1:5])

        self.restart_agents()

        for i_agent in range(self.get_num_agents()):
            agent = self.agents[i_agent]
            agent.speed_data['position_fraction'] = 0.0

        self.num_resets += 1
        self._elapsed_steps = 0

        # TODO perhaps dones should be part of each agent.
        self.dones = dict.fromkeys(list(range(self.get_num_agents())) + ["__all__"], False)

        # Reset the state of the observation builder with the new environment
        self.obs_builder.reset()
        self.observation_space = self.obs_builder.observation_space  # <-- change on reset?

        # Return the new observation vectors for each agent
        return self._get_observations()

    def step(self, action_dict_):
        self._elapsed_steps += 1

        action_dict = action_dict_.copy()

        alpha = 1.0
        beta = 1.0

        invalid_action_penalty = 0  # previously -2; GIACOMO: we decided that invalid actions will carry no penalty
        step_penalty = -1 * alpha
        global_reward = 1 * beta
        stop_penalty = 0  # penalty for stopping a moving agent
        start_penalty = 0  # penalty for starting a stopped agent

        # Reset the step rewards
        self.rewards_dict = dict()
        for i_agent in range(self.get_num_agents()):
            self.rewards_dict[i_agent] = 0

        if self.dones["__all__"]:
            self.rewards_dict = {i: r + global_reward for i, r in self.rewards_dict.items()}
            return self._get_observations(), self.rewards_dict, self.dones, {}

        # for i in range(len(self.agents_handles)):
        for i_agent in range(self.get_num_agents()):
            agent = self.agents[i_agent]
            agent.old_direction = agent.direction
            agent.old_position = agent.position
            if self.dones[i_agent]:  # this agent has already completed...
                continue

            if i_agent not in action_dict:  # no action has been supplied for this agent
                action_dict[i_agent] = RailEnvActions.DO_NOTHING

            if action_dict[i_agent] < 0 or action_dict[i_agent] > len(RailEnvActions):
                print('ERROR: illegal action=', action_dict[i_agent],
                      'for agent with index=', i_agent,
                      '"DO NOTHING" will be executed instead')
                action_dict[i_agent] = RailEnvActions.DO_NOTHING

            action = action_dict[i_agent]

            if action == RailEnvActions.DO_NOTHING and agent.moving:
                # Keep moving
                action = RailEnvActions.MOVE_FORWARD

            if action == RailEnvActions.STOP_MOVING and agent.moving and agent.speed_data['position_fraction'] == 0.:
                # Only allow halting an agent on entering new cells.
                agent.moving = False
                self.rewards_dict[i_agent] += stop_penalty

            if not agent.moving and not (action == RailEnvActions.DO_NOTHING or action == RailEnvActions.STOP_MOVING):
                # Allow agent to start with any forward or direction action
                agent.moving = True
                self.rewards_dict[i_agent] += start_penalty

            # Now perform a movement.
            # If the agent is in an initial position within a new cell (agent.speed_data['position_fraction']<eps)
            #   store the desired action in `transition_action_on_cellexit' (only if the desired transition is
            #   allowed! otherwise DO_NOTHING!)
            # Then in any case (if agent.moving) and the `transition_action_on_cellexit' is valid, increment the
            #   position_fraction by the speed of the agent   (regardless of action taken, as long as no
            #   STOP_MOVING, but that makes agent.moving=False)
            # If the new position fraction is >= 1, reset to 0, and perform the stored
            #   transition_action_on_cellexit

            # If the agent can make an action
            action_selected = False
            if agent.speed_data['position_fraction'] == 0.:
                if action != RailEnvActions.DO_NOTHING and action != RailEnvActions.STOP_MOVING:
                    cell_free, new_cell_valid, new_direction, new_position, transition_valid = \
                        self._check_action_on_agent(action, agent)

                    if all([new_cell_valid, transition_valid]):
                        agent.speed_data['transition_action_on_cellexit'] = action
                        action_selected = True

                    else:
                        # But, if the chosen invalid action was LEFT/RIGHT, and the agent is moving,
                        # try to keep moving forward!
                        if (action == RailEnvActions.MOVE_LEFT or action == RailEnvActions.MOVE_RIGHT) and agent.moving:
                            cell_free, new_cell_valid, new_direction, new_position, transition_valid = \
                                self._check_action_on_agent(RailEnvActions.MOVE_FORWARD, agent)

                            if all([new_cell_valid, transition_valid]):
                                agent.speed_data['transition_action_on_cellexit'] = RailEnvActions.MOVE_FORWARD
                                action_selected = True

                            else:
                                # TODO: an invalid action was chosen after entering the cell. The agent cannot move.
                                self.rewards_dict[i_agent] += invalid_action_penalty
                                self.rewards_dict[i_agent] += step_penalty * agent.speed_data['speed']
                                self.rewards_dict[i_agent] += stop_penalty
                                agent.moving = False
                                continue
                        else:
                            # TODO: an invalid action was chosen after entering the cell. The agent cannot move.
                            self.rewards_dict[i_agent] += invalid_action_penalty
                            self.rewards_dict[i_agent] += step_penalty * agent.speed_data['speed']
                            self.rewards_dict[i_agent] += stop_penalty
                            agent.moving = False
                            continue

            if agent.moving and (action_selected or agent.speed_data['position_fraction'] > 0.0):
                agent.speed_data['position_fraction'] += agent.speed_data['speed']

            if agent.speed_data['position_fraction'] >= 1.0:

                # Perform stored action to transition to the next cell

                cell_free, new_cell_valid, new_direction, new_position, transition_valid = \
                    self._check_action_on_agent(agent.speed_data['transition_action_on_cellexit'], agent)

                # Check that everything is still fee and that the agent can move
                if all([new_cell_valid, transition_valid, cell_free]):
                    agent.position = new_position
                    agent.direction = new_direction
                    agent.speed_data['position_fraction'] = 0.0
                else:
                    # If the agent cannot move due to any reason, we set its state to not moving.
                    agent.moving = False

            if np.equal(agent.position, agent.target).all():
                self.dones[i_agent] = True
                agent.moving = False
            else:
                self.rewards_dict[i_agent] += step_penalty * agent.speed_data['speed']

        # Check for end of episode + add global reward to all rewards!
        if np.all([np.array_equal(agent2.position, agent2.target) for agent2 in self.agents]):
            self.dones["__all__"] = True
            self.rewards_dict = {i: 0 * r + global_reward for i, r in self.rewards_dict.items()}

        if (self._max_episode_steps is not None) and (self._elapsed_steps >= self._max_episode_steps):
            self.dones["__all__"] = True
            for k in self.dones.keys():
                self.dones[k] = True

        return self._get_observations(), self.rewards_dict, self.dones, {}

    def _check_action_on_agent(self, action, agent):
        # compute number of possible transitions in the current
        # cell used to check for invalid actions
        new_direction, transition_valid = self.check_action(agent, action)
        new_position = get_new_position(agent.position, new_direction)

        # Is it a legal move?
        # 1) transition allows the new_direction in the cell,
        # 2) the new cell is not empty (case 0),
        # 3) the cell is free, i.e., no agent is currently in that cell
        new_cell_valid = (
            np.array_equal(  # Check the new position is still in the grid
                new_position,
                np.clip(new_position, [0, 0], [self.height - 1, self.width - 1]))
            and  # check the new position has some transitions (ie is not an empty cell)
            self.rail.get_full_transitions(*new_position) > 0)

        # If transition validity hasn't been checked yet.
        if transition_valid is None:
            transition_valid = self.rail.get_transition(
                (*agent.position, agent.direction),
                new_direction)

        # Check the new position is not the same as any of the existing agent positions
        # (including itself, for simplicity, since it is moving)
        cell_free = not np.any(
            np.equal(new_position, [agent2.position for agent2 in self.agents]).all(1))
        return cell_free, new_cell_valid, new_direction, new_position, transition_valid

    def check_action(self, agent, action):
        transition_valid = None
        possible_transitions = self.rail.get_transitions(*agent.position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        new_direction = agent.direction
        if action == RailEnvActions.MOVE_LEFT:
            new_direction = agent.direction - 1
            if num_transitions <= 1:
                transition_valid = False

        elif action == RailEnvActions.MOVE_RIGHT:
            new_direction = agent.direction + 1
            if num_transitions <= 1:
                transition_valid = False

        new_direction %= 4

        if action == RailEnvActions.MOVE_FORWARD:
            if num_transitions == 1:
                # - dead-end, straight line or curved line;
                # new_direction will be the only valid transition
                # - take only available transition
                new_direction = np.argmax(possible_transitions)
                transition_valid = True
        return new_direction, transition_valid

    def _get_observations(self):
        self.obs_dict = self.obs_builder.get_many(list(range(self.get_num_agents())))
        return self.obs_dict

    def get_full_state_msg(self):
        grid_data = self.rail.grid.tolist()
        agent_static_data = [agent.to_list() for agent in self.agents_static]
        agent_data = [agent.to_list() for agent in self.agents]

        msgpack.packb(grid_data)
        msgpack.packb(agent_data)
        msgpack.packb(agent_static_data)

        msg_data = {
            "grid": grid_data,
            "agents_static": agent_static_data,
            "agents": agent_data}
        return msgpack.packb(msg_data, use_bin_type=True)

    def get_agent_state_msg(self):
        agent_data = [agent.to_list() for agent in self.agents]
        msg_data = {
            "agents": agent_data}
        return msgpack.packb(msg_data, use_bin_type=True)

    def set_full_state_msg(self, msg_data):
        data = msgpack.unpackb(msg_data, use_list=False)
        self.rail.grid = np.array(data[b"grid"])
        # agents are always reset as not moving
        self.agents_static = [EnvAgentStatic(d[0], d[1], d[2], moving=False) for d in data[b"agents_static"]]
        self.agents = [EnvAgent(d[0], d[1], d[2], d[3], d[4]) for d in data[b"agents"]]
        # setup with loaded data
        self.height, self.width = self.rail.grid.shape
        self.rail.height = self.height
        self.rail.width = self.width
        self.dones = dict.fromkeys(list(range(self.get_num_agents())) + ["__all__"], False)

    def set_full_state_dist_msg(self, msg_data):
        data = msgpack.unpackb(msg_data, use_list=False)
        self.rail.grid = np.array(data[b"grid"])
        # agents are always reset as not moving
        self.agents_static = [EnvAgentStatic(d[0], d[1], d[2], moving=False) for d in data[b"agents_static"]]
        self.agents = [EnvAgent(d[0], d[1], d[2], d[3], d[4]) for d in data[b"agents"]]
        if hasattr(self.obs_builder, 'distance_map') and b"distance_maps" in data.keys():
            self.obs_builder.distance_map = data[b"distance_maps"]
        # setup with loaded data
        self.height, self.width = self.rail.grid.shape
        self.rail.height = self.height
        self.rail.width = self.width
        self.dones = dict.fromkeys(list(range(self.get_num_agents())) + ["__all__"], False)

    def get_full_state_dist_msg(self):
        grid_data = self.rail.grid.tolist()
        agent_static_data = [agent.to_list() for agent in self.agents_static]
        agent_data = [agent.to_list() for agent in self.agents]

        msgpack.packb(grid_data)
        msgpack.packb(agent_data)
        msgpack.packb(agent_static_data)
        if hasattr(self.obs_builder, 'distance_map'):
            distance_map_data = self.obs_builder.distance_map
            msgpack.packb(distance_map_data)
            msg_data = {
                "grid": grid_data,
                "agents_static": agent_static_data,
                "agents": agent_data,
                "distance_maps": distance_map_data}
        else:
            msg_data = {
                "grid": grid_data,
                "agents_static": agent_static_data,
                "agents": agent_data}

        return msgpack.packb(msg_data, use_bin_type=True)

    def save(self, filename):
        if hasattr(self.obs_builder, 'distance_map'):
            if len(self.obs_builder.distance_map) > 0:
                with open(filename, "wb") as file_out:
                    file_out.write(self.get_full_state_dist_msg())
            else:
                with open(filename, "wb") as file_out:
                    file_out.write(self.get_full_state_msg())
        else:
            with open(filename, "wb") as file_out:
                file_out.write(self.get_full_state_msg())

    def load(self, filename):
        if hasattr(self.obs_builder, 'distance_map'):
            with open(filename, "rb") as file_in:
                load_data = file_in.read()
                self.set_full_state_dist_msg(load_data)
        else:
            with open(filename, "rb") as file_in:
                load_data = file_in.read()
                self.set_full_state_msg(load_data)

    def load_pkl(self, pkl_data):
        self.set_full_state_msg(pkl_data)

    def load_resource(self, package, resource):
        from importlib_resources import read_binary
        load_data = read_binary(package, resource)
        self.set_full_state_msg(load_data)
