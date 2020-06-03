"""
Definition of the RailEnv environment.
"""
import random
# TODO:  _ this is a global method --> utils or remove later
from enum import IntEnum
from typing import List, NamedTuple, Optional, Dict

import msgpack
import msgpack_numpy as m
import numpy as np
from gym.utils import seeding
from msgpack import Packer

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4 import Grid4TransitionsEnum, Grid4Transitions
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import IntVector2D
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import EnvAgent, RailAgentStatus
from flatland.envs.distance_map import DistanceMap

# Need to use circular imports for persistence.
from flatland.envs import malfunction_generators as mal_gen
from flatland.envs import rail_generators as rail_gen
from flatland.envs import schedule_generators as sched_gen
from flatland.envs import persistence

# Direct import of objects / classes does not work with circular imports.
# from flatland.envs.malfunction_generators import no_malfunction_generator, Malfunction, MalfunctionProcessData
# from flatland.envs.observations import GlobalObsForRailEnv
# from flatland.envs.rail_generators import random_rail_generator, RailGenerator
# from flatland.envs.schedule_generators import random_schedule_generator, ScheduleGenerator

from flatland.envs.observations import GlobalObsForRailEnv



import pickle

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


RailEnvGridPos = NamedTuple('RailEnvGridPos', [('r', int), ('c', int)])
RailEnvNextAction = NamedTuple('RailEnvNextAction', [('action', RailEnvActions), ('next_position', RailEnvGridPos),
                                                     ('next_direction', Grid4TransitionsEnum)])


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
    - epsilon = avoid rounding errors
    - stop_penalty = 0  # penalty for stopping a moving agent
    - start_penalty = 0  # penalty for starting a stopped agent

    Stochastic malfunctioning of trains:
    Trains in RailEnv can malfunction if they are halted too often (either by their own choice or because an invalid
    action or cell is selected.

    Every time an agent stops, an agent has a certain probability of malfunctioning. Malfunctions of trains follow a
    poisson process with a certain rate. Not all trains will be affected by malfunctions during episodes to keep
    complexity managable.

    TODO: currently, the parameters that control the stochasticity of the environment are hard-coded in init().
    For Round 2, they will be passed to the constructor as arguments, to allow for more flexibility.

    """
    alpha = 1.0
    beta = 1.0
    # Epsilon to avoid rounding errors
    epsilon = 0.01
    invalid_action_penalty = 0  # previously -2; GIACOMO: we decided that invalid actions will carry no penalty
    step_penalty = -1 * alpha
    global_reward = 1 * beta
    stop_penalty = 0  # penalty for stopping a moving agent
    start_penalty = 0  # penalty for starting a stopped agent

    def __init__(self,
                 width,
                 height,
                 rail_generator = None, 
                 schedule_generator = None, # : sched_gen.ScheduleGenerator = sched_gen.random_schedule_generator(),
                 number_of_agents=1,
                 obs_builder_object: ObservationBuilder = GlobalObsForRailEnv(),
                 malfunction_generator_and_process_data=None, #mal_gen.no_malfunction_generator(),
                 remove_agents_at_target=True,
                 random_seed=1,
                 record_steps=False
                 ):
        """
        Environment init.

        Parameters
        ----------
        rail_generator : function
            The rail_generator function is a function that takes the width,
            height and agents handles of a  rail environment, along with the number of times
            the env has been reset, and returns a GridTransitionMap object and a list of
            starting positions, targets, and initial orientations for agent handle.
            The rail_generator can pass a distance map in the hints or information for specific schedule_generators.
            Implementations can be found in flatland/envs/rail_generators.py
        schedule_generator : function
            The schedule_generator function is a function that takes the grid, the number of agents and optional hints
            and returns a list of starting positions, targets, initial orientations and speed for all agent handles.
            Implementations can be found in flatland/envs/schedule_generators.py
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
        remove_agents_at_target : bool
            If remove_agents_at_target is set to true then the agents will be removed by placing to
            RailEnv.DEPOT_POSITION when the agent has reach it's target position.
        random_seed : int or None
            if None, then its ignored, else the random generators are seeded with this number to ensure
            that stochastic operations are replicable across multiple operations
        """
        super().__init__()

        if malfunction_generator_and_process_data is None:
            malfunction_generator_and_process_data = mal_gen.no_malfunction_generator()
        self.malfunction_generator, self.malfunction_process_data = malfunction_generator_and_process_data
        #self.rail_generator: RailGenerator = rail_generator
        if rail_generator is None:
            rail_generator = rail_gen.random_rail_generator()
        self.rail_generator = rail_generator
        #self.schedule_generator: ScheduleGenerator = schedule_generator
        if schedule_generator is None:
            schedule_generator = sched_gen.random_schedule_generator()
        self.schedule_generator = schedule_generator

        self.rail: Optional[GridTransitionMap] = None
        self.width = width
        self.height = height

        self.remove_agents_at_target = remove_agents_at_target

        self.rewards = [0] * number_of_agents
        self.done = False
        self.obs_builder = obs_builder_object
        self.obs_builder.set_env(self)

        self._max_episode_steps: Optional[int] = None
        self._elapsed_steps = 0

        self.dones = dict.fromkeys(list(range(number_of_agents)) + ["__all__"], False)

        self.obs_dict = {}
        self.rewards_dict = {}
        self.dev_obs_dict = {}
        self.dev_pred_dict = {}

        self.agents: List[EnvAgent] = []
        self.number_of_agents = number_of_agents
        self.num_resets = 0
        self.distance_map = DistanceMap(self.agents, self.height, self.width)

        self.action_space = [5]

        self._seed()
        self._seed()
        self.random_seed = random_seed
        if self.random_seed:
            self._seed(seed=random_seed)

        self.valid_positions = None

        # global numpy array of agents position, True means that there is an agent at that cell
        self.agent_positions: np.ndarray = np.full((height, width), False)

        # save episode timesteps ie agent positions, orientations.  (not yet actions / observations)
        self.record_steps = record_steps  # whether to save timesteps
        # save timesteps in here: [[[row, col, dir, malfunction],...nAgents], ...nSteps]
        self.cur_episode = []  
        self.list_actions = [] # save actions in here

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    # no more agent_handles
    def get_agent_handles(self):
        return range(self.get_num_agents())

    def get_num_agents(self) -> int:
        return len(self.agents)

    def add_agent(self, agent):
        """ Add static info for a single agent.
            Returns the index of the new agent.
        """
        self.agents.append(agent)
        return len(self.agents) - 1

    def set_agent_active(self, agent: EnvAgent):
        if agent.status == RailAgentStatus.READY_TO_DEPART and self.cell_free(agent.initial_position):
            agent.status = RailAgentStatus.ACTIVE
            self._set_agent_to_initial_position(agent, agent.initial_position)

    def reset_agents(self):
        """ Reset the agents to their starting positions
        """
        for agent in self.agents:
            agent.reset()
        self.active_agents = [i for i in range(len(self.agents))]



    def action_required(self, agent):
        """
        Check if an agent needs to provide an action

        Parameters
        ----------
        agent: RailEnvAgent
        Agent we want to check

        Returns
        -------
        True: Agent needs to provide an action
        False: Agent cannot provide an action
        """
        return (agent.status == RailAgentStatus.READY_TO_DEPART or (
            agent.status == RailAgentStatus.ACTIVE and np.isclose(agent.speed_data['position_fraction'], 0.0,
                                                                  rtol=1e-03)))

    def reset(self, regenerate_rail: bool = True, regenerate_schedule: bool = True, activate_agents: bool = False,
              random_seed: bool = None) -> (Dict, Dict):
        """
        reset(regenerate_rail, regenerate_schedule, activate_agents, random_seed)

        The method resets the rail environment

        Parameters
        ----------
        regenerate_rail : bool, optional
            regenerate the rails
        regenerate_schedule : bool, optional
            regenerate the schedule and the static agents
        activate_agents : bool, optional
            activate the agents
        random_seed : bool, optional
            random seed for environment

        Returns
        -------
        observation_dict: Dict
            Dictionary with an observation for each agent
        info_dict: Dict with agent specific information

        """

        if random_seed:
            self._seed(random_seed)

        optionals = {}
        if regenerate_rail or self.rail is None:
            rail, optionals = self.rail_generator(self.width, self.height, self.number_of_agents, self.num_resets,
                                                  self.np_random)

            self.rail = rail
            self.height, self.width = self.rail.grid.shape

            # Do a new set_env call on the obs_builder to ensure
            # that obs_builder specific instantiations are made according to the
            # specifications of the current environment : like width, height, etc
            self.obs_builder.set_env(self)

        if optionals and 'distance_map' in optionals:
            self.distance_map.set(optionals['distance_map'])



        if regenerate_schedule or regenerate_rail or self.get_num_agents() == 0:
            agents_hints = None
            if optionals and 'agents_hints' in optionals:
                agents_hints = optionals['agents_hints']

            schedule = self.schedule_generator(self.rail, self.number_of_agents, agents_hints, self.num_resets,
                                               self.np_random)
            self.agents = EnvAgent.from_schedule(schedule)

            # Get max number of allowed time steps from schedule generator
            # Look at the specific schedule generator used to see where this number comes from
            self._max_episode_steps = schedule.max_episode_steps

        self.agent_positions = np.zeros((self.height, self.width), dtype=int) - 1

        # Reset agents to initial
        self.reset_agents()

        for agent in self.agents:
            # Induce malfunctions
            if activate_agents:
                self.set_agent_active(agent)

            self._break_agent(agent)

            if agent.malfunction_data["malfunction"] > 0:
                agent.speed_data['transition_action_on_cellexit'] = RailEnvActions.DO_NOTHING

            # Fix agents that finished their malfunction
            self._fix_agent_after_malfunction(agent)

        self.num_resets += 1
        self._elapsed_steps = 0

        # TODO perhaps dones should be part of each agent.
        self.dones = dict.fromkeys(list(range(self.get_num_agents())) + ["__all__"], False)

        # Reset the state of the observation builder with the new environment
        self.obs_builder.reset()
        self.distance_map.reset(self.agents, self.rail)

        # Reset the malfunction generator
        self.malfunction_generator(reset=True)

        # Empty the episode store of agent positions
        self.cur_episode = []

        info_dict: Dict = {
            'action_required': {i: self.action_required(agent) for i, agent in enumerate(self.agents)},
            'malfunction': {
                i: agent.malfunction_data['malfunction'] for i, agent in enumerate(self.agents)
            },
            'speed': {i: agent.speed_data['speed'] for i, agent in enumerate(self.agents)},
            'status': {i: agent.status for i, agent in enumerate(self.agents)}
        }
        # Return the new observation vectors for each agent
        observation_dict: Dict = self._get_observations()
        return observation_dict, info_dict

    def _fix_agent_after_malfunction(self, agent: EnvAgent):
        """
        Updates agent malfunction variables and fixes broken agents

        Parameters
        ----------
        agent
        """

        # Ignore agents that are OK
        if self._is_agent_ok(agent):
            return

        # Reduce number of malfunction steps left
        if agent.malfunction_data['malfunction'] > 1:
            agent.malfunction_data['malfunction'] -= 1
            return

        # Restart agents at the end of their malfunction
        agent.malfunction_data['malfunction'] -= 1
        if 'moving_before_malfunction' in agent.malfunction_data:
            agent.moving = agent.malfunction_data['moving_before_malfunction']
            return

    def _break_agent(self, agent: EnvAgent):
        """
        Malfunction generator that breaks agents at a given rate.

        Parameters
        ----------
        agent

        """

        malfunction: Malfunction = self.malfunction_generator(agent, self.np_random)
        if malfunction.num_broken_steps > 0:
            agent.malfunction_data['malfunction'] = malfunction.num_broken_steps
            agent.malfunction_data['moving_before_malfunction'] = agent.moving
            agent.malfunction_data['nr_malfunctions'] += 1

        return

    def step(self, action_dict_: Dict[int, RailEnvActions]):
        """
        Updates rewards for the agents at a step.

        Parameters
        ----------
        action_dict_ : Dict[int,RailEnvActions]

        """
        self._elapsed_steps += 1

        # If we're done, set reward and info_dict and step() is done.
        if self.dones["__all__"]:
            self.rewards_dict = {}
            info_dict = {
                "action_required": {},
                "malfunction": {},
                "speed": {},
                "status": {},
            }
            for i_agent, agent in enumerate(self.agents):
                self.rewards_dict[i_agent] = self.global_reward
                info_dict["action_required"][i_agent] = False
                info_dict["malfunction"][i_agent] = 0
                info_dict["speed"][i_agent] = 0
                info_dict["status"][i_agent] = agent.status

            return self._get_observations(), self.rewards_dict, self.dones, info_dict

        # Reset the step rewards
        self.rewards_dict = dict()
        info_dict = {
            "action_required": {},
            "malfunction": {},
            "speed": {},
            "status": {},
        }
        have_all_agents_ended = True  # boolean flag to check if all agents are done

        for i_agent, agent in enumerate(self.agents):
            # Reset the step rewards
            self.rewards_dict[i_agent] = 0

            # Induce malfunction before we do a step, thus a broken agent can't move in this step
            self._break_agent(agent)

            # Perform step on the agent
            self._step_agent(i_agent, action_dict_.get(i_agent))

            # manage the boolean flag to check if all agents are indeed done (or done_removed)
            have_all_agents_ended &= (agent.status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED])

            # Build info dict
            info_dict["action_required"][i_agent] = self.action_required(agent)
            info_dict["malfunction"][i_agent] = agent.malfunction_data['malfunction']
            info_dict["speed"][i_agent] = agent.speed_data['speed']
            info_dict["status"][i_agent] = agent.status

            # Fix agents that finished their malfunction such that they can perform an action in the next step
            self._fix_agent_after_malfunction(agent)

        # Check for end of episode + set global reward to all rewards!
        if have_all_agents_ended:
            self.dones["__all__"] = True
            self.rewards_dict = {i: self.global_reward for i in range(self.get_num_agents())}
        if (self._max_episode_steps is not None) and (self._elapsed_steps >= self._max_episode_steps):
            self.dones["__all__"] = True
            for i_agent in range(self.get_num_agents()):
                self.dones[i_agent] = True
        if self.record_steps:
            self.record_timestep(action_dict_)

        return self._get_observations(), self.rewards_dict, self.dones, info_dict

    def _step_agent(self, i_agent, action: Optional[RailEnvActions] = None):
        """
        Performs a step and step, start and stop penalty on a single agent in the following sub steps:
        - malfunction
        - action handling if at the beginning of cell
        - movement

        Parameters
        ----------
        i_agent : int
        action_dict_ : Dict[int,RailEnvActions]

        """
        agent = self.agents[i_agent]
        if agent.status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:  # this agent has already completed...
            return

        # agent gets active by a MOVE_* action and if c
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            if action in [RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT,
                          RailEnvActions.MOVE_FORWARD] and self.cell_free(agent.initial_position):
                agent.status = RailAgentStatus.ACTIVE
                self._set_agent_to_initial_position(agent, agent.initial_position)
                self.rewards_dict[i_agent] += self.step_penalty * agent.speed_data['speed']
                return
            else:
                # TODO: Here we need to check for the departure time in future releases with full schedules
                self.rewards_dict[i_agent] += self.step_penalty * agent.speed_data['speed']
                return

        agent.old_direction = agent.direction
        agent.old_position = agent.position

        # if agent is broken, actions are ignored and agent does not move.
        # full step penalty in this case
        if agent.malfunction_data['malfunction'] > 0:
            self.rewards_dict[i_agent] += self.step_penalty * agent.speed_data['speed']
            return

        # Is the agent at the beginning of the cell? Then, it can take an action.
        # As long as the agent is malfunctioning or stopped at the beginning of the cell,
        # different actions may be taken!
        if np.isclose(agent.speed_data['position_fraction'], 0.0, rtol=1e-03):
            # No action has been supplied for this agent -> set DO_NOTHING as default
            if action is None:
                action = RailEnvActions.DO_NOTHING

            if action < 0 or action > len(RailEnvActions):
                print('ERROR: illegal action=', action,
                      'for agent with index=', i_agent,
                      '"DO NOTHING" will be executed instead')
                action = RailEnvActions.DO_NOTHING

            if action == RailEnvActions.DO_NOTHING and agent.moving:
                # Keep moving
                action = RailEnvActions.MOVE_FORWARD

            if action == RailEnvActions.STOP_MOVING and agent.moving:
                # Only allow halting an agent on entering new cells.
                agent.moving = False
                self.rewards_dict[i_agent] += self.stop_penalty

            if not agent.moving and not (
                action == RailEnvActions.DO_NOTHING or
                action == RailEnvActions.STOP_MOVING):
                # Allow agent to start with any forward or direction action
                agent.moving = True
                self.rewards_dict[i_agent] += self.start_penalty

            # Store the action if action is moving
            # If not moving, the action will be stored when the agent starts moving again.
            if agent.moving:
                _action_stored = False
                _, new_cell_valid, new_direction, new_position, transition_valid = \
                    self._check_action_on_agent(action, agent)

                if all([new_cell_valid, transition_valid]):
                    agent.speed_data['transition_action_on_cellexit'] = action
                    _action_stored = True
                else:
                    # But, if the chosen invalid action was LEFT/RIGHT, and the agent is moving,
                    # try to keep moving forward!
                    if (action == RailEnvActions.MOVE_LEFT or action == RailEnvActions.MOVE_RIGHT):
                        _, new_cell_valid, new_direction, new_position, transition_valid = \
                            self._check_action_on_agent(RailEnvActions.MOVE_FORWARD, agent)

                        if all([new_cell_valid, transition_valid]):
                            agent.speed_data['transition_action_on_cellexit'] = RailEnvActions.MOVE_FORWARD
                            _action_stored = True

                if not _action_stored:
                    # If the agent cannot move due to an invalid transition, we set its state to not moving
                    self.rewards_dict[i_agent] += self.invalid_action_penalty
                    self.rewards_dict[i_agent] += self.stop_penalty
                    agent.moving = False

        # Now perform a movement.
        # If agent.moving, increment the position_fraction by the speed of the agent
        # If the new position fraction is >= 1, reset to 0, and perform the stored
        #   transition_action_on_cellexit if the cell is free.
        if agent.moving:
            agent.speed_data['position_fraction'] += agent.speed_data['speed']
            if agent.speed_data['position_fraction'] > 1.0 or np.isclose(agent.speed_data['position_fraction'], 1.0,
                                                                         rtol=1e-03):
                # Perform stored action to transition to the next cell as soon as cell is free
                # Notice that we've already checked new_cell_valid and transition valid when we stored the action,
                # so we only have to check cell_free now!

                # cell and transition validity was checked when we stored transition_action_on_cellexit!
                cell_free, new_cell_valid, new_direction, new_position, transition_valid = self._check_action_on_agent(
                    agent.speed_data['transition_action_on_cellexit'], agent)

                # N.B. validity of new_cell and transition should have been verified before the action was stored!
                assert new_cell_valid
                assert transition_valid
                if cell_free:
                    self._move_agent_to_new_position(agent, new_position)
                    agent.direction = new_direction
                    agent.speed_data['position_fraction'] = 0.0

            # has the agent reached its target?
            if np.equal(agent.position, agent.target).all():
                agent.status = RailAgentStatus.DONE
                self.dones[i_agent] = True
                self.active_agents.remove(i_agent)
                agent.moving = False
                self._remove_agent_from_scene(agent)
            else:
                self.rewards_dict[i_agent] += self.step_penalty * agent.speed_data['speed']
        else:
            # step penalty if not moving (stopped now or before)
            self.rewards_dict[i_agent] += self.step_penalty * agent.speed_data['speed']

    def _set_agent_to_initial_position(self, agent: EnvAgent, new_position: IntVector2D):
        """
        Sets the agent to its initial position. Updates the agent object and the position
        of the agent inside the global agent_position numpy array

        Parameters
        -------
        agent: EnvAgent object
        new_position: IntVector2D
        """
        agent.position = new_position
        self.agent_positions[agent.position] = agent.handle

    def _move_agent_to_new_position(self, agent: EnvAgent, new_position: IntVector2D):
        """
        Move the agent to the a new position. Updates the agent object and the position
        of the agent inside the global agent_position numpy array

        Parameters
        -------
        agent: EnvAgent object
        new_position: IntVector2D
        """
        agent.position = new_position
        self.agent_positions[agent.old_position] = -1
        self.agent_positions[agent.position] = agent.handle

    def _remove_agent_from_scene(self, agent: EnvAgent):
        """
        Remove the agent from the scene. Updates the agent object and the position
        of the agent inside the global agent_position numpy array

        Parameters
        -------
        agent: EnvAgent object
        """
        self.agent_positions[agent.position] = -1
        if self.remove_agents_at_target:
            agent.position = None
            agent.status = RailAgentStatus.DONE_REMOVED

    def _check_action_on_agent(self, action: RailEnvActions, agent: EnvAgent):
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
        new_direction, transition_valid = self.check_action(agent, action)
        new_position = get_new_position(agent.position, new_direction)

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

        # only call cell_free() if new cell is inside the scene
        if new_cell_valid:
            # Check the new position is not the same as any of the existing agent positions
            # (including itself, for simplicity, since it is moving)
            cell_free = self.cell_free(new_position)
        else:
            # if new cell is outside of scene -> cell_free is False
            cell_free = False
        return cell_free, new_cell_valid, new_direction, new_position, transition_valid

    def record_timestep(self, dActions):
        ''' Record the positions and orientations of all agents in memory, in the cur_episode
        '''
        list_agents_state = []
        for i_agent in range(self.get_num_agents()):
            agent = self.agents[i_agent]
            # the int cast is to avoid numpy types which may cause problems with msgpack
            # in env v2, agents may have position None, before starting
            if agent.position is None:
                pos = (0, 0)
            else:
                pos = (int(agent.position[0]), int(agent.position[1]))
            # print("pos:", pos, type(pos[0]))
            list_agents_state.append(
                [*pos, int(agent.direction), agent.malfunction_data["malfunction"] ])

        self.cur_episode.append(list_agents_state)
        self.list_actions.append(dActions)

    def cell_free(self, position: IntVector2D) -> bool:
        """
        Utility to check if a cell is free

        Parameters:
        --------
        position : Tuple[int, int]

        Returns
        -------
        bool
            is the cell free or not?

        """
        return self.agent_positions[position] == -1

    def check_action(self, agent: EnvAgent, action: RailEnvActions):
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

        if action == RailEnvActions.MOVE_FORWARD and num_transitions == 1:
            # - dead-end, straight line or curved line;
            # new_direction will be the only valid transition
            # - take only available transition
            new_direction = np.argmax(possible_transitions)
            transition_valid = True
        return new_direction, transition_valid

    def _get_observations(self):
        """
        Utility which returns the observations for an agent with respect to environment

        Returns
        ------
        Dict object
        """
        #print(f"_get_obs - num agents: {self.get_num_agents()} {list(range(self.get_num_agents()))}")
        self.obs_dict = self.obs_builder.get_many(list(range(self.get_num_agents())))
        return self.obs_dict

    def get_valid_directions_on_grid(self, row: int, col: int) -> List[int]:
        """
        Returns directions in which the agent can move

        Parameters:
        ---------
        row : int
        col : int

        Returns:
        -------
        List[int]
        """
        return Grid4Transitions.get_entry_directions(self.rail.get_full_transitions(row, col))



    def _exp_distirbution_synced(self, rate: float) -> float:
        """
        Generates sample from exponential distribution
        We need this to guarantee synchronity between different instances with same seed.
        :param rate:
        :return:
        """
        u = self.np_random.rand()
        x = - np.log(1 - u) * rate
        return x

    def _is_agent_ok(self, agent: EnvAgent) -> bool:
        """
        Check if an agent is ok, meaning it can move and is not malfuncitoinig
        Parameters
        ----------
        agent

        Returns
        -------
        True if agent is ok, False otherwise

        """
        return agent.malfunction_data['malfunction'] < 1

    def save(self, filename):
        print("deprecated call to env.save() - pls call RailEnvPersister.save()")
        persistence.RailEnvPersister.save(self, filename)


