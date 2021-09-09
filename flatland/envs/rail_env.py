"""
Definition of the RailEnv environment.
"""
import random
# TODO:  _ this is a global method --> utils or remove later
from enum import IntEnum
from typing import List, NamedTuple, Optional, Dict, Tuple

import numpy as np
from numpy.lib.shape_base import vsplit
from numpy.testing._private.utils import import_nose


from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4 import Grid4TransitionsEnum, Grid4Transitions
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import IntVector2D, position_to_coordinate
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import Agent, EnvAgent
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env_action import RailEnvActions

# Need to use circular imports for persistence.
from flatland.envs import malfunction_generators as mal_gen
from flatland.envs import rail_generators as rail_gen
from flatland.envs import line_generators as line_gen
from flatland.envs.timetable_generators import timetable_generator
from flatland.envs import persistence
from flatland.envs import agent_chains as ac

from flatland.envs.observations import GlobalObsForRailEnv
from gym.utils import seeding

# Direct import of objects / classes does not work with circular imports.
# from flatland.envs.malfunction_generators import no_malfunction_generator, Malfunction, MalfunctionProcessData
# from flatland.envs.observations import GlobalObsForRailEnv
# from flatland.envs.rail_generators import random_rail_generator, RailGenerator
# from flatland.envs.line_generators import random_line_generator, LineGenerator


from flatland.envs.timetable_generators import timetable_generator
from flatland.envs.step_utils.states import TrainState
from flatland.envs.step_utils.transition_utils import check_action

# Env Step Facelift imports
from flatland.envs.step_utils.action_preprocessing import preprocess_raw_action, preprocess_moving_action, preprocess_action_when_waiting

# Adrian Egli performance fix (the fast methods brings more than 50%)
def fast_isclose(a, b, rtol):
    return (a < (b + rtol)) or (a < (b - rtol))

def fast_position_equal(pos_1: (int, int), pos_2: (int, int)) -> bool:
    if pos_1 is None: # TODO: Dipam - Consider making default of agent.position as (-1, -1) instead of None
        return False
    else:
        return pos_1[0] == pos_2[0] and pos_1[1] == pos_2[1]

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
    # Epsilon to avoid rounding errors
    epsilon = 0.01
    # NEW : REW: Sparse Reward
    alpha = 0
    beta = 0
    step_penalty = -1 * alpha
    global_reward = 1 * beta
    invalid_action_penalty = 0  # previously -2; GIACOMO: we decided that invalid actions will carry no penalty
    stop_penalty = 0  # penalty for stopping a moving agent
    start_penalty = 0  # penalty for starting a stopped agent
    cancellation_factor = 1
    cancellation_time_buffer = 0

    def __init__(self,
                 width,
                 height,
                 rail_generator=None,
                 line_generator=None,  # : line_gen.LineGenerator = line_gen.random_line_generator(),
                 number_of_agents=2,
                 obs_builder_object: ObservationBuilder = GlobalObsForRailEnv(),
                 malfunction_generator_and_process_data=None,  # mal_gen.no_malfunction_generator(),
                 malfunction_generator=None,
                 remove_agents_at_target=True,
                 random_seed=1,
                 record_steps=False,
                 close_following=True
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
            The rail_generator can pass a distance map in the hints or information for specific line_generators.
            Implementations can be found in flatland/envs/rail_generators.py
        line_generator : function
            The line_generator function is a function that takes the grid, the number of agents and optional hints
            and returns a list of starting positions, targets, initial orientations and speed for all agent handles.
            Implementations can be found in flatland/envs/line_generators.py
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

        if malfunction_generator_and_process_data is not None:
            print("DEPRECATED - RailEnv arg: malfunction_and_process_data - use malfunction_generator")
            self.malfunction_generator, self.malfunction_process_data = malfunction_generator_and_process_data
        elif malfunction_generator is not None:
            self.malfunction_generator = malfunction_generator
            # malfunction_process_data is not used
            # self.malfunction_generator, self.malfunction_process_data = malfunction_generator_and_process_data
            self.malfunction_process_data = self.malfunction_generator.get_process_data()
        # replace default values here because we can't use default args values because of cyclic imports
        else:
            self.malfunction_generator = mal_gen.NoMalfunctionGen()
            self.malfunction_process_data = self.malfunction_generator.get_process_data()
        
        self.number_of_agents = number_of_agents

        # self.rail_generator: RailGenerator = rail_generator
        if rail_generator is None:
            rail_generator = rail_gen.sparse_rail_generator()
        self.rail_generator = rail_generator
        if line_generator is None:
            line_generator = line_gen.sparse_line_generator()
        self.line_generator = line_generator

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
        self.list_actions = []  # save actions in here

        self.close_following = close_following  # use close following logic
        self.motionCheck = ac.MotionCheck()

        self.agent_helpers = {}

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
        return agent.state == TrainState.READY_TO_DEPART or \
               (TrainState.on_map_state(agent.state) and \
                fast_isclose(agent.speed_data['position_fraction'], 0.0, rtol=1e-03) )

    def reset(self, regenerate_rail: bool = True, regenerate_schedule: bool = True, *,
              random_seed: bool = None) -> Tuple[Dict, Dict]:
        """
        reset(regenerate_rail, regenerate_schedule, activate_agents, random_seed)

        The method resets the rail environment

        Parameters
        ----------
        regenerate_rail : bool, optional
            regenerate the rails
        regenerate_schedule : bool, optional
            regenerate the schedule and the static agents
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

            if "__call__" in dir(self.rail_generator):
                rail, optionals = self.rail_generator(
                    self.width, self.height, self.number_of_agents, self.num_resets, self.np_random)
            elif "generate" in dir(self.rail_generator):
                rail, optionals = self.rail_generator.generate(
                    self.width, self.height, self.number_of_agents, self.num_resets, self.np_random)
            else:
                raise ValueError("Could not invoke __call__ or generate on rail_generator")

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

            line = self.line_generator(self.rail, self.number_of_agents, agents_hints, 
                                               self.num_resets, self.np_random)
            self.agents = EnvAgent.from_line(line)

            # Reset distance map - basically initializing
            self.distance_map.reset(self.agents, self.rail)

            # NEW : Time Schedule Generation
            timetable = timetable_generator(self.agents, self.distance_map, 
                                               agents_hints, self.np_random)

            self._max_episode_steps = timetable.max_episode_steps

            for agent_i, agent in enumerate(self.agents):
                agent.earliest_departure = timetable.earliest_departures[agent_i]         
                agent.latest_arrival = timetable.latest_arrivals[agent_i]
        else:
            self.distance_map.reset(self.agents, self.rail)

        # Agent Positions Map
        self.agent_positions = np.zeros((self.height, self.width), dtype=int) - 1
        
        # Reset agents to initial states
        self.reset_agents()

        # for agent in self.agents:
        #     # Induce malfunctions
        #     if activate_agents:
        #         self.set_agent_active(agent)

        #     self._break_agent(agent)

        #     if agent.malfunction_data["malfunction"] > 0:
        #         agent.speed_data['transition_action_on_cellexit'] = RailEnvActions.DO_NOTHING

        #     # Fix agents that finished their malfunction
        #     self._fix_agent_after_malfunction(agent)

        self.num_resets += 1
        self._elapsed_steps = 0

        # TODO perhaps dones should be part of each agent.
        self.dones = dict.fromkeys(list(range(self.get_num_agents())) + ["__all__"], False)

        # Reset the state of the observation builder with the new environment
        self.obs_builder.reset()

        # Empty the episode store of agent positions
        self.cur_episode = []

        info_dict: Dict = {
            'action_required': {i: self.action_required(agent) for i, agent in enumerate(self.agents)},
            'malfunction': {
                i: agent.malfunction_data['malfunction'] for i, agent in enumerate(self.agents)
            },
            'speed': {i: agent.speed_data['speed'] for i, agent in enumerate(self.agents)},
            'state': {i: agent.state for i, agent in enumerate(self.agents)}
        }
        # Return the new observation vectors for each agent
        observation_dict: Dict = self._get_observations()
        return observation_dict, info_dict
    
    def apply_action_independent(self, action, rail, position, direction):
        if RailEnvActions.is_moving_action(action):
            new_direction, _ = check_action(action, position, direction, rail)
            new_position = get_new_position(position, new_direction)
        else:
            new_position, new_direction = position, direction
        return new_position, direction
    
    def generate_state_transition_signals(self, agent, preprocessed_action, movement_allowed):
        st_signals = {}
        
        st_signals['malfunction_onset'] = agent.malfunction_handler.in_malfunction
        st_signals['malfunction_counter_complete'] = agent.malfunction_handler.malfunction_counter_complete
        st_signals['earliest_departure_reached'] = self._elapsed_steps >= agent.earliest_departure
        st_signals['stop_action_given'] = (preprocessed_action == RailEnvActions.STOP_MOVING)
        st_signals['valid_movement_action_given'] = RailEnvActions.is_moving_action(preprocessed_action)
        st_signals['target_reached'] = fast_position_equal(agent.position, agent.target)
        st_signals['movement_conflict'] = (not movement_allowed) and agent.speed_counter.is_cell_exit # TODO: Modify motion check to provide proper conflict information

        return st_signals

    def _handle_end_reward(self, agent: EnvAgent) -> int:
        '''
        Handles end-of-episode reward for a particular agent.

        Parameters
        ----------
        agent : EnvAgent
        '''
        reward = None
        # agent done? (arrival_time is not None)
        if agent.status == RailAgentStatus.DONE or agent.status == RailAgentStatus.DONE_REMOVED:
            # if agent arrived earlier or on time = 0
            # if agent arrived later = -ve reward based on how late
            reward = min(agent.latest_arrival - agent.arrival_time, 0)

        # Agents not done (arrival_time is None)
        else:
            # CANCELLED check (never departed)
            if (agent.status == RailAgentStatus.READY_TO_DEPART):
                reward = -1 * self.cancellation_factor * \
                    (agent.get_travel_time_on_shortest_path(self.distance_map) + self.cancellation_time_buffer)

            # Departed but never reached
            if (agent.status == RailAgentStatus.ACTIVE):
                reward = agent.get_current_delay(self._elapsed_steps, self.distance_map)
        
        return reward

    def step(self, action_dict_: Dict[int, RailEnvActions]):
        """
        Updates rewards for the agents at a step.
        """
        self._elapsed_steps += 1

        # If we're done, set reward and info_dict and step() is done.
        if self.dones["__all__"]: # TODO: Move boilerplate to different function
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

        self.motionCheck = ac.MotionCheck()  # reset the motion check

        temp_saved_data = {} # TODO : Change name
        
        for i_agent, agent in enumerate(self.agents): # TODO: Important - Do not use i_agent like this, use agent.handle if needed
            # Generate malfunction
            agent.malfunction_handler.generate_malfunction(self.malfunction_generator, self.np_random)

            # Get action for the agent
            action = action_dict_.get(i_agent, RailEnvActions.DO_NOTHING)
            # TODO: Add the bottom stuff to separate function(s)

            # Preprocess action
            action = preprocess_raw_action(action, agent.state)
            action = preprocess_action_when_waiting(action, agent.state)

            # Try moving actions on current position
            current_position, current_direction = agent.position, agent.direction
            agent_not_on_map = current_position is None
            if agent_not_on_map: # Agent not added on map yet
                current_position, current_direction = agent.initial_position, agent.initial_direction
            action = preprocess_moving_action(action, self.rail, current_position, current_direction)

            # Save moving actions in not already saved
            agent.action_saver.save_action_if_allowed(action, agent.state)

            # Calculate new position
            # Add agent to the map if not on it yet
            if agent_not_on_map and agent.action_saver.is_action_saved:
                temp_new_position = agent.initial_position
                temp_new_direction = agent.initial_direction
                preprocessed_action = action
                
            # When cell exit occurs apply saved action independent of other agents
            elif agent.speed_counter.is_cell_exit and agent.action_saver.is_action_saved:
                saved_action = agent.action_saver.saved_action
                # Apply action independent of other agents and get temporary new position and direction
                temp_pd = self.apply_action_independent(saved_action, self.rail, agent.position, agent.direction)
                temp_new_position, temp_new_direction = temp_pd
                preprocessed_action = saved_action
            else:
                temp_new_position, temp_new_direction = agent.position, agent.direction
                preprocessed_action = action

            # TODO: Saving temporary positon shouldn't be needed if recheck of position is not needed later (see TAG#1)
            temp_saved_data[i_agent] = temp_new_position, temp_new_direction, preprocessed_action
            self.motionCheck.addAgent(i_agent, agent.position, temp_new_position)

        # Find conflicts
        # TODO : Important - Modify conflicted positions and select one of them randomly to go to new position
        self.motionCheck.find_conflicts()
        
        for agent in self.agents:
            i_agent = agent.handle

            ## Update positions
            movement_allowed, _ = self.motionCheck.check_motion(i_agent, agent.position) # TODO: Remove final_new_postion from motioncheck
            # TODO : Important : Original code rechecks the next position here again - not sure why? TAG#1
            preprocessed_action = temp_saved_data[i_agent][2] # TODO : Important : Make this namedtuple or class

            # TODO : Looks like a hacky conditionm, improve the handling
            if agent.malfunction_handler.in_malfunction:
                movement_allowed = False

            if movement_allowed:
                final_new_position, final_new_direction = temp_saved_data[i_agent][:2] # TODO : Important : Make this namedtuple or class
            else:
                final_new_position = agent.position
                final_new_direction = agent.direction
            # if final_new_position and self.rail.grid[final_new_position] == 0:
                # import pdb; pdb.set_trace()
            # if final_new_position and not (final_new_position[0] >= 0 and final_new_position[1] >= 0 and final_new_position[0] < self.rail.height and final_new_position[1] < self.rail.width): # TODO: Remove this
                # import pdb; pdb.set_trace()
            agent.position = final_new_position
            agent.direction = final_new_direction

            ## Update states
            state_transition_signals = self.generate_state_transition_signals(agent, preprocessed_action, movement_allowed)
            agent.state_machine.set_transition_signals(state_transition_signals)
            agent.state_machine.step()
            agent.state = agent.state_machine.state # TODO : Make this a property instead?

            # TODO : Important : Looks like a hacky condiition, improve the handling
            if agent.state == TrainState.DONE:
                agent.position = None

            ## Update rewards
            # self.update_rewards(i_agent, agent, rail)

            ## Update counters (malfunction and speed)
            agent.speed_counter.update_counter(agent.state)
            agent.malfunction_handler.update_counter()

            # Clear old action when starting in new cell
            if agent.speed_counter.is_cell_entry:
                agent.action_saver.clear_saved_action()
        
        self.rewards_dict = {i_agent: 0 for i_agent in range(len(self.agents))} # TODO : Remove this
        return self._get_observations(), self.rewards_dict, self.dones, info_dict # TODO : Will need changes?

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
            list_agents_state.append([
                    *pos, int(agent.direction), 
                    agent.malfunction_data["malfunction"],  
                    int(agent.status),
                    int(agent.position in self.motionCheck.svDeadlocked)
                    ])

        self.cur_episode.append(list_agents_state)
        self.list_actions.append(dActions)

    def _get_observations(self):
        """
        Utility which returns the observations for an agent with respect to environment

        Returns
        ------
        Dict object
        """
        # print(f"_get_obs - num agents: {self.get_num_agents()} {list(range(self.get_num_agents()))}")
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
        return agent.malfunction_handler.in_malfunction

    def save(self, filename):
        print("deprecated call to env.save() - pls call RailEnvPersister.save()")
        persistence.RailEnvPersister.save(self, filename)
