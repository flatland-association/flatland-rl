"""
Definition of the RailEnv environment.
"""
import random
from typing import List, Optional, Dict, Tuple

import numpy as np

import flatland.envs.timetable_generators as ttg
from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4 import Grid4Transitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs import agent_chains as ac
from flatland.envs import line_generators as line_gen
from flatland.envs import malfunction_generators as mal_gen
from flatland.envs import persistence
from flatland.envs import rail_generators as rail_gen
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import DistanceMap
from flatland.envs.fast_methods import fast_position_equal
from flatland.envs.line_generators import LineGenerator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils import action_preprocessing
from flatland.envs.step_utils import env_utils
from flatland.envs.step_utils.states import TrainState, StateTransitionSignals
from flatland.envs.step_utils.transition_utils import check_valid_action
from flatland.utils import seeding
from flatland.utils.decorators import send_infrastructure_data_change_signal_to_reset_lru_cache, \
    enable_infrastructure_lru_cache
from flatland.utils.rendertools import RenderTool, AgentRenderVariant


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

    alpha = 0
    beta = 0
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
                 line_generator: LineGenerator = None,  # : line_gen.LineGenerator = line_gen.random_line_generator(),
                 number_of_agents=2,
                 obs_builder_object: ObservationBuilder = GlobalObsForRailEnv(),
                 malfunction_generator_and_process_data=None,  # mal_gen.no_malfunction_generator(),
                 malfunction_generator=None,
                 remove_agents_at_target=True,
                 random_seed=None,
                 record_steps=False,
                 timetable_generator=ttg.timetable_generator,
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

        if rail_generator is None:
            rail_generator = rail_gen.sparse_rail_generator()
        self.rail_generator = rail_generator
        if line_generator is None:
            line_generator = line_gen.sparse_line_generator()
        self.line_generator: LineGenerator = line_generator
        self.timetable_generator = timetable_generator

        self.rail: Optional[GridTransitionMap] = None
        self.width = width
        self.height = height

        self.remove_agents_at_target = remove_agents_at_target

        self.obs_builder = obs_builder_object
        self.obs_builder.set_env(self)

        self._max_episode_steps: Optional[int] = None
        self._elapsed_steps = 0

        self.obs_dict = {}
        self.rewards_dict = {}
        self.dev_obs_dict = {}
        self.dev_pred_dict = {}

        self.agents: List[EnvAgent] = []
        self.num_resets = 0
        self.distance_map = DistanceMap(self.agents, self.height, self.width)

        self.action_space = [5]

        self._seed(seed=random_seed)

        self.agent_positions = None

        # save episode timesteps ie agent positions, orientations.  (not yet actions / observations)
        self.record_steps = record_steps  # whether to save timesteps
        # save timesteps in here: [[[row, col, dir, malfunction],...nAgents], ...nSteps]
        self.cur_episode = []
        self.list_actions = []  # save actions in here

        self.motionCheck = ac.MotionCheck()

    def _seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        self.random_seed = seed

        # Keep track of all the seeds in order
        if not hasattr(self, 'seed_history'):
            self.seed_history = [seed]
        if self.seed_history[-1] != seed:
            self.seed_history.append(seed)

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

    @enable_infrastructure_lru_cache()
    def action_required(self, agent_state, is_cell_entry):
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
        return agent_state == TrainState.READY_TO_DEPART or \
            (agent_state.is_on_map_state() and is_cell_entry)

    def reset(self, regenerate_rail: bool = True, regenerate_schedule: bool = True, *,
              random_seed: int = None) -> Tuple[Dict, Dict]:
        """
        reset(regenerate_rail, regenerate_schedule, activate_agents, random_seed)

        The method resets the rail environment

        Parameters
        ----------
        regenerate_rail : bool, optional
            regenerate the rails
        regenerate_schedule : bool, optional
            regenerate the schedule and the static agents
        random_seed : int, optional
            random seed for environment

        Returns
        -------
        observation_dict: Dict
            Dictionary with an observation for each agent
        info_dict: Dict with agent specific information

        """

        send_infrastructure_data_change_signal_to_reset_lru_cache()

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
            timetable = self.timetable_generator(self.agents, self.distance_map,
                                                 agents_hints, self.np_random)

            self._max_episode_steps = timetable.max_episode_steps

            for agent_i, agent in enumerate(self.agents):
                agent.earliest_departure = timetable.earliest_departures[agent_i]
                agent.latest_arrival = timetable.latest_arrivals[agent_i]
        else:
            self.distance_map.reset(self.agents, self.rail)

        # Reset agents to initial states
        self.reset_agents()

        self.num_resets += 1
        self._elapsed_steps = 0

        # Agent positions map
        self.agent_positions = np.zeros((self.height, self.width), dtype=int) - 1
        self._update_agent_positions_map(ignore_old_positions=False)

        self.dones = dict.fromkeys(list(range(self.get_num_agents())) + ["__all__"], False)

        # Reset the state of the observation builder with the new environment
        self.obs_builder.reset()

        # Empty the episode store of agent positions
        self.cur_episode = []

        info_dict = self.get_info_dict()
        # Return the new observation vectors for each agent
        observation_dict: Dict = self._get_observations()
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer = None
        return observation_dict, info_dict

    def _update_agent_positions_map(self, ignore_old_positions=True):
        """ Update the agent_positions array for agents that changed positions """
        for agent in self.agents:
            if not ignore_old_positions or agent.old_position != agent.position:
                if agent.position is not None:
                    self.agent_positions[agent.position] = agent.handle
                if agent.old_position is not None:
                    self.agent_positions[agent.old_position] = -1

    def generate_state_transition_signals(self, agent, preprocessed_action, movement_allowed):
        """ Generate State Transitions Signals used in the state machine """
        st_signals = StateTransitionSignals()

        # Malfunction starts when in_malfunction is set to true
        st_signals.in_malfunction = agent.malfunction_handler.in_malfunction

        # Malfunction counter complete - Malfunction ends next timestep
        st_signals.malfunction_counter_complete = agent.malfunction_handler.malfunction_counter_complete

        # Earliest departure reached - Train is allowed to move now
        st_signals.earliest_departure_reached = self._elapsed_steps >= agent.earliest_departure

        # Stop Action Given
        st_signals.stop_action_given = (preprocessed_action == RailEnvActions.STOP_MOVING)

        # Valid Movement action Given
        st_signals.valid_movement_action_given = preprocessed_action.is_moving_action() and movement_allowed

        # Target Reached
        st_signals.target_reached = fast_position_equal(agent.position, agent.target)

        # Movement conflict - Multiple trains trying to move into same cell
        # If speed counter is not in cell exit, the train can enter the cell
        st_signals.movement_conflict = (not movement_allowed) and agent.speed_counter.is_cell_exit

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
        if agent.state == TrainState.DONE:
            # if agent arrived earlier or on time = 0
            # if agent arrived later = -ve reward based on how late
            reward = min(agent.latest_arrival - agent.arrival_time, 0)

        # Agents not done (arrival_time is None)
        else:
            # CANCELLED check (never departed)
            if (agent.state.is_off_map_state()):
                reward = -1 * self.cancellation_factor * \
                         (agent.get_travel_time_on_shortest_path(self.distance_map) + self.cancellation_time_buffer)

            # Departed but never reached
            if (agent.state.is_on_map_state()):
                reward = agent.get_current_delay(self._elapsed_steps, self.distance_map)

        return reward

    def preprocess_action(self, action, agent):
        """
        Preprocess the provided action
            * Change to DO_NOTHING if illegal action
            * Block all actions when in waiting state
            * Check MOVE_LEFT/MOVE_RIGHT actions on current position else try MOVE_FORWARD
        """
        action = action_preprocessing.preprocess_raw_action(action, agent.state, agent.action_saver.saved_action)
        action = action_preprocessing.preprocess_action_when_waiting(action, agent.state)

        # Try moving actions on current position
        current_position, current_direction = agent.position, agent.direction
        if current_position is None:  # Agent not added on map yet
            current_position, current_direction = agent.initial_position, agent.initial_direction

        action = action_preprocessing.preprocess_moving_action(action, self.rail, current_position, current_direction)

        # Check transitions, bounts for executing the action in the given position and directon
        if action.is_moving_action() and not check_valid_action(action, self.rail, current_position, current_direction):
            action = RailEnvActions.STOP_MOVING

        return action

    def clear_rewards_dict(self):
        """ Reset the rewards dictionary """
        self.rewards_dict = {i_agent: 0 for i_agent in range(len(self.agents))}

    def get_info_dict(self):
        """
        Returns dictionary of infos for all agents
        dict_keys : action_required -
                    malfunction - Counter value for malfunction > 0 means train is in malfunction
                    speed - Speed of the train
                    state - State from the trains's state machine
        """
        info_dict = {
            'action_required': {i: self.action_required(agent.state, agent.speed_counter.is_cell_entry)
                                for i, agent in enumerate(self.agents)},
            'malfunction': {
                i: agent.malfunction_handler.malfunction_down_counter for i, agent in enumerate(self.agents)
            },
            'speed': {i: agent.speed_counter.speed for i, agent in enumerate(self.agents)},
            'state': {i: agent.state for i, agent in enumerate(self.agents)}
        }
        return info_dict

    def update_step_rewards(self, i_agent):
        """
        Update the rewards dict for agent id i_agent for every timestep
        """
        pass

    def end_of_episode_update(self, have_all_agents_ended):
        """
        Updates made when episode ends
        Parameters: have_all_agents_ended - Indicates if all agents have reached done state
        """
        if have_all_agents_ended or \
            ((self._max_episode_steps is not None) and (self._elapsed_steps >= self._max_episode_steps)):

            for i_agent, agent in enumerate(self.agents):
                reward = self._handle_end_reward(agent)
                self.rewards_dict[i_agent] += reward

                self.dones[i_agent] = True

            self.dones["__all__"] = True

    def handle_done_state(self, agent):
        """ Any updates to agent to be made in Done state """
        if agent.state == TrainState.DONE and agent.arrival_time is None:
            agent.arrival_time = self._elapsed_steps
            self.dones[agent.handle] = True
            if self.remove_agents_at_target:
                agent.position = None

    def step(self, action_dict_: Dict[int, RailEnvActions]):
        """
        Updates rewards for the agents at a step.
        """
        self._elapsed_steps += 1

        # Not allowed to step further once done
        if self.dones["__all__"]:
            raise Exception("Episode is done, cannot call step()")

        self.clear_rewards_dict()

        have_all_agents_ended = True  # Boolean flag to check if all agents are done

        self.motionCheck = ac.MotionCheck()  # reset the motion check

        temp_transition_data = {}

        for agent in self.agents:
            i_agent = agent.handle
            agent.old_position = agent.position
            agent.old_direction = agent.direction
            # Generate malfunction
            agent.malfunction_handler.generate_malfunction(self.malfunction_generator, self.np_random)

            # Get action for the agent
            action = action_dict_.get(i_agent, RailEnvActions.DO_NOTHING)

            preprocessed_action = self.preprocess_action(action, agent)

            # Save moving actions in not already saved
            agent.action_saver.save_action_if_allowed(preprocessed_action, agent.state)

            # Train's next position can change if train is at cell's exit and train is not in malfunction
            position_update_allowed = agent.speed_counter.is_cell_exit and \
                                      not agent.malfunction_handler.malfunction_down_counter > 0 and \
                                      not preprocessed_action == RailEnvActions.STOP_MOVING

            # Calculate new position
            # Keep agent in same place if already done
            if agent.state == TrainState.DONE:
                new_position, new_direction = agent.position, agent.direction
            # Add agent to the map if not on it yet
            elif agent.position is None and agent.action_saver.is_action_saved:
                new_position = agent.initial_position
                new_direction = agent.initial_direction
            # If movement is allowed apply saved action independent of other agents
            elif agent.action_saver.is_action_saved and position_update_allowed:
                saved_action = agent.action_saver.saved_action
                # Apply action independent of other agents and get temporary new position and direction
                new_position, new_direction = env_utils.apply_action_independent(saved_action,
                                                                                 self.rail,
                                                                                 agent.position,
                                                                                 agent.direction)
                preprocessed_action = saved_action
            else:
                new_position, new_direction = agent.position, agent.direction

            temp_transition_data[i_agent] = env_utils.AgentTransitionData(position=new_position,
                                                                          direction=new_direction,
                                                                          preprocessed_action=preprocessed_action)

            # This is for storing and later checking for conflicts of agents trying to occupy same cell
            self.motionCheck.addAgent(i_agent, agent.position, new_position)

        # Find conflicts between trains trying to occupy same cell
        self.motionCheck.find_conflicts()

        for agent in self.agents:
            i_agent = agent.handle

            ## Update positions
            if agent.malfunction_handler.in_malfunction:
                movement_allowed = False
            else:
                movement_allowed = self.motionCheck.check_motion(i_agent, agent.position)

            movement_inside_cell = agent.state == TrainState.STOPPED and not agent.speed_counter.is_cell_exit
            movement_allowed = movement_allowed or movement_inside_cell

            # Fetch the saved transition data
            agent_transition_data = temp_transition_data[i_agent]
            preprocessed_action = agent_transition_data.preprocessed_action

            ## Update states
            state_transition_signals = self.generate_state_transition_signals(agent, preprocessed_action,
                                                                              movement_allowed)
            agent.state_machine.set_transition_signals(state_transition_signals)
            agent.state_machine.step()

            # Needed when not removing agents at target
            movement_allowed = movement_allowed and agent.state != TrainState.DONE

            # Agent is being added to map
            if agent.state.is_on_map_state():
                if agent.state_machine.previous_state.is_off_map_state():
                    agent.position = agent.initial_position
                    agent.direction = agent.initial_direction
                # Speed counter completes
                elif movement_allowed and (agent.speed_counter.is_cell_exit):
                    agent.position = agent_transition_data.position
                    agent.direction = agent_transition_data.direction
                    agent.state_machine.update_if_reached(agent.position, agent.target)

            # Off map or on map state and position should match
            env_utils.state_position_sync_check(agent.state, agent.position, agent.handle)

            # Handle done state actions, optionally remove agents
            self.handle_done_state(agent)

            have_all_agents_ended &= (agent.state == TrainState.DONE)

            ## Update rewards
            self.update_step_rewards(i_agent)

            ## Update counters (malfunction and speed)
            agent.speed_counter.update_counter(agent.state, agent.old_position)
            #    agent.state_machine.previous_state)
            agent.malfunction_handler.update_counter()

            # Clear old action when starting in new cell
            if agent.speed_counter.is_cell_entry and agent.position is not None:
                agent.action_saver.clear_saved_action()

        # Check if episode has ended and update rewards and dones
        self.end_of_episode_update(have_all_agents_ended)

        self._update_agent_positions_map()
        if self.record_steps:
            self.record_timestep(action_dict_)

        return self._get_observations(), self.rewards_dict, self.dones, self.get_info_dict()

    def record_timestep(self, dActions):
        """
        Record the positions and orientations of all agents in memory, in the cur_episode
        """
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
                agent.malfunction_handler.malfunction_down_counter,
                0,  # int(agent.status), #  TODO: find appropriate attribute for agent status
                int(agent.position in self.motionCheck.svDeadlocked)
            ])

        self.cur_episode.append(list_agents_state)
        self.list_actions.append(dActions)

    def _get_observations(self):
        """
        Utility which returns the dictionary of observations for an agent with respect to environment
        """
        # print(f"_get_obs - num agents: {self.get_num_agents()} {list(range(self.get_num_agents()))}")
        self.obs_dict = self.obs_builder.get_many(list(range(self.get_num_agents())))
        return self.obs_dict

    def get_valid_directions_on_grid(self, row: int, col: int) -> List[int]:
        """
        Returns directions in which the agent can move
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
        print("DEPRECATED call to env.save() - pls call RailEnvPersister.save()")
        persistence.RailEnvPersister.save(self, filename)

    def render(self, mode="rgb_array", gl="PGL", agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
               show_debug=False, clear_debug_text=True, show=False,
               screen_height=600, screen_width=800,
               show_observations=False, show_predictions=False,
               show_rowcols=False, return_image=True):
        """
        This methods provides the option to render the
        environment's behavior as an image or to a window.
        Parameters
        ----------
        mode

        Returns
        -------
        Image if mode is rgb_array, opens a window otherwise
        """
        if not hasattr(self, "renderer") or self.renderer is None:
            self.initialize_renderer(mode=mode, gl=gl,  # gl="TKPILSVG",
                                     agent_render_variant=agent_render_variant,
                                     show_debug=show_debug,
                                     clear_debug_text=clear_debug_text,
                                     show=show,
                                     screen_height=screen_height,  # Adjust these parameters to fit your resolution
                                     screen_width=screen_width)
        return self.update_renderer(mode=mode, show=show, show_observations=show_observations,
                                    show_predictions=show_predictions,
                                    show_rowcols=show_rowcols, return_image=return_image)

    def initialize_renderer(self, mode, gl,
                            agent_render_variant,
                            show_debug,
                            clear_debug_text,
                            show,
                            screen_height,
                            screen_width):
        # Initiate the renderer
        self.renderer = RenderTool(self, gl=gl,  # gl="TKPILSVG",
                                   agent_render_variant=agent_render_variant,
                                   show_debug=show_debug,
                                   clear_debug_text=clear_debug_text,
                                   screen_height=screen_height,  # Adjust these parameters to fit your resolution
                                   screen_width=screen_width)  # Adjust these parameters to fit your resolution
        self.renderer.show = show
        self.renderer.reset()

    def update_renderer(self, mode, show, show_observations, show_predictions,
                        show_rowcols, return_image):
        """
        This method updates the render.
        Parameters
        ----------
        mode

        Returns
        -------
        Image if mode is rgb_array, None otherwise
        """
        image = self.renderer.render_env(show=show, show_observations=show_observations,
                                         show_predictions=show_predictions,
                                         show_rowcols=show_rowcols, return_image=return_image)
        if mode == 'rgb_array':
            return image[:, :, :3]

    def close(self):
        """
        This methods closes any renderer window.
        """
        if hasattr(self, "renderer") and self.renderer is not None:
            try:
                if self.renderer.show:
                    self.renderer.close_window()
            except Exception as e:
                print("Could Not close window due to:", e)
            self.renderer = None
