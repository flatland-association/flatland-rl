"""
Definition of the RailEnv environment.
"""
import random
import warnings
from collections import defaultdict
from functools import lru_cache
from typing import List, Optional, Dict, Tuple, Set, Any

import numpy as np

import flatland.envs.timetable_generators as ttg
from flatland.core.effects_generator import EffectsGenerator
from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4 import Grid4Transitions
from flatland.core.grid.grid_utils import Vector2D
from flatland.core.grid.rail_env_grid import RailEnvTransitionsEnum
from flatland.envs import agent_chains as ac
from flatland.envs import line_generators as line_gen
from flatland.envs import malfunction_generators as mal_gen
from flatland.envs import persistence
from flatland.envs import rail_generators as rail_gen
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import DistanceMap
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rewards import Rewards
from flatland.envs.step_utils import env_utils
from flatland.envs.step_utils.state_machine import TrainStateMachine
from flatland.envs.step_utils.states import TrainState, StateTransitionSignals
from flatland.utils import seeding
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



    Stochastic malfunctioning of trains:
    Trains in RailEnv can malfunction if they are halted too often (either by their own choice or because an invalid
    action or cell is selected.

    Every time an agent stops, an agent has a certain probability of malfunctioning. Malfunctions of trains follow a
    poisson process with a certain rate. Not all trains will be affected by malfunctions during episodes to keep
    complexity manageable.

    TODO: currently, the parameters that control the stochasticity of the environment are hard-coded in init().
    For Round 2, they will be passed to the constructor as arguments, to allow for more flexibility.

    """

    def __init__(self,
                 width,
                 height,
                 rail_generator: "RailGenerator" = None,
                 line_generator: "LineGenerator" = None,
                 number_of_agents=2,
                 obs_builder_object: ObservationBuilder = GlobalObsForRailEnv(),
                 malfunction_generator_and_process_data=None,
                 malfunction_generator: "MalfunctionGenerator" = None,
                 remove_agents_at_target=True,
                 random_seed=None,
                 record_steps=False,
                 timetable_generator=ttg.timetable_generator,
                 acceleration_delta=1.0,
                 braking_delta=-1.0,
                 rewards: Rewards = None,
                 effects_generator: EffectsGenerator["RailEnv"] = None
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
            RailEnv.DEPOT_POSITION when the agent has reached its target position.
        random_seed : int or None
            if None, then its ignored, else the random generators are seeded with this number to ensure
            that stochastic operations are replicable across multiple operations
        acceleration_delta : float
            Determines how much speed is increased by MOVE_FORWARD action up to max_speed set by train's Line (sampled from `speed_ratios` by `LineGenerator`).
            As speed is between 0.0 and 1.0, acceleration_delta=1.0 restores to previous constant speed behaviour
            (i.e. MOVE_FORWARD always sets to max speed allowed for train).
        braking_delta : float
            Determines how much speed is decreased by STOP_MOVING action.
            As speed is between 0.0 and 1.0, braking_delta=-1.0 restores to previous full stop behaviour.
        rewards : Rewards
            The rewards function to use. Defaults to standard settings of Flatland 3 behaviour.
        effects_generator : Optional[EffectsGenerator["RailEnv"]]
            The effects generator that can modify the env at the env of env reset, at the beginning of the env step and at the end of the env step.
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
        self.line_generator: "LineGenerator" = line_generator
        self.timetable_generator = timetable_generator

        self.rail: Optional[RailGridTransitionMap] = None
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

        self.level_free_positions: Set[Vector2D] = set()

        if rewards is None:
            self.rewards = Rewards()
        else:
            self.rewards = rewards

        self.acceleration_delta = acceleration_delta
        self.braking_delta = braking_delta

        self.effects_generator = effects_generator

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
    def get_agent_handles(self) -> List[int]:
        return list(range(self.get_num_agents()))

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

    @lru_cache()
    @staticmethod
    def action_required(agent_state, is_cell_entry):
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

    def reset(self, regenerate_rail: bool = True, regenerate_schedule: bool = True, *, random_seed: int = None) -> Tuple[Dict, Dict]:
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
        if random_seed is not None:
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
            if optionals and 'level_free_positions' in optionals:
                self.level_free_positions = optionals['level_free_positions']

            line = self.line_generator(self.rail, self.number_of_agents, agents_hints,
                                       self.num_resets, self.np_random)
            self.agents = EnvAgent.from_line(line)

            # Reset distance map - basically initializing
            self.distance_map.reset(self.agents, self.rail)

            # NEW : Timetable Generation
            timetable = self.timetable_generator(self.agents, self.distance_map,
                                                 agents_hints, self.np_random)

            self._max_episode_steps = timetable.max_episode_steps

            EnvAgent.apply_timetable(self.agents, timetable)
        else:
            self.distance_map.reset(self.agents, self.rail)

        # Reset agents to initial states
        self.reset_agents()

        self.num_resets += 1
        self._elapsed_steps = 0

        # Agent positions map
        self.agent_positions = np.zeros((self.height, self.width), dtype=int) - 1
        self._update_agent_positions_map(ignore_old_positions=False)

        if self.effects_generator is not None:
            self.effects_generator.on_episode_start(self)

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
            * Change to DO_NOTHING if illegal action (not one of the defined action)
            * Check MOVE_LEFT/MOVE_RIGHT actions on current position else try MOVE_FORWARD
            * Change to STOP_MOVING if the movement is not possible in the grid (e.g. if MOVE_FORWARD in a symmetric switch or MOVE_LEFT in straight element or leads outside of bounds).
        """
        action = RailEnvActions(action)
        action = RailEnv._process_illegal_action(action)

        # Try moving actions on current position
        current_position, current_direction = agent.position, agent.direction
        if current_position is None:  # Agent not added on map yet
            current_position, current_direction = agent.initial_position, agent.initial_direction

        # TODO revise design: should we stop the agent instead and penalize it?
        action = self.rail.preprocess_left_right_action(action, current_position, current_direction)

        # TODO https://github.com/flatland-association/flatland-rl/issues/185 Streamline flatland.envs.step_utils.transition_utils and flatland.envs.step_utils.action_preprocessing
        if ((action.is_moving_action() or action == RailEnvActions.DO_NOTHING)
            and
            not self.rail.check_valid_action(action, current_position, current_direction)):
            # TODO revise design: should we add penalty if the action cannot be executed?
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
            # TODO https://github.com/flatland-association/flatland-rl/issues/149 revise action required
            'action_required': {i: RailEnv.action_required(agent.state, agent.speed_counter.is_cell_entry)
                                for i, agent in enumerate(self.agents)},
            'malfunction': {
                i: agent.malfunction_handler.malfunction_down_counter for i, agent in enumerate(self.agents)
            },
            'speed': {i: agent.speed_counter.speed for i, agent in enumerate(self.agents)},
            'state': {i: agent.state for i, agent in enumerate(self.agents)}
        }
        return info_dict

    def end_of_episode_update(self, have_all_agents_ended):
        """
        Updates made when episode ends
        Parameters: have_all_agents_ended - Indicates if all agents have reached done state
        """
        if have_all_agents_ended or \
            ((self._max_episode_steps is not None) and (self._elapsed_steps >= self._max_episode_steps)):

            for i_agent, agent in enumerate(self.agents):
                reward = self.rewards.end_of_episode_reward(agent, self.distance_map, self._elapsed_steps)
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

    def step(self, action_dict: Dict[int, RailEnvActions]):
        """
        Updates rewards for the agents at a step.
        """
        self._elapsed_steps += 1

        # Not allowed to step further once done
        if self.dones["__all__"]:
            raise Exception("Episode is done, cannot call step()")

        self.clear_rewards_dict()

        self.motionCheck = ac.MotionCheck()  # reset the motion check

        if self.effects_generator is not None:
            self.effects_generator.on_episode_step_start(self)

        temp_transition_data = {}
        for agent in self.agents:
            i_agent = agent.handle

            agent.old_position = agent.position
            agent.old_direction = agent.direction

            # Generate malfunction
            agent.malfunction_handler.generate_malfunction(self.malfunction_generator, self.np_random)

            # Get action for the agent
            raw_action = action_dict.get(i_agent, RailEnvActions.DO_NOTHING)
            preprocessed_action = self.preprocess_action(raw_action, agent)

            # get desired new_position and new_direction
            stop_action_given = preprocessed_action == RailEnvActions.STOP_MOVING
            in_malfunction = agent.malfunction_handler.in_malfunction
            movement_action_given = preprocessed_action.is_moving_action()
            earliest_departure_reached = agent.earliest_departure <= self._elapsed_steps

            new_speed = agent.speed_counter.speed
            state = agent.state
            # TODO revise design: should we instead of correcting LEFT/RIGHT to FORWARD instead preprocess to DO_NOTHING?
            if (preprocessed_action == RailEnvActions.MOVE_FORWARD and raw_action == RailEnvActions.MOVE_FORWARD) or (
                (state == TrainState.STOPPED or state == TrainState.MALFUNCTION) and preprocessed_action.is_moving_action()):
                new_speed += self.acceleration_delta
            elif preprocessed_action == RailEnvActions.STOP_MOVING:
                new_speed += self.braking_delta
            new_speed = max(0.0, min(agent.speed_counter.max_speed, new_speed))

            if state == TrainState.READY_TO_DEPART and movement_action_given:
                new_position = agent.initial_position
                new_direction = agent.initial_direction
            elif state == TrainState.MALFUNCTION_OFF_MAP and not in_malfunction and earliest_departure_reached and (
                preprocessed_action.is_moving_action() or preprocessed_action == RailEnvActions.STOP_MOVING):
                # TODO revise design: weirdly, MALFUNCTION_OFF_MAP does not go via READY_TO_DEPART, but STOP_MOVING and MOVE_* adds to map if possible
                new_position = agent.initial_position
                new_direction = agent.initial_direction
            elif state.is_on_map_state():
                new_position, new_direction = agent.position, agent.direction
                # transition to next cell: at end of cell and next state potentially MOVING
                if (agent.speed_counter.is_cell_exit(new_speed)
                    and
                    TrainStateMachine.can_get_moving_independent(state, in_malfunction, movement_action_given, new_speed, stop_action_given)
                ):
                    new_position, new_direction = self.rail.apply_action_independent(
                        preprocessed_action,
                        agent.position,
                        agent.direction
                    )
                assert agent.position is not None
            else:
                assert state.is_off_map_state() or state == TrainState.DONE
                new_position = None
                new_direction = None

            if new_position is not None:
                valid_position_direction = any(self.rail.get_transitions(*new_position, new_direction))
                if not valid_position_direction:
                    warnings.warn(f"{(new_position, new_direction)} not valid on the grid."
                                  f" Coming from {(agent.position, agent.direction)} with raw action {raw_action} and preprocessed action {preprocessed_action}. {RailEnvTransitionsEnum(self.rail.get_full_transitions(*agent.position)).name}")
                assert valid_position_direction

            # only conflict if the level-free cell is traversed through the same axis (horizontally (0 north or 2 south), or vertically (1 east or 3 west)
            new_position_level_free = new_position
            if new_position in self.level_free_positions:
                new_position_level_free = (new_position, new_direction % 2)
            agent_position_level_free = agent.position
            if agent.position in self.level_free_positions:
                agent_position_level_free = (agent.position, agent.direction % 2)

            state_transition_signals = StateTransitionSignals(
                # Malfunction starts when in_malfunction is set to true (inverse of malfunction_counter_complete)
                in_malfunction=agent.malfunction_handler.in_malfunction,
                # Earliest departure reached - Train is allowed to move now
                earliest_departure_reached=self._elapsed_steps >= agent.earliest_departure,
                # Stop action given
                stop_action_given=(preprocessed_action == RailEnvActions.STOP_MOVING),
                # Movement action given
                movement_action_given=preprocessed_action.is_moving_action(),
                # Target reached - we only know after state and positions update - see handle_done_state below
                target_reached=None,  # we only know after motion check
                # Movement allowed if inside cell or at end of cell and no conflict with other trains - we only know after motion check!
                movement_allowed=None,  # we only know after motion check
                # New desired speed if movement allowed
                new_speed=new_speed
            )

            agent_transition_data = env_utils.AgentTransitionData(
                speed=agent.speed_counter.speed,
                agent_position_level_free=agent_position_level_free,

                new_position=new_position,
                new_direction=new_direction,
                new_speed=new_speed,
                new_position_level_free=new_position_level_free,

                preprocessed_action=preprocessed_action,
                state_transition_signal=state_transition_signals
            )
            temp_transition_data[i_agent] = agent_transition_data

            self.motionCheck.addAgent(i_agent, agent_position_level_free, new_position_level_free)

        # Find conflicts between trains trying to occupy same cell
        self.motionCheck.find_conflicts()

        have_all_agents_ended = True
        for agent in self.agents:
            i_agent = agent.handle

            # Fetch the saved transition data
            agent_transition_data = temp_transition_data[i_agent]

            # motion_check is False if agent wants to stay in the cell
            motion_check = self.motionCheck.check_motion(i_agent, agent_transition_data.agent_position_level_free)
            # Movement allowed if inside cell or at end of cell and no conflict with other trains
            movement_allowed = (agent.state.is_on_map_state() and not agent.speed_counter.is_cell_exit(agent_transition_data.new_speed)) or motion_check

            agent_transition_data.state_transition_signal.movement_allowed = movement_allowed

            # state machine step
            agent.state_machine.set_transition_signals(agent_transition_data.state_transition_signal)
            agent.state_machine.step()

            # position and speed_counter update
            if agent.state == TrainState.MOVING:
                # only position update while MOVING and motion_check OK
                agent.position = agent_transition_data.new_position
                agent.direction = agent_transition_data.new_direction
                # N.B. no movement in first time step after READY_TO_DEPART or MALFUNCTION_OFF_MAP!
                if not (agent.state_machine.previous_state == TrainState.READY_TO_DEPART or
                        agent.state_machine.previous_state == TrainState.MALFUNCTION_OFF_MAP):
                    agent.speed_counter.step(speed=agent_transition_data.new_speed)
                agent.state_machine.update_if_reached(agent.position, agent.target)
            elif agent.state_machine.previous_state == TrainState.MALFUNCTION_OFF_MAP and agent.state == TrainState.STOPPED:
                agent.position = agent.initial_position
                agent.direction = agent.initial_direction

            # TODO revise design: condition could be generalized to not MOVING if we would enforce MALFUNCTION_OFF_MAP to go to READY_TO_DEPART first.
            if agent.state.is_on_map_state() and agent.state != TrainState.MOVING:
                agent.speed_counter.step(speed=0)

            # Handle done state actions, optionally remove agents
            self.handle_done_state(agent)
            have_all_agents_ended &= (agent.state == TrainState.DONE)

            ## Update rewards
            self.rewards_dict[i_agent] += self.rewards.step_reward(agent, agent_transition_data, self.distance_map, self._elapsed_steps)

            # update malfunction counter
            agent.malfunction_handler.update_counter()

            # Off map or on map state and position should match
            agent.state_machine.state_position_sync_check(agent.position, agent.handle, self.remove_agents_at_target)

        # Check if episode has ended and update rewards and dones
        self.end_of_episode_update(have_all_agents_ended)

        self._update_agent_positions_map()

        self._verify_mutually_exclusive_cell_occupation()

        if self.record_steps:
            self.record_timestep(action_dict)

        if self.effects_generator is not None:
            self.effects_generator.on_episode_step_end(self)

        return self._get_observations(), self.rewards_dict, self.dones, self.get_info_dict()

    def _verify_mutually_exclusive_cell_occupation(self):
        agent_positions_same_level = []
        agent_positions_level_free = defaultdict(lambda: [])
        for agent in self.agents:
            if agent.position is not None:
                if agent.position in self.level_free_positions:
                    agent_positions_level_free[agent.position].append(agent.direction)
                else:
                    agent_positions_same_level.append(agent.position)
        msgs = f"Found two agents occupying same cell in step {self._elapsed_steps}: {agent_positions_same_level}\n"
        msgs += f"- motion check: {list(self.motionCheck.G.edges)}"
        if len(agent_positions_same_level) != len(set(agent_positions_same_level)):
            warnings.warn(msgs)
            counts = {pos: agent_positions_same_level.count(pos) for pos in set(agent_positions_same_level)}
            dup_positions = [pos for pos, count in counts.items() if count > 1]
            for dup in dup_positions:
                for agent in self.agents:
                    if agent.position == dup:
                        msg = (f"\n================== BAD AGENT ==================================\n\n\n\n\n"
                               f"- agent:\t{agent} \n"
                               f"- state_machine:\t{agent.state_machine}\n"
                               f"- speed_counter:\t{agent.speed_counter}\n"
                               f"- breakpoint:\tself._elapsed_steps == {self._elapsed_steps} and agent.handle == {agent.handle}\n"
                               f"- motion check:\t{list(self.motionCheck.G.edges)}\n\n\n"
                               f"- agents:\t{self.agents}")
                        warnings.warn(msg)
                        msgs += msg
        assert len(agent_positions_same_level) == len(set(agent_positions_same_level)), msgs

        for position, directions in agent_positions_level_free.items():
            if len(directions) >= 2:
                warnings.warn(f"Found more than two agents occupying same level-free cell in step {self._elapsed_steps}: {agent_positions_level_free}")
            assert len(directions) <= 2
            if len(directions) == 2:
                conflict = directions[0] % 2 == directions[1] % 2
                if conflict:
                    warnings.warn(
                        f"Found two agents occupying same level-free cell along the same axis in step {self._elapsed_steps}: {agent_positions_level_free}")
                assert not conflict

    # TODO extract to callbacks instead!
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
                agent.state.value,
                int(agent.position in self.motionCheck.svDeadlocked),
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
        We need this to guarantee synchronicity between different instances with the same seed.
        :param rate:
        :return:
        """
        u = self.np_random.rand()
        x = - np.log(1 - u) * rate
        return x

    def _is_agent_ok(self, agent: EnvAgent) -> bool:
        """
        Checks if an agent is ok, meaning it can move and is not malfunctioning.
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
        Provides the option to render the
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
        Closes any renderer window.
        """
        if hasattr(self, "renderer") and self.renderer is not None:
            try:
                if self.renderer.show:
                    self.renderer.close_window()
            except Exception as e:
                print("Could Not close window due to:", e)
            self.renderer = None

    @staticmethod
    @lru_cache()
    def _process_illegal_action(action: Any) -> RailEnvActions:
        """
        Returns the action if valid (either int value or in RailEnvActions), returns RailEnvActions.DO_NOTHING otherwise.
        """
        if not RailEnvActions.is_action_valid(action):
            return RailEnvActions.DO_NOTHING
        else:
            return RailEnvActions(action)
