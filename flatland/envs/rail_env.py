"""
Definition of the RailEnv environment.
"""
import pickle
import random
import warnings
from functools import lru_cache
from typing import List, Optional, Dict, Tuple, Any, Generic, TypeVar

import numpy as np

import flatland.envs.timetable_generators as ttg
from flatland.core.effects_generator import EffectsGenerator, make_multi_effects_generator
from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid_resource_map import GridResourceMap
from flatland.core.resource_map import ResourceMap
from flatland.core.transition_map import GridTransitionMap, TransitionMap
from flatland.envs import agent_chains as ac
from flatland.envs import line_generators as line_gen
from flatland.envs import malfunction_generators as mal_gen
from flatland.envs import rail_generators as rail_gen
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import DistanceMap, AbstractDistanceMap
from flatland.envs.grid.rail_env_grid import RailEnvTransitionsEnum
from flatland.envs.malfunction_effects_generators import MalfunctionEffectsGenerator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rewards import DefaultRewards, Rewards
from flatland.envs.step_utils import env_utils
from flatland.envs.step_utils.state_machine import TrainStateMachine
from flatland.envs.step_utils.states import TrainState, StateTransitionSignals
from flatland.utils import seeding

UnderlyingTransitionMapType = TypeVar('UnderlyingTransitionMapType', bound=TransitionMap)
UnderlyingResourceMapType = TypeVar('UnderlyingResourceMapType', bound=ResourceMap)


class AbstractRailEnv(Environment, Generic[UnderlyingTransitionMapType, UnderlyingResourceMapType]):
    """
    AbstractRailEnv environment class.

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

    In order for agents to be able to "understand" the simulation behaviour from the observations,
    the execution order of actions should not matter (i.e. not depend on the agent handle).
    However, the agent ordering is still used to resolve conflicts between two agents trying to move into the same cell,
    for example, head-on collisions, or agents "merging" at junctions.
    See `MotionCheck` for more details.




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
                 rail_generator: "RailGenerator" = None,
                 line_generator: "LineGenerator" = None,
                 number_of_agents=2,
                 obs_builder_object: ObservationBuilder = GlobalObsForRailEnv(),
                 malfunction_generator_and_process_data=None,
                 malfunction_generator: "MalfunctionGenerator" = None,
                 remove_agents_at_target=True,
                 random_seed=None,
                 timetable_generator=ttg.timetable_generator,
                 acceleration_delta=1.0,
                 braking_delta=-1.0,
                 rewards: Rewards = None,
                 effects_generator: EffectsGenerator["RailEnv"] = None,
                 distance_map: AbstractDistanceMap = None,
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
        number_of_agents : int
            Number of agents to spawn on the map. Potentially in the future,
            a range of number of agents to sample from.
        obs_builder_object: ObservationBuilder
            ObservationBuilder-derived object that takes builds observation
            vectors for each agent.
        malfunction_generator_and_process_data : Tuple["MalfunctionGenerator","MalfunctionProcessData"]
            Deprecated. Use `malfunction_generator` option instead.
        malfunction_generator: "MalfunctionGenerator"
            Convenience option to inject effects generator. Defaults to `NoMalfunctionGen`.
        remove_agents_at_target : bool
            If remove_agents_at_target is set to true then the agents will be removed by placing to
            RailEnv.DEPOT_POSITION when the agent has reached its target position.
        random_seed : int or None
            if None, then its ignored, else the random generators are seeded with this number to ensure
            that stochastic operations are replicable across multiple operations
        timetable_generator
            Timetable generator to be used in `reset()`. Defaults to "ttg.timetable_generator".
        acceleration_delta : float
            Determines how much speed is increased by MOVE_FORWARD action up to max_speed set by train's Line (sampled from `speed_ratios` by `LineGenerator`).
            As speed is between 0.0 and 1.0, acceleration_delta=1.0 restores to previous constant speed behaviour
            (i.e. MOVE_FORWARD always sets to max speed allowed for train).
        braking_delta : float
            Determines how much speed is decreased by STOP_MOVING action.
            As speed is between 0.0 and 1.0, braking_delta=-1.0 restores to previous full stop behaviour.
        rewards : DefaultRewards
            The rewards function to use. Defaults to standard settings of Flatland 3 behaviour.
        effects_generator : Optional[EffectsGenerator["RailEnv"]]
            The effects generator that can modify the env at the env of env reset, at the beginning of the env step and at the end of the env step.
        distance_map: AbstractDistanceMap
            Use pre-computed distance map. Defaults to new distance map.
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

        # TODO typing
        self.rail: Optional[RailGridTransitionMap] = None

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

        self.dones = None

        self.action_space = [5]

        self._seed(seed=random_seed)

        self.motion_check = ac.MotionCheck()

        # TODO bad design smell - resource map is not persisted, in particular level_free_positions is not persisted, only rail!
        self.resource_map: UnderlyingResourceMapType = self._extract_resource_map_from_optionals({})

        if rewards is None:
            self.rewards = DefaultRewards()
        else:
            self.rewards = rewards

        self.acceleration_delta = acceleration_delta
        self.braking_delta = braking_delta

        mf = MalfunctionEffectsGenerator(self.malfunction_generator)
        if effects_generator is None:
            self.effects_generator = mf
        else:
            self.effects_generator = make_multi_effects_generator(effects_generator, mf)

        self.temp_transition_data = {i: env_utils.AgentTransitionData(None, None, None, None, None, None, None) for i in range(self.get_num_agents())}
        for i_agent in range(self.get_num_agents()):
            self.temp_transition_data[i_agent].state_transition_signal = StateTransitionSignals()

        self.distance_map = distance_map

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
            optionals, rail = self._call_rail_generator(optionals)
            self.rail = rail

            # Do a new set_env call on the obs_builder to ensure
            # that obs_builder specific instantiations are made according to the
            # specifications of the current environment : like width, height, etc
            self.obs_builder.set_env(self)

        if regenerate_schedule or regenerate_rail or self.get_num_agents() == 0:
            agents_hints = None
            if optionals and 'agents_hints' in optionals:
                agents_hints = optionals['agents_hints']
            self.resource_map = self._extract_resource_map_from_optionals(optionals)

            line = self.line_generator(self.rail, self.number_of_agents, agents_hints, self.num_resets, self.np_random)

            self.agents = EnvAgent.from_line(line)

            # Reset distance map - basically initializing
            self.distance_map.reset(self.agents, self.rail)

            # Timetable Generation
            timetable = self.timetable_generator(self.agents, self.distance_map, agents_hints, self.np_random)

            self._max_episode_steps = timetable.max_episode_steps
            EnvAgent.apply_timetable(self.agents, timetable)
        else:
            self.resource_map = self._extract_resource_map_from_optionals(optionals)
            self.distance_map.reset(self.agents, self.rail)

        # Reset agents to initial states
        self.reset_agents()

        self.num_resets += 1
        self._elapsed_steps = 0

        self.effects_generator.on_episode_start(self)

        self.dones = dict.fromkeys(list(range(self.get_num_agents())) + ["__all__"], False)

        # Reset the state of the observation builder with the new environment
        self.obs_builder.reset()

        # Empty the episode store of agent positions
        self.cur_episode = []

        self.temp_transition_data = {i: env_utils.AgentTransitionData(None, None, None, None, None, None, None) for i in range(self.get_num_agents())}
        for i_agent in range(self.get_num_agents()):
            self.temp_transition_data[i_agent].state_transition_signal = StateTransitionSignals()

        info_dict = self.get_info_dict()
        # Return the new observation vectors for each agent
        observation_dict: Dict = self._get_observations()
        return observation_dict, info_dict

    def _extract_resource_map_from_optionals(self, optionals: dict) -> UnderlyingResourceMapType:
        raise NotImplementedError()

    def clear_rewards_dict(self):
        """ Reset the rewards dictionary """
        self.rewards_dict = {i_agent: self.rewards.empty() for i_agent in range(len(self.agents))}

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
                self.rewards_dict[i_agent] = self.rewards.cumulate(
                    self.rewards_dict[i_agent], self.rewards.end_of_episode_reward(agent, self.distance_map, self._elapsed_steps)
                )
                self.dones[i_agent] = True

            self.dones["__all__"] = True

    def handle_done_state(self, agent):
        """ Any updates to agent to be made in Done state """
        if agent.state == TrainState.DONE and agent.arrival_time is None:
            agent.arrival_time = self._elapsed_steps
            self.dones[agent.handle] = True
            if self.remove_agents_at_target:
                # TODO refactor for configurations
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

        self.motion_check = ac.MotionCheck()  # reset the motion check

        self.effects_generator.on_episode_step_start(self)

        for agent in self.agents:
            i_agent = agent.handle

            current_or_initial_configuration = agent.current_configuration
            initial_configuration = agent.initial_configuration
            agent.old_configuration = agent.current_configuration

            # Get action for the agent
            raw_action = action_dict.get(i_agent, RailEnvActions.DO_NOTHING)
            # Try moving actions on current position
            if current_or_initial_configuration[0] is None:  # Agent not added on map yet
                current_or_initial_configuration = initial_configuration

            _, new_configuration_independent, _, preprocessed_action = self.rail.check_action_on_agent(
                RailEnvActions.from_value(raw_action), current_or_initial_configuration
            )

            # get desired new_position and new_direction
            stop_action_given = preprocessed_action == RailEnvActions.STOP_MOVING
            in_malfunction = agent.malfunction_handler.in_malfunction
            movement_action_given = RailEnvActions.is_moving_action(preprocessed_action)
            earliest_departure_reached = agent.earliest_departure <= self._elapsed_steps
            new_speed = agent.speed_counter.speed
            state = agent.state
            agent_max_speed = agent.speed_counter.max_speed
            # TODO revise design: should we instead of correcting LEFT/RIGHT to FORWARD instead preprocess to DO_NOTHING. Caveat: DO_NOTHING would be undefined for symmetric switches!
            if (state == TrainState.STOPPED or state == TrainState.MALFUNCTION) and movement_action_given:
                # start moving
                new_speed += self.acceleration_delta
            elif preprocessed_action == RailEnvActions.MOVE_FORWARD and raw_action == RailEnvActions.MOVE_FORWARD:
                # accelerate, but not if left/right corrected to forward
                new_speed += self.acceleration_delta
            elif stop_action_given:
                # decelerate
                new_speed += self.braking_delta
            new_speed = max(0.0, min(agent_max_speed, new_speed))

            if state == TrainState.READY_TO_DEPART and movement_action_given:
                new_configuration = initial_configuration
            elif state == TrainState.MALFUNCTION_OFF_MAP and not in_malfunction and earliest_departure_reached and (
                movement_action_given or stop_action_given):
                # TODO revise design: weirdly, MALFUNCTION_OFF_MAP does not go via READY_TO_DEPART, but STOP_MOVING and MOVE_* adds to map if possible
                new_configuration = initial_configuration
            elif state.is_on_map_state():
                new_configuration = current_or_initial_configuration
                # transition to next cell: at end of cell and next state potentially MOVING
                if (agent.speed_counter.is_cell_exit(new_speed)
                    and
                    TrainStateMachine.can_get_moving_independent(state, in_malfunction, movement_action_given, new_speed, stop_action_given)
                ):
                    new_configuration = new_configuration_independent
                # TODO replace with None instead of tuple
                assert agent.current_configuration[0] is not None
            else:
                assert state.is_off_map_state() or state == TrainState.DONE
                # TODO replace with None instead of tuple
                new_configuration = (None, None)

            # TODO replace with None instead of tuple
            if new_configuration[0] is not None:
                valid_position_direction = any(self.rail.get_transitions(new_configuration))
                if not valid_position_direction:
                    warnings.warn(f"{new_configuration} not valid on the grid."
                                  f" Coming from {current_or_initial_configuration} with raw action {raw_action} and preprocessed action {preprocessed_action}. {RailEnvTransitionsEnum(self.rail.get_full_transitions(*agent.position)).name}")
                # fails if initial position has invalid direction
                # assert valid_position_direction

            # only conflict if the level-free cell is traversed through the same axis (horizontally (0 north or 2 south), or vertically (1 east or 3 west)
            current_resource = self.resource_map.get_resource(agent.current_configuration)
            new_resource = self.resource_map.get_resource(new_configuration)

            # Malfunction starts when in_malfunction is set to true (inverse of malfunction_counter_complete)
            self.temp_transition_data[i_agent].state_transition_signal.in_malfunction = agent.malfunction_handler.in_malfunction
            # Earliest departure reached - Train is allowed to move now
            self.temp_transition_data[i_agent].state_transition_signal.earliest_departure_reached = self._elapsed_steps >= agent.earliest_departure
            # Stop action given
            self.temp_transition_data[i_agent].state_transition_signal.stop_action_given = stop_action_given
            # Movement action given
            self.temp_transition_data[i_agent].state_transition_signal.movement_action_given = movement_action_given
            # Target reached - we only know after state and positions update - see handle_done_state below
            self.temp_transition_data[i_agent].state_transition_signal.target_reached = None  # we only know after motion check
            # Movement allowed if inside cell or at end of cell and no conflict with other trains - we only know after motion check!
            self.temp_transition_data[i_agent].state_transition_signal.movement_allowed = None  # we only know after motion check
            # New desired speed zero?
            self.temp_transition_data[i_agent].state_transition_signal.new_speed_zero = new_speed == 0.0

            self.temp_transition_data[i_agent].speed = agent.speed_counter.speed
            self.temp_transition_data[i_agent].current_resource = current_resource

            self.temp_transition_data[i_agent].new_configuration = new_configuration
            self.temp_transition_data[i_agent].new_speed = new_speed
            self.temp_transition_data[i_agent].new_position_level_free = new_resource
            self.temp_transition_data[i_agent].preprocessed_action = preprocessed_action

            self.motion_check.add_agent(i_agent, current_resource, new_resource)

        # Find conflicts between trains trying to occupy same cell
        self.motion_check.find_conflicts()

        have_all_agents_ended = True
        for agent in self.agents:
            i_agent = agent.handle
            current_or_initial_configuration = agent.current_configuration
            initial_configuration = agent.initial_configuration

            # Fetch the saved transition data
            agent_transition_data = self.temp_transition_data[i_agent]

            # motion_check is False if agent wants to stay in the cell
            motion_check = self.motion_check.check_motion(i_agent, agent_transition_data.current_resource)
            # Movement allowed if inside cell or at end of cell and no conflict with other trains
            movement_allowed = (agent.state.is_on_map_state() and not agent.speed_counter.is_cell_exit(agent_transition_data.new_speed)) or motion_check

            agent_transition_data.state_transition_signal.movement_allowed = movement_allowed

            # state machine step
            agent.state_machine.set_transition_signals(agent_transition_data.state_transition_signal)
            agent.state_machine.step()

            # position and speed_counter update
            if agent.state == TrainState.MOVING:
                # only position update while MOVING and motion_check OK
                agent.current_configuration = agent_transition_data.new_configuration
                # N.B. no movement in first time step after READY_TO_DEPART or MALFUNCTION_OFF_MAP!
                if not (agent.state_machine.previous_state == TrainState.READY_TO_DEPART or
                        agent.state_machine.previous_state == TrainState.MALFUNCTION_OFF_MAP):
                    agent.speed_counter.step(speed=agent_transition_data.new_speed)
                # TODO generalize to configuration
                agent.state_machine.update_if_reached(agent.position, agent.target)
            elif agent.state_machine.previous_state == TrainState.MALFUNCTION_OFF_MAP and agent.state == TrainState.STOPPED:
                agent.current_configuration = initial_configuration

            # TODO revise design: condition could be generalized to not MOVING if we would enforce MALFUNCTION_OFF_MAP to go to READY_TO_DEPART first.
            if agent.state.is_on_map_state() and agent.state != TrainState.MOVING:
                agent.speed_counter.step(speed=0)

            # Handle done state actions, optionally remove agents
            self.handle_done_state(agent)
            have_all_agents_ended &= (agent.state == TrainState.DONE)

            ## Update rewards
            self.rewards_dict[i_agent] = self.rewards.cumulate(
                self.rewards_dict[i_agent],
                self.rewards.step_reward(
                    agent=agent,
                    agent_transition_data=agent_transition_data,
                    distance_map=self.distance_map,
                    elapsed_steps=self._elapsed_steps
                )
            )

            # update malfunction counter
            # TODO revise design: updating the malfunction counter after the state transition leaves ugly situation that malfunction_counter == 0 but state is in malfunction - move to begining of step function?
            agent.malfunction_handler.update_counter()

            # Off map or on map state and position should match
            # TODO generalize to configuration
            if not self._fast_state_position_sync_check(agent.state, agent.position, self.remove_agents_at_target):
                agent.state_machine.state_position_sync_check(agent.position, agent.handle, self.remove_agents_at_target)

        # Check if episode has ended and update rewards and dones
        self.end_of_episode_update(have_all_agents_ended)

        self._verify_mutually_exclusive_resource_allocation()

        self.effects_generator.on_episode_step_end(self)

        return self._get_observations(), self.rewards_dict, self.dones, self.get_info_dict()

    @lru_cache()
    def _fast_state_position_sync_check(self, state, position, remove_agents_at_target):
        """ Check for whether on map and off map states are matching with position being None """
        if TrainState.is_on_map_state(state) and position is None:
            return False
        elif TrainState.is_off_map_state(state) and position is not None:
            return False
        elif state == TrainState.DONE and remove_agents_at_target and position is not None:
            return False
        return True

    def _verify_mutually_exclusive_resource_allocation(self):
        resources = [self.resource_map.get_resource(agent.current_configuration) for agent in self.agents if agent.position is not None]
        if len(resources) != len(set(resources)):
            msgs = f"Found two agents occupying same resource (cell or level-free cell) in step {self._elapsed_steps}: {resources}\n"
            msgs += f"- motion check: {list(self.motion_check.stopped)}"
            warnings.warn(msgs)
            counts = {resource: resources.count(resource) for resource in set(resources)}
            dup_resources = [res for res, count in counts.items() if count > 1]
            for dup in dup_resources:
                for agent in self.agents:
                    if self.resource_map.get_resource(agent.current_configuration) == dup:
                        msg = (f"\n================== BAD AGENT ==================================\n\n\n\n\n"
                               f"- agent:\t{agent} \n"
                               f"- state_machine:\t{agent.state_machine}\n"
                               f"- speed_counter:\t{agent.speed_counter}\n"
                               f"- breakpoint:\tself._elapsed_steps == {self._elapsed_steps} and agent.handle == {agent.handle}\n"
                               f"- motion check:\t{list(self.motion_check.stopped)}\n\n\n"
                               f"- agents:\t{self.agents}")
                        warnings.warn(msg)
                        msgs += msg
            assert len(resources) == len(set(resources)), msgs

    # TODO https://github.com/flatland-association/flatland-rl/issues/195  extract to callbacks instead!
    def record_timestep(self, dActions):
        """
        Record the positions and orientations of all agents in memory, in the cur_episode
        """
        list_agents_state = []
        for i_agent in range(self.get_num_agents()):
            agent = self.agents[i_agent]
            # the int cast is to avoid numpy types which may cause problems with msgpack
            # in env v2, agents may have position None, before starting
            # TODO test for configuration None instead
            if agent.position is None:
                pos = (None, None)
                dir = None
            else:
                pos = (int(agent.position[0]), int(agent.position[1]))
                dir = int(agent.direction)
            # print("pos:", pos, type(pos[0]))
            list_agents_state.append([
                *pos, dir,
                agent.malfunction_handler.malfunction_down_counter,
                agent.state.value,
                int(agent.position in self.motion_check.deadlocked),
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

    def _call_rail_generator(self, optionals) -> Tuple[dict, UnderlyingTransitionMapType]:
        # TODO https://github.com/flatland-association/flatland-rl/issues/242 fix signature
        return self.rail_generator(self.number_of_agents, self.num_resets, self.np_random)


class RailEnv(AbstractRailEnv[GridTransitionMap, GridResourceMap]):
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
        All parameters from parent `AbstractRailEnv`

        Parameters
        ----------
        width : int
            The width of the rail map. Potentially in the future,
            a range of widths to sample from.
        height : int
            The height of the rail map. Potentially in the future,
            a range of heights to sample from.
        """
        self.width = width
        self.height = height

        super().__init__(
            rail_generator=rail_generator,
            line_generator=line_generator,
            number_of_agents=number_of_agents,
            obs_builder_object=obs_builder_object,
            malfunction_generator_and_process_data=malfunction_generator_and_process_data,
            malfunction_generator=malfunction_generator,
            remove_agents_at_target=remove_agents_at_target,
            random_seed=random_seed,
            timetable_generator=timetable_generator,
            acceleration_delta=acceleration_delta,
            braking_delta=braking_delta,
            rewards=rewards,
            effects_generator=effects_generator,
            distance_map=DistanceMap([], height, width),
        )

        self.agent_positions = None

        # save episode timesteps ie agent positions, orientations.  (not yet actions / observations)
        self.record_steps = record_steps  # whether to save timesteps
        # save timesteps in here: [[[row, col, dir, malfunction],...nAgents], ...nSteps]
        self.cur_episode = []
        self.list_actions = []  # save actions in here

        # Agent positions map
        self.agent_positions = np.zeros((self.height, self.width), dtype=int) - 1
        self._update_agent_positions_map(ignore_old_positions=False)

    def _call_rail_generator(self, optionals) -> Tuple[dict, GridTransitionMap]:
        if "__call__" in dir(self.rail_generator):
            rail, optionals = self.rail_generator(
                self.width, self.height, self.number_of_agents, self.num_resets, self.np_random)
        elif "generate" in dir(self.rail_generator):
            rail, optionals = self.rail_generator.generate(
                self.width, self.height, self.number_of_agents, self.num_resets, self.np_random)
        else:
            raise ValueError("Could not invoke __call__ or generate on rail_generator")
        self.height, self.width = rail.grid.shape
        if optionals and 'distance_map' in optionals:
            self.distance_map.set(optionals['distance_map'])

        return optionals, rail

    def _extract_resource_map_from_optionals(self, optionals: dict) -> GridResourceMap:
        resource_map = GridResourceMap()
        if optionals and 'level_free_positions' in optionals:
            resource_map.level_free_positions = optionals['level_free_positions']
        return resource_map

    def _update_agent_positions_map(self, ignore_old_positions=True):
        """ Update the agent_positions array for agents that changed positions """
        for agent in self.agents:
            # TODO refactor for configurations
            if not ignore_old_positions or agent.old_position != agent.position:
                if agent.position is not None:
                    self.agent_positions[agent.position] = agent.handle
                if agent.old_position is not None:
                    self.agent_positions[agent.old_position] = -1

    def clone_from(self, env: 'RailEnv', obs_builder: Optional[ObservationBuilder["RailEnv", Any]] = None):
        from flatland.envs.persistence import RailEnvPersister
        # avoid in-memory references
        env_dict = pickle.loads(pickle.dumps(RailEnvPersister.get_full_state(env)))
        RailEnvPersister.load(self, env_dict=env_dict, obs_builder=obs_builder)

    def step(self, action_dict: Dict[int, RailEnvActions]):
        obs, rewards, dones, info = super().step(action_dict=action_dict)
        # TODO https://github.com/flatland-association/flatland-rl/issues/195 add idiomatic wrapper instead of override
        if self.record_steps:
            self.record_timestep(action_dict)
        # TODO add idiomatic wrapper instead of override
        self._update_agent_positions_map()
        return obs, rewards, dones, info
