"""
Definition of the RailEnv environment.
"""
import pickle
import random
import warnings
from fractions import Fraction
from functools import lru_cache
from typing import List, Optional, Dict, Tuple, Any, Generic, TypeVar, NamedTuple

import numpy as np

import flatland.envs.timetable_generators as ttg
from flatland.core.distance_map import AgentSourceTargetDistanceMap
from flatland.core.effects_generator import EffectsGenerator, find_effects_generator, make_multi_effects_generator
from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid_resource_map import GridResourceMap
from flatland.core.resource_map import ResourceMap
from flatland.core.transition_map import GridTransitionMap, TransitionMap
from flatland.envs import agent_chains as ac
from flatland.envs import line_generators as line_gen
from flatland.envs import malfunction_effects_generators as mfg
from flatland.envs import malfunction_generators as mal_gen
from flatland.envs import rail_generators as rail_gen
from flatland.envs.agent_utils import EnvAgent, _filter_valid_target_entry_points, _sanitize_entry_point
from flatland.envs.grid.distance_map import DistanceMap
from flatland.envs.grid.rail_env_grid import RailEnvTransitionsEnum
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.record_steps_effects_generator import RecordStepsEffectsGenerator
from flatland.envs.rewards import DefaultRewards, Rewards
from flatland.envs.step_utils import env_utils
from flatland.envs.step_utils.speed_counter import _cap_speed, SEGMENT_LENGTH, SpeedCounter
from flatland.envs.step_utils.states import TrainState, StateTransitionSignals
from flatland.utils import seeding

TransitionMapT = TypeVar('TransitionMapT', bound=TransitionMap)
ResourceMapT = TypeVar('ResourceMapT', bound=ResourceMap)
EntryPointT = TypeVar('EntryPointT')


class PreStepSnapshot(NamedTuple):
    """ Per-agent state captured before `step()` runs, for `AbstractRailEnv._check_post_speed_invariants()`
    to verify the post-step speed update against. """
    pre_speeds: Dict[int, Optional[Fraction]]
    pre_current_entry_points: Dict[int, Optional[EntryPointT]]
    pre_next_entry_points: Dict[int, Optional[EntryPointT]]
    pre_dones: Dict[int, bool]
    pre_in_malfunctions: Dict[int, bool]
    pre_offsets: Dict[int, Optional[Fraction]]


class AbstractRailEnv(Environment, Generic[TransitionMapT, ResourceMapT, EntryPointT]):
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

    Parameters
    ----------
    rail_generator : function
        The rail_generator function is a function that takes the width,
        height and agent handles of a rail environment, along with the number of times
        the env has been reset, and returns a GridTransitionMap object and a list of
        starting positions, targets, and initial orientations for agent handles.
        The rail_generator can pass a distance map in the hints or information for specific line_generators.
        Implementations can be found in flatland/envs/rail_generators.py
    line_generator : function
        The line_generator function is a function that takes the grid, the number of agents and optional hints
        and returns a list of starting positions, targets, initial orientations and maximum speeds for all agent handles.
        Implementations can be found in flatland/envs/line_generators.py
    number_of_agents : int
        Number of agents to spawn on the map. Potentially in the future,
        a range of number of agents to sample from.
    obs_builder_object: ObservationBuilder
        ObservationBuilder-derived object that builds observation
        vectors for each agent.
    malfunction_generator_and_process_data : Tuple["MalfunctionGenerator","MalfunctionProcessData"]
        Deprecated. Use `malfunction_generator` option instead.
    malfunction_generator: "MalfunctionGenerator"
        Convenience option to inject effects generator. Defaults to `NoMalfunctionGen`.
    remove_agents_at_target : bool
        If remove_agents_at_target is set to true then the agents will be removed by placing to
        RailEnv.DEPOT_POSITION when the agent has reached its target position.
    random_seed : int or None
        if None, then it is ignored, else the random generators are seeded with this number to ensure
        that stochastic operations are replicable across multiple operations
    timetable_generator
        Timetable generator to be used in `reset()`. Defaults to "ttg.timetable_generator".
    acceleration_delta : float
        Determines how much speed is increased by MOVE_FORWARD action up to max_speed set by train's Line (sampled from `speed_ratios` by `LineGenerator`).
        As speed is between 0.0 and 1.0, acceleration_delta=1.0 restores the previous constant speed behaviour
        (i.e. MOVE_FORWARD always sets to max speed allowed for train).
    braking_delta : float
        Determines how much speed is decreased by STOP_MOVING action.
        As speed is between 0.0 and 1.0, braking_delta=-1.0 restores to previous full stop behaviour.
    check_step_pre_post_conditions : bool
        Set to False to skip checking step() pre- and postconditions, e.g. in performance-sensitive production use.
    rewards : DefaultRewards
        The rewards function to use. Defaults to standard settings of Flatland 3 behaviour.
    effects_generator : Optional[EffectsGenerator["RailEnv"]]
        The effects generator that can modify the env at the end of env reset, at the beginning of the env step and at the end of the env step.
    distance_map: AgentSourceTargetDistanceMap
        Use pre-computed distance map. Defaults to new distance map.
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
                 acceleration_delta: Fraction = Fraction(1),
                 braking_delta: Fraction = -Fraction(1),
                 check_step_pre_post_conditions: bool = True,
                 rewards: Rewards = None,
                 effects_generator: EffectsGenerator["RailEnv"] = None,
                 distance_map: AgentSourceTargetDistanceMap = None,
                 ):

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

        self.rail: Optional[TransitionMapT] = None
        self.stations_links = None

        self.remove_agents_at_target = remove_agents_at_target

        self.obs_builder = obs_builder_object

        self._max_episode_steps: Optional[int] = None
        self._elapsed_steps = 0

        self.obs_dict = {}
        self.rewards_dict = {}
        self.dev_obs_dict = {}
        self.dev_pred_dict = {}

        self.agents: List[EnvAgent[EntryPointT]] = []
        self.num_resets = 0

        self.dones = None

        self.action_space = [5]

        self._seed(seed=random_seed)

        self.resource_check = ac.MotionCheck()

        # TODO https://github.com/flatland-association/flatland-rl/issues/242 bad design smell - resource map is not persisted, in particular level_free_positions is not persisted, only rail!
        self.resource_map: ResourceMapT = self._extract_resource_map_from_optionals({})

        if rewards is None:
            self.rewards = DefaultRewards()
        else:
            self.rewards = rewards

        self.acceleration_delta = acceleration_delta
        self.braking_delta = braking_delta
        self.check_step_pre_post_conditions = check_step_pre_post_conditions

        mf = mfg.MalfunctionEffectsGenerator(self.malfunction_generator)
        if effects_generator is None:
            self.effects_generator = mf
        else:
            self.effects_generator = make_multi_effects_generator(effects_generator, mf)

        self.temp_transition_data = {i: env_utils.AgentTransitionData(None, None, None) for i in range(self.get_num_agents())}
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
            self.stations_links = None
            optionals, rail = self._call_rail_generator(optionals)
            self.rail = rail

        if regenerate_schedule or regenerate_rail or self.get_num_agents() == 0:
            agents_hints = None
            if optionals and 'agents_hints' in optionals:
                agents_hints = optionals['agents_hints']
            self.resource_map = self._extract_resource_map_from_optionals(optionals)

            line = self.line_generator(self.rail, self.number_of_agents, agents_hints, self.num_resets, self.np_random)

            self.agents = self._agents_from_line(line, self.rail)

            # Reset distance map - basically initializing
            self.distance_map.reset(self.agents, self.rail)

            # Timetable Generation
            timetable = self.timetable_generator(self.agents, self.distance_map, agents_hints, self.np_random)

            self._max_episode_steps = timetable.max_episode_steps
            self.agents = self._apply_timetable_to_agents(self.agents, timetable)
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
        self.obs_builder.reset(self)

        # Empty the episode store of agent positions
        self.cur_episode = []

        self.temp_transition_data = {i: env_utils.AgentTransitionData(None, None, None) for i in range(self.get_num_agents())}
        for i_agent in range(self.get_num_agents()):
            self.temp_transition_data[i_agent].state_transition_signal = StateTransitionSignals()

        info_dict = self.get_info_dict()
        # Return the new observation vectors for each agent
        observation_dict: Dict = self._get_observations()
        return observation_dict, info_dict

    def _extract_resource_map_from_optionals(self, optionals: dict) -> ResourceMapT:
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
            'action_required': {i: RailEnv.action_required(agent.state, agent.speed_counter.is_cell_exit())
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
            # capture which specific target alternative was reached before current_entry_point is
            # possibly cleared below - see EnvAgent.target_entry_point.
            agent.target_entry_point = agent.current_entry_point
            self.dones[agent.handle] = True
            if self.remove_agents_at_target:
                agent.current_entry_point = None
                agent.next_entry_point = None
                # design: distance is None when off map -- passing speed=None sets distance back
                # to None exactly when the agent's position leaves the map.
                agent.speed_counter.step(speed=None, crossing_completed=False)

    def step(self, action_dict: Dict[int, RailEnvActions]):
        """
        Updates rewards for the agents at a step.
        """
        self._elapsed_steps += 1

        # Not allowed to step further once done
        if self.dones["__all__"]:
            raise Exception("Episode is done, cannot call step()")

        pre_step_snapshot = self._check_pre_step_invariants_and_capture_snapshot() \
            if self.check_step_pre_post_conditions else None

        self.clear_rewards_dict()

        self.resource_check = ac.MotionCheck()  # reset the motion check

        for agent in self.agents:
            # (0a) UPDATE MALFUNCTION COUNTER
            # N.B. in order to keep malfunction counter and agent state in sync
            agent.malfunction_handler.update_counter()

        # (0b) GENERATE NEW MALFUNCTIONS AND OTHER RANDOM EFFECTS
        self.effects_generator.on_episode_step_start(self)

        for agent in self.agents:
            i_agent = agent.handle

            initial_entry_point = agent.initial_entry_point
            agent.old_entry_point = agent.current_entry_point

            action = RailEnvActions.from_value(action_dict.get(i_agent, RailEnvActions.DO_NOTHING))

            # N.B. every candidate_ variable in this loop (candidate_speed, candidate_entry_point,
            # candidate_entry_point_independent, candidate_next_entry_point, ...) reflects
            # the unilateral update of the collect phase (loop 1) - computed from the action alone,
            # including for an invalid action, which itself just yields a
            # zeroed/unchanged candidate (e.g. candidate_speed = 0) rather than skipping computation
            # entirely. Distribute phase (loop 2) checks whether the resource check actually granted it.

            # Invariant: both None off-map, both set and different on-map).

            # (1) STATE TRANSITION SIGNALS
            stop_action_given = action == RailEnvActions.STOP_MOVING
            in_malfunction = agent.malfunction_handler.in_malfunction
            # design (issue #280): an earliest_departure=0 agent never goes through a state_machine.step()
            # call before the very first step() runs - tweak state directly here, before this step's own
            # state read below, so it already sees READY_TO_DEPART instead of stale WAITING (symmetric
            # with MALFUNCTION_OFF_MAP's own straight-to-MOVING shortcut in _handle_malfunction_off_map).
            if self._elapsed_steps == 1 and agent.earliest_departure == 0 and not in_malfunction and agent.state == TrainState.WAITING:
                agent.state_machine.set_state(TrainState.READY_TO_DEPART)
            movement_action_given = RailEnvActions.is_moving_action(action)
            # design (issue #280): earliest_departure_reached is signalled one step earlier,
            # so the WAITING -> READY_TO_DEPART transition it drives in
            # the state machine completes in N-1, so the agens is READY_TO_DEPART in step N and can enter
            # the grid in step N
            earliest_departure_reached = agent.earliest_departure <= self._elapsed_steps + 1
            state = agent.state

            # (2) CANDIDATE ENTRY POINT: action validity - need both by speed update (3a) and position update (3b) below
            is_on_map = agent.next_entry_point is not None
            # whether the action leads to a valid transition
            if is_on_map:
                candidate_entry_point_independent = self.rail.apply_action_independent(action, agent.next_entry_point)
            else:
                # TODO this is wrong: if done, we should not try to reservee the initial edge!
                candidate_entry_point_independent = self.rail.apply_action_independent(action, initial_entry_point)

            # mid cell or valid transition (only invalid actions are non-L/R on symmetric switches)
            is_cell_exit = agent.speed_counter.is_cell_exit()
            action_valid = not is_cell_exit or candidate_entry_point_independent is not None

            # (3a) SPEED UPDATE
            # N.B. new speed is only applied if MOVING state and previous was not READY_TO_DEPART/MALFUNCTION_OFF_MAP, see below

            # N.B. no acceleration if the action isn't (corrected to) MOVE_FORWARD, e.g. facing a
            # symmetric switch with the action corrected to STOP_MOVING, or MOVE_LEFT/MOVE_RIGHT
            # corrected to MOVE_FORWARD but not accelerated as the original action wasn't forward.
            # get desired candidate speed independent of resource check
            agent_max_speed = agent.speed_counter.max_speed
            # (3a.1) done
            if state == TrainState.DONE:
                candidate_speed = Fraction(0)
            # (3a.2) malfunction
            elif in_malfunction:
                candidate_speed = Fraction(0)
            # (3a.3) map entry
            elif not is_on_map and movement_action_given and earliest_departure_reached:
                candidate_speed = self.acceleration_delta
            # (3a.4) stay off map
            elif not is_on_map:
                candidate_speed = Fraction(0)
            # (3a.5) invalid action
            elif is_on_map and candidate_entry_point_independent is None and is_cell_exit:
                candidate_speed = Fraction(0)
            # (3a.6) accelerate upon forward
            elif action == RailEnvActions.MOVE_FORWARD:
                candidate_speed = agent.speed_counter.speed + self.acceleration_delta
            # (3a.7) start moving
            elif agent.speed_counter.speed == 0 and movement_action_given:
                candidate_speed = agent.speed_counter.speed + self.acceleration_delta
            # (3a.8) braking
            elif stop_action_given:
                # decelerate
                candidate_speed = agent.speed_counter.speed + self.braking_delta
            # (3a.9) default
            else:
                candidate_speed = agent.speed_counter.speed
            candidate_speed = _cap_speed(agent_max_speed, candidate_speed)

            # (3b) POSITION UPDATE - mirrors _check_post_position_invariants's "candidates accepted"
            # derivation: done/malfunction/map-entry/on-map-transition/stay.
            # (3b.1) done
            if state == TrainState.DONE:
                # design: for remove_agents_at_target=True, agent.current_entry_point is already
                # None (handle_done_state() cleared it on removal) - a no-op. For
                # remove_agents_at_target=False, this reserves the agent's occupied target cell as
                # both its current and candidate resource
                candidate_entry_point = agent.current_entry_point
                candidate_next_entry_point = agent.next_entry_point
            # (3b.2) malfunction - takes priority over (3b.3)/(3b.5) below: in_malfunction is resolved
            # before this loop runs (see (0a)/(0b)), so it can be True while `state` (read above, still
            # the pre-transition value) hasn't caught up to MALFUNCTION/MALFUNCTION_OFF_MAP yet -
            # without this priority, a same-step malfunction onset could let a genuine crossing through.
            elif in_malfunction:
                candidate_entry_point = agent.current_entry_point
                candidate_next_entry_point = agent.next_entry_point
            #  (3b.3) map entry
            elif action_valid and (
                (state == TrainState.READY_TO_DEPART and movement_action_given)
                # design (issue #280): MALFUNCTION_OFF_MAP intentionally does not go via READY_TO_DEPART -
                # STOP_MOVING and MOVE_* both add to map directly if possible, same as MOVING's own
                # straight-to-MOVING shortcut in _handle_malfunction_off_map.
                or (state == TrainState.MALFUNCTION_OFF_MAP and earliest_departure_reached
                    and (movement_action_given or stop_action_given))
            ):
                candidate_entry_point = initial_entry_point
                candidate_next_entry_point = candidate_entry_point_independent
            # (3b.5) on-map cell transition
            elif is_on_map and is_cell_exit and candidate_entry_point_independent is not None and state == TrainState.MOVING:
                # design: actions applied at cell entry -- attempt the already-decided target
                # (guaranteed on-map by is_on_map above); this step's action instead decides the
                # look-ahead beyond it (candidate_entry_point_independent, computed above in (2)).
                candidate_entry_point = agent.next_entry_point
                candidate_next_entry_point = candidate_entry_point_independent
            # (3b.2bis/3b.4/3b.6) unchanged - start-moving-from-stop, off-map-stay, and cell-stay
            # (mid-cell / attempted-but-denied) all set the identical "keep current" candidate - a
            # pure identity copy of already-valid current/next_entry_point, so it can't violate the
            # off/on-map invariant that already held.
            else:
                candidate_entry_point = agent.current_entry_point
                candidate_next_entry_point = agent.next_entry_point

            if self.check_step_pre_post_conditions:
                self._check_off_on_map_invariant(candidate_entry_point, candidate_next_entry_point)

            if candidate_entry_point is not None:
                valid_position_direction = any(self.rail.get_transitions(candidate_entry_point))
                if not valid_position_direction:
                    infra = self._infrastructure_representation(agent.current_entry_point)
                    warnings.warn(f"{candidate_entry_point} not valid on the grid."
                                  f" Coming from {agent.current_entry_point if state.is_on_map_state() else initial_entry_point} with action {action}"
                                  f" and action valid {action_valid}. {infra}")
                # fails if initial position has invalid direction or if the grid is not closed
                # assert valid_position_direction

            # (4) MOTION/RESOURCE CHECK
            # only conflict if the level-free cell is traversed through the same axis (horizontally (0 north or 2 south), or vertically (1 east or 3 west)
            current_resource = self.resource_map.get_resource(agent.current_entry_point, agent.next_entry_point)
            new_resource = self.resource_map.get_resource(candidate_entry_point, candidate_next_entry_point)

            # (5) GATHER STATE TRANSITION SIGNALS
            # Malfunction starts when in_malfunction is set to true (inverse of malfunction_counter_complete)
            self.temp_transition_data[i_agent].state_transition_signal.in_malfunction = agent.malfunction_handler.in_malfunction
            # Earliest departure reached - Train is allowed to move now
            self.temp_transition_data[i_agent].state_transition_signal.earliest_departure_reached = self._elapsed_steps + 1 >= agent.earliest_departure
            # Stop action given
            self.temp_transition_data[i_agent].state_transition_signal.stop_action_given = stop_action_given
            # Movement action given
            self.temp_transition_data[i_agent].state_transition_signal.movement_action_given = movement_action_given
            # Target reached - we only know after state and positions update - see handle_done_state below
            self.temp_transition_data[i_agent].state_transition_signal.target_reached = None  # we only know after motion check

            # action_valid allowed if both
            # - action leading to valid next cell
            # - inside cell or at end of cell and no conflict with other trains
            self.temp_transition_data[i_agent].state_transition_signal.action_valid = action_valid
            self.temp_transition_data[i_agent].state_transition_signal.movement_allowed = action_valid  # remainder we only know after motion check!
            # New desired speed zero?
            self.temp_transition_data[i_agent].state_transition_signal.new_speed_zero = self._is_speed_zero(candidate_speed)

            self.temp_transition_data[i_agent].speed = agent.speed_counter.speed

            # design: actions applied at cell entry -- carry this step's attempted target and
            # look-ahead candidate via per-step scratch data; loop 2 decides whether to promote
            # the candidate into agent.next_entry_point once the attempt's outcome (motion check)
            # is known. agent.next_entry_point itself is left untouched here so it still holds the
            # value being contested until that outcome is known.
            self.temp_transition_data[i_agent].candidate_entry_point = candidate_entry_point
            self.temp_transition_data[i_agent].candidate_next_entry_point = candidate_next_entry_point
            self.temp_transition_data[i_agent].candidate_speed = candidate_speed

            self.resource_check.add_agent(i_agent, current_resource, new_resource)

        # (6) RESOURCE CONFLICT RESOLUTION
        # Find conflicts between trains trying to occupy same cell
        self.resource_check.find_conflicts()

        have_all_agents_ended = True
        for agent in self.agents:
            i_agent = agent.handle

            # (7) FETCH THE SAVED TRANSITION DATA FOR AGENT
            agent_transition_data = self.temp_transition_data[i_agent]
            candidate_entry_point = agent_transition_data.candidate_entry_point

            # (8) FETCH CONFLICT RESOLUTION FOR AGENT AND FINALIZE STATE TRANSITION SIGNALS FROM MOTION_CHECK
            resource_check = self.resource_check.check_resource(i_agent)

            if not agent.speed_counter.is_cell_exit() and agent.state.is_on_map_state():
                assert resource_check == True

            # design (D1/D2): a STOPPED/MALFUNCTION agent given a movement action self-loops into
            # MotionCheck (see (3b.2bis)/(3b.5)/(3b.6)), so resource_check here is trivially granted
            # regardless of whether its target is actually free - movement_allowed (and so the state
            # machine's STOPPED/MALFUNCTION->MOVING promotion) is granted optimistically on the
            # operator's request. Position/distance stay deferred either way (see (10a)/(10b)'s
            # speed==0 handling) - if the target is genuinely still occupied, the *next* step (now
            # pre-speed > 0) attempts the crossing for real via (3b.5), gets denied by MotionCheck's
            # real (non-self-loop) resolution, and the state machine demotes back to STOPPED then
            # (see _handle_moving's `not movement_allowed` branch in state_machine.py) - one step of
            # MOVING with no actual progress, rather than never promoting at all.
            movement_allowed = agent_transition_data.state_transition_signal.action_valid and resource_check
            agent_transition_data.state_transition_signal.movement_allowed = movement_allowed
            agent_transition_data.resource_check = resource_check

            # (9) STATE MACHINE STEP
            agent.state_machine.set_transition_signals(agent_transition_data.state_transition_signal)
            agent.state_machine.step()

            # (10a) POSITION UPDATE
            # INVARIANT: agent.current_entry_point and agent.next_entry_point are always both None
            # (agent off-map) XOR both not None (agent on-map); while on-map, the two are always
            # updated together, on every crossing, and next_entry_point must never equal
            # current_entry_point - there is no "nothing pending" sentinel: entering a new cell is
            # only ever committed together with a valid, genuinely different candidate for what lies
            # beyond it (see (3b) above, where the crossing itself is only attempted if that candidate
            # exists - so candidate_entry_point below is guaranteed non-None whenever entering_new_cell
            # is True).
            if agent.state == TrainState.MOVING and agent_transition_data.speed == 0:
                # design (D1): pre-step speed 0 (STOPPED/MALFUNCTION promoted to MOVING this step) -
                # this step only grants permission to move; it contributes zero speed/distance itself,
                # so no crossing may be committed yet. Position/next_entry_point stay exactly as they
                # were, deferred to the following (genuinely pre-speed>0) step. See (10b) below for the
                # matching distance/speed treatment.
                pass
            elif agent.state == TrainState.MOVING or (
                agent.state == TrainState.STOPPED and agent.state_machine.previous_state == TrainState.MOVING
                and resource_check
            ):
                entering_new_cell = agent.current_entry_point != candidate_entry_point
                agent.current_entry_point = _sanitize_entry_point(candidate_entry_point)
                if entering_new_cell:
                    assert agent_transition_data.candidate_next_entry_point is not None
                    agent.next_entry_point = _sanitize_entry_point(agent_transition_data.candidate_next_entry_point)
                agent.state_machine.update_if_reached(agent.current_entry_point, agent.targets)

            # (10b) SPEED_COUNTER UPDATE (SPEED AND DISTANCE)
            if agent.state == TrainState.MOVING and agent_transition_data.speed == 0:
                # design (D1): pre-step speed 0 -> distance held unchanged this step
                # (SpeedCounter.step() with crossing_completed=False and old self._speed==0 is a no-op
                # on distance); speed ramps from rest via this step's candidate_speed
                # (acceleration_delta). The following step, now genuinely pre-speed>0, is the one that
                # actually completes the crossing (see (3b.5)/(10a) above).
                agent.speed_counter.step(speed=agent_transition_data.candidate_speed, crossing_completed=False)
            elif agent.state == TrainState.MOVING:
                crossing_completed = (agent.old_entry_point != candidate_entry_point) and resource_check
                # N.B. map entry (previous state READY_TO_DEPART/MALFUNCTION_OFF_MAP) also flows through
                # here: candidate_speed is already acceleration_delta (see (3a.3)), and speed_counter.step()
                # bootstraps distance to 0 for an off-map -> on-map transition regardless of crossing_completed.
                speed = agent_transition_data.candidate_speed
                # design: reaching or remaining in MOVING requires movement_allowed (see (9), state_machine's
                # _handle_moving/_handle_stopped/_handle_ready_to_depart/_handle_malfunction_off_map/
                # _handle_malfunction all gate a transition into MOVING on movement_action_given and
                # movement_allowed) - so action_valid and resource_check are always True here; distance
                # advances by the pre-step speed and, if the crossing genuinely completed
                # (agent.old_entry_point != candidate_entry_point), wraps into the new cell
                # (distance % SEGMENT_LENGTH) - otherwise it's a normal within-cell advance.
                agent.speed_counter.step(speed=speed, crossing_completed=crossing_completed)
            elif agent.state == TrainState.STOPPED and agent.state_machine.previous_state == TrainState.MOVING:
                crossing_completed = (agent.old_entry_point != candidate_entry_point) and resource_check
                speed = Fraction(0)
                if not agent_transition_data.state_transition_signal.action_valid or not resource_check:
                    # MOVING -> STOPPED via invalid action at the cell boundary, or resource_check
                    # denying the crossing to another agent: distance still advances by the pre-step
                    # speed (SpeedCounter.step() -> _distance_update() adds pre-step speed
                    # unconditionally) but is clamped at the cell boundary (min(pre_distance +
                    # pre_speed, SEGMENT_LENGTH)) instead of wrapping into the new cell - the agent
                    # travels up to the boundary this step, it just isn't credited with crossing it.
                    # This banking is intentional, not a (D1) violation, in both cases: pre_speed is
                    # always > 0 here (a MOVING agent's pre-step speed can never be 0), so the advance
                    # is real. What (D1) rules out is a *later*, pre-speed-0 STOPPED->MOVING step
                    # consuming this banked distance for a free crossing - prevented by (10a)/(10b)'s
                    # pre-speed-0 deferral above.
                    agent.speed_counter.step(speed=speed, crossing_completed=crossing_completed)
                else:
                    # MOVING -> STOPPED via an explicit STOP_MOVING action that braked candidate_speed to
                    # exactly 0 this step, action valid and resource_check granted: distance advances by
                    # the pre-step speed and, if the crossing genuinely completed, wraps into the new cell
                    # (distance % SEGMENT_LENGTH) - the agent can still complete an already-in-flight cell
                    # crossing on the very step it comes to a stop.
                    agent.speed_counter.step(speed=speed, crossing_completed=crossing_completed)
            elif agent.state.is_on_map_state():
                # TODO harmonize condition with overleaf - force stop or malfunction
                agent.speed_counter.stop()
            elif agent.state.is_off_map_state():
                # design: speed and distance are None while off map
                agent.speed_counter.step(speed=None, crossing_completed=False)
            elif not self.remove_agents_at_target:
                # design: DONE but not removed is neither on-map nor off-map (see
                # TrainState.is_on_map_state()/is_off_map_state()) - position stays put, so freeze
                # speed at 0 instead of setting distance to None (agent.speed_counter.step(None, ...)
                # is reserved for when the position itself leaves the map, see handle_done_state()).
                assert agent.state == TrainState.DONE
                agent.speed_counter.step(speed=Fraction(0), crossing_completed=False)
            # and calls agent.speed_counter.step(speed=None, ...) itself.

            # (11) HANDLE DONE STATE ACTIONS, OPTIONALLY REMOVE AGENTS
            self.handle_done_state(agent)
            have_all_agents_ended &= (agent.state == TrainState.DONE)

            # (12) UPDATE REWARDS
            self.rewards_dict[i_agent] = self.rewards.cumulate(
                self.rewards_dict[i_agent],
                self.rewards.step_reward(
                    agent=agent,
                    agent_transition_data=agent_transition_data,
                    distance_map=self.distance_map,
                    elapsed_steps=self._elapsed_steps
                )
            )

            # Off map or on map state and position should match
            if not self._fast_state_position_sync_check(agent.state, agent.current_entry_point, self.remove_agents_at_target):
                agent.state_machine.state_position_sync_check(agent.current_entry_point, agent.handle, self.remove_agents_at_target)

        # Check if episode has ended and update rewards and dones
        self.end_of_episode_update(have_all_agents_ended)

        self.effects_generator.on_episode_step_end(self, action_dict=action_dict)

        if self.check_step_pre_post_conditions:
            self._check_malfunction_state_invariant()
            self._verify_mutually_exclusive_resource_allocation()
            self._check_post_speed_distance_invariants(action_dict, pre_step_snapshot)
            self._check_post_position_invariants(action_dict, pre_step_snapshot)

        return self._get_observations(), self.rewards_dict, self.dones, self.get_info_dict()

    def _check_malfunction_state_invariant(self):
        """
        Guards against the malfunction_down_counter reaching 0 (in_malfunction False) one step before
        the state machine has transitioned agent.state off MALFUNCTION/MALFUNCTION_OFF_MAP - eliminated
        by decrementing the counter at the start of step(), before state_machine.step() runs.
        """
        for agent in self.agents:
            if agent.state == TrainState.DONE:
                continue
            assert (agent.state in (TrainState.MALFUNCTION, TrainState.MALFUNCTION_OFF_MAP)) == agent.malfunction_handler.in_malfunction

    @lru_cache()
    def _is_speed_zero(self, candidate_speed: Fraction) -> bool:
        return candidate_speed == 0.0

    @lru_cache()
    def _fast_state_position_sync_check(self, state, entry_point, remove_agents_at_target):
        """ Check for whether on map and off map states are matching with position being None """
        if TrainState.is_on_map_state(state) and entry_point is None:
            return False
        elif TrainState.is_off_map_state(state) and entry_point is not None:
            return False
        elif state == TrainState.DONE and remove_agents_at_target and entry_point is not None:
            return False
        return True

    def _verify_mutually_exclusive_resource_allocation(self):
        resources = [self.resource_map.get_resource(agent.current_entry_point, agent.next_entry_point)
                     for agent in self.agents if agent.current_entry_point is not None]
        if len(resources) != len(set(resources)):
            msgs = f"Found two agents occupying same resource (cell or level-free cell) in step {self._elapsed_steps}: {resources}\n"
            msgs += f"- motion check: {list(self.resource_check.stopped)}"
            warnings.warn(msgs)
            counts = {resource: resources.count(resource) for resource in set(resources)}
            dup_resources = [res for res, count in counts.items() if count > 1]
            for dup in dup_resources:
                for agent in self.agents:
                    if self.resource_map.get_resource(agent.current_entry_point, agent.next_entry_point) == dup:
                        msg = (f"\n================== BAD AGENT ==================================\n\n\n\n\n"
                               f"- agent:\t{agent} \n"
                               f"- state_machine:\t{agent.state_machine}\n"
                               f"- speed_counter:\t{agent.speed_counter}\n"
                               f"- breakpoint:\tself._elapsed_steps == {self._elapsed_steps} and agent.handle == {agent.handle}\n"
                               f"- motion check:\t{list(self.resource_check.stopped)}\n\n\n"
                               f"- agents:\t{self.agents}")
                        warnings.warn(msg)
                        msgs += msg
            assert len(resources) == len(set(resources)), msgs

    def _check_pre_step_invariants_and_capture_snapshot(self) -> PreStepSnapshot:
        """
        Verify the current_entry_point/next_entry_point invariant holds before `step()` runs, and
        capture per-agent speed/entry-point/done state for `_check_post_speed_invariants()` to later
        verify the post-step speed update against.
        """
        for agent in self.agents:
            # invariant: current_entry_point/next_entry_point are either both None (off-map) or
            # both set and different (on-map, next_entry_point strictly ahead of current_entry_point)
            # - see the design note in step()'s (2) POSITION UPDATE for what this invariant is for.
            # speed/distance are checked too here (unlike the (3b) candidate-configuration call
            # below) since agent.current_entry_point/agent.speed_counter are a consistent snapshot of
            # the same moment in time - candidate_entry_point there is the not-yet-committed *next*
            # position, so pairing it with the *pre-step* speed_counter would be comparing different
            # points in time (e.g. on map entry, candidate_entry_point is already set while
            # speed_counter is still None from before this step).
            current_entry_point = agent.current_entry_point
            next_entry_point = agent.next_entry_point
            self._check_off_on_map_invariant(current_entry_point, next_entry_point)
            assert agent.state is not None
            self._check_state_off_on_map_invariant(current_entry_point, agent.speed_counter.speed,
                                                   agent.speed_counter.distance, agent.state, agent.target_entry_point)

        return PreStepSnapshot(
            pre_speeds={agent.handle: agent.speed_counter.speed for agent in self.agents},
            pre_current_entry_points={agent.handle: agent.current_entry_point for agent in self.agents},
            pre_next_entry_points={agent.handle: agent.next_entry_point for agent in self.agents},
            pre_dones={agent.handle: self.dones[agent.handle] for agent in self.agents},
            pre_in_malfunctions={agent.handle: agent.malfunction_handler.in_malfunction for agent in self.agents},
            pre_offsets={agent.handle: agent.speed_counter.distance for agent in self.agents},
        )

    def _check_off_on_map_invariant(self, current_entry_point: EntryPointT, next_entry_point: EntryPointT):
        """
        Verify current_entry_point/next_entry_point are either both None (off map) or both set (on
        map, next_entry_point the successor of current_entry_point).
        """
        assert (current_entry_point is None) == (next_entry_point is None)
        if current_entry_point is not None:
            assert self.rail.is_successor(current_entry_point, next_entry_point)

    def _check_state_off_on_map_invariant(self, current_entry_point: EntryPointT, speed: Optional[Fraction],
                                          distance: Optional[Fraction], state: TrainState,
                                          target_entry_point: EntryPointT):
        """
        Verify speed/distance are None exactly when off map (the SpeedCounter contract - see "Speed/
        distance are Fractions, and None while off map" in CLAUDE.md), and that position matches the
        agent's state: DONE is neither an off-map nor an on-map TrainState, so a DONE agent (identified
        via target_entry_point, set exactly once and permanently on reaching target) is checked
        separately, branching on remove_agents_at_target (parked at target vs. removed); every other
        state must have current_entry_point is None exactly when state.is_off_map_state(). Only
        meaningful paired with a real (not candidate) configuration - a *candidate* configuration (not
        yet committed) paired with the agent's *pre-step* speed_counter/state would compare two
        different points in time, so this is called only from
        `_check_pre_step_invariants_and_capture_snapshot`, never from step()'s (3b) POSITION UPDATE
        (which calls only `_check_off_on_map_invariant`).
        """
        assert (current_entry_point is None) == (speed is None) == (distance is None)
        if target_entry_point is not None:
            if self.remove_agents_at_target:
                assert current_entry_point is None
            else:
                assert current_entry_point is not None
        else:
            assert (current_entry_point is None) == state.is_off_map_state()

    def _check_post_position_invariants(self, action_dict: Dict[int, RailEnvActions],
                                        pre_step: PreStepSnapshot) -> None:
        """
        Verify, for every agent, that this step's position update matches the expected transition given the
        pre-step snapshot captured.
        """
        for h in pre_step.pre_speeds.keys():
            agent = self.agents[h]
            # candidates discarded in distribute phase -> previous configuration
            if not self.temp_transition_data[h].resource_check:
                assert agent.current_entry_point == pre_step.pre_current_entry_points[h]
                assert agent.next_entry_point == pre_step.pre_next_entry_points[h]
            # candidates accepted in distribute phase
            else:
                action = RailEnvActions.from_value(action_dict.get(h, RailEnvActions.DO_NOTHING))
                pre_current_entry_point = pre_step.pre_current_entry_points[h]
                pre_next_entry_point = pre_step.pre_next_entry_points[h]
                pre_speed = pre_step.pre_speeds[h]
                is_on_map = pre_next_entry_point is not None
                is_cell_exit = pre_speed is not None and pre_speed > 0 and pre_step.pre_offsets[h] + pre_speed >= SEGMENT_LENGTH
                if is_on_map:
                    candidate_entry_point_independent = self.rail.apply_action_independent(action, pre_next_entry_point)
                else:
                    candidate_entry_point_independent = self.rail.apply_action_independent(action, agent.initial_entry_point)

                if not is_on_map and agent.current_entry_point is not None:
                    # (3b.3) map entry
                    candidate_entry_point = agent.initial_entry_point
                    candidate_next_entry_point = candidate_entry_point_independent
                elif is_on_map and is_cell_exit:
                    # (3b.5) on-map cell transition - candidate_next_entry_point stays None here exactly
                    # when the action was invalid at the boundary (denied by (3b.6)); guarded for below.
                    candidate_entry_point = pre_next_entry_point
                    candidate_next_entry_point = candidate_entry_point_independent
                else:
                    # (3b.1/3b.2/3b.2bis/3b.4/3b.6) unchanged
                    candidate_entry_point = pre_current_entry_point
                    candidate_next_entry_point = pre_next_entry_point

                # done
                if pre_step.pre_dones[h]:
                    if self.remove_agents_at_target:
                        assert agent.current_entry_point is None
                        assert agent.next_entry_point is None
                    else:
                        assert agent.current_entry_point is not None
                        assert agent.current_entry_point == pre_step.pre_current_entry_points[h]
                        assert agent.next_entry_point == pre_step.pre_next_entry_points[h]
                        assert agent.current_entry_point == agent.target_entry_point
                # in malfunction
                elif agent.malfunction_handler.in_malfunction:
                    assert agent.current_entry_point == pre_step.pre_current_entry_points[h]
                    assert agent.next_entry_point == pre_step.pre_next_entry_points[h]
                # map entry
                elif pre_step.pre_current_entry_points[h] is None and candidate_entry_point is not None:
                    assert agent.current_entry_point == agent.initial_entry_point
                    assert agent.current_entry_point == candidate_entry_point
                    assert agent.next_entry_point == candidate_next_entry_point
                # target reached
                elif candidate_entry_point in agent.targets and (
                    pre_step.pre_speeds[h] > 0 and pre_step.pre_offsets[h] + pre_step.pre_speeds[h] >= SEGMENT_LENGTH):
                    assert agent.target_entry_point == candidate_entry_point
                    if self.remove_agents_at_target:
                        assert agent.current_entry_point is None
                        assert agent.next_entry_point is None
                    else:
                        assert agent.current_entry_point is not None
                        assert agent.current_entry_point == candidate_entry_point
                        assert agent.next_entry_point == candidate_next_entry_point
                        assert agent.current_entry_point == agent.target_entry_point
                # on-map cell transition
                elif candidate_next_entry_point is not None and (
                    pre_step.pre_speeds[h] > 0 and pre_step.pre_offsets[h] + pre_step.pre_speeds[h] >= SEGMENT_LENGTH):
                    assert agent.current_entry_point is not None
                    assert agent.current_entry_point == candidate_entry_point
                    assert agent.next_entry_point == candidate_next_entry_point
                # stay
                else:
                    assert agent.current_entry_point == pre_step.pre_current_entry_points[h]
                    assert agent.next_entry_point == pre_step.pre_next_entry_points[h]

    def _check_post_speed_distance_invariants(self, action_dict: Dict[int, RailEnvActions],
                                              pre_step: PreStepSnapshot) -> None:
        """
        Verify, for every agent, that this step's speed update matches the expected transition given the
        pre-step snapshot.
        """
        # distance update invariant
        for h, pre_speed in pre_step.pre_speeds.items():
            agent = self.agents[h]

            # candidates discarded in distribute phase -> speed 0 and distance updated with pre-speed
            # (distance_without_crossing(None, None) == None covers the off-map map-entry-failed case
            # the same way as the on-map case, since pre_offsets[h] is None exactly when the agent was
            # off map pre-step)
            if not self.temp_transition_data[h].resource_check:
                assert agent.speed_counter.distance == SpeedCounter.distance_without_crossing(
                    pre_step.pre_offsets[h], pre_speed)

            # candidates accepted in distribute phase
            else:
                action_valid = self.temp_transition_data[h].state_transition_signal.action_valid
                # done
                if pre_step.pre_dones[h]:
                    assert agent.target_entry_point is not None
                    assert agent.target_entry_point in agent.targets
                    assert agent.speed_counter.distance == SpeedCounter.distance_without_crossing(
                        pre_step.pre_offsets[h], pre_speed)
                # target reached
                elif self.temp_transition_data[h].candidate_entry_point in agent.targets:
                    assert agent.target_entry_point is not None
                    assert agent.target_entry_point in agent.targets
                    if self.remove_agents_at_target:
                        assert agent.speed_counter.distance is None
                    else:
                        assert agent.speed_counter.distance == SpeedCounter.distance_without_crossing(
                            pre_step.pre_offsets[h], pre_speed)
                # off map (stayed off map) / map entry - genuinely post-hoc: whether map entry actually
                # happened isn't decided by resource_check/in_malfunction alone, it also needs the state
                # machine to have promoted to MOVING (e.g. a MALFUNCTION_OFF_MAP/READY_TO_DEPART agent
                # given STOP_MOVING can get an optimistic non-None candidate_entry_point from loop 1's
                # (3b.3) yet never be promoted - see issue #280) - so this can't be dedup'd against
                # self.temp_transition_data[h].candidate_entry_point the way the discarded/malfunction
                # branches above/below are.
                elif pre_step.pre_offsets[h] is None and agent.current_entry_point is None:
                    assert agent.speed_counter.distance is None
                elif pre_step.pre_offsets[h] is None and agent.current_entry_point is not None:
                    assert agent.speed_counter.distance == 0
                # malfunction
                elif agent.malfunction_handler.in_malfunction:
                    assert agent.speed_counter.distance == pre_step.pre_offsets[h]
                # invalid action
                elif not action_valid:
                    # design: an invalid action denies the crossing attempt at the cell boundary, same
                    # consequence as a resource_check denial (see the top-level "candidates discarded"
                    # branch above, and (10b)'s matching MOVING->STOPPED branch in step()) - distance
                    # banks up to the boundary, it just isn't credited with crossing it. pre_speed is
                    # always > 0 here (a MOVING agent's pre-step speed can never be 0).
                    assert agent.speed_counter.distance == SpeedCounter.distance_without_crossing(
                        pre_step.pre_offsets[h], pre_speed)
                # stopped, or stopped -> moving (or malfunction recovering)
                # (D1): pre-step speed 0 must never advance distance this step, whether the agent
                # stays STOPPED (a plain no-op) or is promoted to MOVING - (10a)/(10b)'s pre-speed-0
                # deferral in step() defers any crossing to the following (genuinely positive-pre-speed)
                # step.
                elif pre_speed == 0:
                    assert agent.speed_counter.distance == pre_step.pre_offsets[h]
                else:
                    assert agent.speed_counter.distance == SpeedCounter.distance_after_crossing(
                        pre_step.pre_offsets[h], pre_speed)

        # speed update invariant
        for h, pre_speed in pre_step.pre_speeds.items():
            agent = self.agents[h]
            action = RailEnvActions.from_value(action_dict.get(h, RailEnvActions.DO_NOTHING))

            # candidates discarded in distribute phase -> speed 0 and distance updated with pre-speed
            if not self.temp_transition_data[h].resource_check:
                if pre_step.pre_current_entry_points[h] is None:
                    # rejected map entry - agent stays off map, speed stays None
                    assert agent.speed_counter.speed is None
                else:
                    assert agent.speed_counter.speed == 0
            # candidates accepted in distribute phase
            else:
                action_valid = self.temp_transition_data[h].state_transition_signal.action_valid
                # done or target reached
                if pre_step.pre_dones[h] or self.temp_transition_data[h].candidate_entry_point in agent.targets:
                    assert agent.target_entry_point is not None
                    assert agent.target_entry_point in agent.targets
                    if self.remove_agents_at_target:
                        # handle_done_state() clears position and nulls speed_counter (speed/distance
                        # both None) on the exact step the agent reaches DONE and gets removed.
                        assert agent.speed_counter.speed is None
                    else:
                        assert agent.speed_counter.speed == Fraction(0)
                # malfunction
                elif agent.malfunction_handler.in_malfunction:
                    if pre_step.pre_current_entry_points[h] is None:
                        assert agent.speed_counter.speed is None
                    else:
                        assert agent.speed_counter.speed == 0
                # map entry - genuinely post-hoc, same reason as the distance-update loop's "map entry"
                # branch above: can't be dedup'd against self.temp_transition_data[h].candidate_entry_point,
                # since that candidate can be optimistically non-None (loop 1's (3b.3)) without the state
                # machine ever actually promoting to MOVING this step (see issue #280).
                elif pre_step.pre_current_entry_points[h] is None and agent.current_entry_point is not None:
                    assert agent.speed_counter.speed == _cap_speed(agent.speed_counter.max_speed,
                                                                    self.acceleration_delta)
                # stay off map
                elif pre_step.pre_current_entry_points[h] is None:
                    assert agent.speed_counter.speed is None
                    assert agent.speed_counter.distance is None
                # invalid action
                elif not action_valid:
                    assert agent.speed_counter.speed == 0
                # acceleration or start moving
                elif action == RailEnvActions.MOVE_FORWARD or (pre_speed == 0 and RailEnvActions.is_moving_action(action)):
                    # pre_speed is None while WAITING (not yet departing this step) -
                    # speed_after_acceleration(None, ...) == None matches that case too.
                    assert agent.speed_counter.speed == SpeedCounter.speed_after_acceleration(
                        pre_speed, agent.speed_counter.max_speed, self.acceleration_delta)
                # braking
                elif action == RailEnvActions.STOP_MOVING:
                    # pre_speed is None for WAITING/READY_TO_DEPART/MALFUNCTION_OFF_MAP -
                    # speed_after_braking(None, ...) == None matches that case too.
                    assert agent.speed_counter.speed == SpeedCounter.speed_after_braking(pre_speed, self.braking_delta)
                # default
                else:
                    assert agent.speed_counter.speed == pre_speed

    def _infrastructure_representation(self, entry_point: EntryPointT) -> str:
        raise NotImplementedError()

    def _get_observations(self):
        """
        Utility which returns the dictionary of observations for an agent with respect to environment
        """
        # print(f"_get_obs - num agents: {self.get_num_agents()} {list(range(self.get_num_agents()))}")
        self.obs_dict = self.obs_builder.get_many(list(range(self.get_num_agents())))
        return self.obs_dict

    def _call_rail_generator(self, optionals) -> Tuple[dict, TransitionMapT]:

        # TODO https://github.com/flatland-association/flatland-rl/issues/242 fix signature
        return self.rail_generator(self.number_of_agents, self.num_resets, self.np_random)

    def _apply_timetable_to_agents(self, agents, timetable: "Timetable") -> List[EnvAgent[EntryPointT]]:
        return EnvAgent.apply_timetable(agents, timetable)

    def _agents_from_line(self, line: "Line") -> List[EnvAgent[EntryPointT]]:
        raise NotImplementedError()


class RailEnv(AbstractRailEnv[GridTransitionMap, GridResourceMap, Tuple[Tuple[int, int], int]]):
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
                 check_step_pre_post_conditions: bool = True,
                 rewards: Rewards = None,
                 effects_generator: EffectsGenerator["RailEnv"] = None
                 ):
        """
        All parameters from parent `AbstractRailEnv`. Classic grid rail env, called rail env tout court for continuity.

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
            check_step_pre_post_conditions=check_step_pre_post_conditions,
            rewards=rewards,
            effects_generator=effects_generator,
            distance_map=DistanceMap([], height, width),
        )

        self.agent_positions = None

        # save timesteps in here: [[[row, col, dir, malfunction],...nAgents], ...nSteps]
        self.cur_episode = []
        if record_steps and find_effects_generator(self.effects_generator, RecordStepsEffectsGenerator) is None:
            # `make_multi_effects_generator` flattens automatically, so this never nests regardless of what the
            # caller passed as `effects_generator`; the `find_effects_generator` check above avoids adding a
            # second `RecordStepsEffectsGenerator` if the caller's `effects_generator` already contains one.
            self.effects_generator = make_multi_effects_generator(self.effects_generator, RecordStepsEffectsGenerator())

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
        if optionals and 'stations_links' in optionals:
            self.stations_links = optionals['stations_links']

        return optionals, rail

    def _extract_resource_map_from_optionals(self, optionals: dict) -> GridResourceMap:
        resource_map = GridResourceMap()
        if optionals and 'level_free_positions' in optionals:
            resource_map.level_free_positions = optionals['level_free_positions']
        return resource_map

    def _update_agent_positions_map(self, ignore_old_positions=True):
        """ Update the agent_positions array for agents that changed positions """
        for agent in self.agents:
            position = agent.current_entry_point[0] if agent.current_entry_point is not None else None
            old_position = agent.old_entry_point[0] if agent.old_entry_point is not None else None
            if not ignore_old_positions or old_position != position:
                if position is not None:
                    self.agent_positions[position] = agent.handle
                if old_position is not None:
                    self.agent_positions[old_position] = -1

    def clone_from(self, env: 'RailEnv', obs_builder: Optional[ObservationBuilder["RailEnv", Any]] = None):
        from flatland.envs.persistence import RailEnvPersister
        # avoid in-memory references
        env_dict = pickle.loads(pickle.dumps(RailEnvPersister.get_full_state(env)))
        RailEnvPersister.load(self, env_dict=env_dict, obs_builder=obs_builder)

    def step(self, action_dict: Dict[int, RailEnvActions]):
        obs, rewards, dones, info = super().step(action_dict=action_dict)
        # TODO https://github.com/flatland-association/flatland-rl/issues/195 add idiomatic wrapper instead of override
        self._update_agent_positions_map()
        return obs, rewards, dones, info

    def _infrastructure_representation(self, entry_point: Tuple[Tuple[int, int], int]) -> str:
        return RailEnvTransitionsEnum(self.rail.get_full_transitions(*entry_point[0])).name

    def _agents_from_line(self, line: "Line", rail: GridTransitionMap) -> List[EnvAgent[Tuple[Tuple[int, int], int]]]:
        agents = EnvAgent.from_line(line)
        for agent in agents:
            agent.targets = {t for t in agent.targets if rail.is_valid_entry_point(t)}
            # N.B. only the target's direction alternatives (last waypoint group) can be invalid - the
            # line generator's own routing already guarantees valid entry points everywhere else.
            agent.waypoints[-1] = _filter_valid_target_entry_points(rail, agent.waypoints[-1])
            assert len(agent.targets) > 0, (
                f"agent {agent.handle}: none of the target's direction alternatives are valid "
                f"entry points on the rail - the agent would end up with an empty `targets`."
            )
        return agents
