from collections import defaultdict
from typing import Generic, TypeVar, Tuple, Dict, Set, Optional, List, Iterator, NamedTuple, Callable

import numpy as np
from fastenum import fastenum

from flatland.core.env_observation_builder import AgentHandle
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.grid.distance_map import DistanceMap
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.env_utils import AgentTransitionData

RewardT = TypeVar('RewardT')
EntryPointT = TypeVar('EntryPointT')


class Rewards(Generic[RewardT]):
    """
    Rewards Function Interface.
    """

    def step_reward(self, agent: EnvAgent, agent_transition_data: AgentTransitionData, distance_map: DistanceMap, elapsed_steps: int) -> RewardT:
        """
        Handles end-of-step-reward for a particular agent.

        Parameters
        ----------
        agent: EnvAgent
        agent_transition_data: AgentTransitionData
        distance_map: DistanceMap
        elapsed_steps: int
        """
        raise NotImplementedError()

    def end_of_episode_reward(self, agent: EnvAgent, distance_map: DistanceMap, elapsed_steps: int) -> RewardT:
        """
        Handles end-of-episode reward for a particular agent.

        Parameters
        ----------
        agent: EnvAgent
        distance_map: DistanceMap
        elapsed_steps: int
        """
        raise NotImplementedError()

    def cumulate(self, *rewards: RewardT) -> RewardT:
        """
        Cumulate multiple rewards to one.

        Parameters
        ----------
        rewards

        Returns
        -------
        Cumulative rewards

        """
        raise NotImplementedError()

    def empty(self) -> RewardT:
        """
        Return empty initial value neutral for the cumulation.
        """
        raise NotImplementedError()

    def normalize(self, *rewards: RewardT, num_agents: int, max_episode_steps: int) -> Optional[float]:
        """
        Return normalized cumulated rewards. Can be `None` for some rewards.

        Parameters
        ----------
        rewards : List[RewardT]
        num_agents : int
        max_episode_steps : int

        Returns
        -------

        """
        return None

    # TODO https://github.com/flatland-association/flatland-rl/issues/527 we should drop these methods once EnvAgent.waypoints is also of EntryPointT instead of Waypoint.
    @staticmethod
    def _sanitize_waypoints(agent_waypoints: List[List[Waypoint]]) -> List[List[EntryPointT]]:
        agent_waypoints = [[(Rewards._sanitize_waypoint(wp)) for wp in wps] for wps in agent_waypoints]
        return agent_waypoints

    @staticmethod
    def _sanitize_waypoint(wp: Waypoint) -> EntryPointT:
        return wp._to_tuple() if isinstance(wp, Waypoint) else wp

    @staticmethod
    def _intermediate_waypoints(agent_waypoints: List[List[EntryPointT]],
                                agent: EnvAgent) -> Iterator[Tuple[List[EntryPointT], int, int]]:
        """
        Zips an agent's intermediate waypoint alternatives (i.e. excluding the initial and target stops)
        with their corresponding earliest-departure/latest-arrival time windows.
        """
        return zip(agent_waypoints[1:-1], agent.waypoints_latest_arrival[1:-1], agent.waypoints_earliest_departure[1:-1])


class PendingHalt(NamedTuple):
    """An agent's last halt at an entry point it has not left yet."""
    entry_point: object
    elapsed_steps: int


class DepartureClock:
    """
    Dates an agent's departure from an entry point: the step in which it starts moving after its last halt there.

    An agent that starts moving from a halt enters the next cell only once it has covered the rest of its cell, which
    takes up to ``1 / max_speed`` steps. Its departure is the step in which its speed becomes positive again, i.e. the
    step after its last halt there, not the step in which it enters the next cell. An agent that passes an entry point
    without halting departs in the step in which it enters the next cell.

    - Halt: the agent is on the map and has speed 0 after the step, for any reason: ``STOP_MOVING``, ``DO_NOTHING``
      while stopped, a denied crossing (invalid action or occupied cell), or a malfunction.
    - Last halt counts: an agent that halts, creeps forward within the cell and halts again departs in the step after
      its second halt. Creeping forward is not a departure.
    - Malfunction: an agent that halts at a stop, has a malfunction there (speed 0) and starts after the stop's
      earliest departure departs on time. Whether the stop counts as served is not decided here:
      :class:`BaseDefaultRewards` requires a halt without malfunction for that, :class:`PunctualityRewards` none.

    Each :class:`VisitLog` keeps its own clock. Per step and agent, call :meth:`departure` (only when a departure is
    booked) before :meth:`record`, so that a halt in the new cell does not date the departure from the old one.
    """

    def __init__(self):
        self._pending_halts: Dict[AgentHandle, PendingHalt] = {}

    def record(self, agent: EnvAgent, agent_transition_data: Optional[AgentTransitionData], elapsed_steps: int) -> None:
        """
        Keeps the agent's halt at its current entry point if it is on the map and halted in this step; a later halt at
        the same entry point replaces it. Without transition data (hand-built calls), the agent did not halt.
        """
        if agent_transition_data is None or agent.current_entry_point is None:
            return
        movement_allowed = agent_transition_data.action_valid and agent_transition_data.resource_check
        # candidate_speed is 0 for STOP_MOVING down to 0, for a stopped agent without a moving action and in a malfunction;
        # a denied crossing (not movement_allowed) ends with speed 0 as well.
        if agent_transition_data.candidate_speed == 0 or not movement_allowed:
            self._pending_halts[agent.handle] = PendingHalt(agent.current_entry_point, elapsed_steps)

    def departure(self, agent: EnvAgent, elapsed_steps: int) -> int:
        """
        The departure of the agent from its old entry point, which it left in this step: the step after its last halt
        there, or this step if it did not halt there.
        """
        halt = self._pending_halts.pop(agent.handle, None)
        if halt is not None and halt.entry_point == agent.old_entry_point:
            return halt.elapsed_steps + 1
        return elapsed_steps


class DefaultPenalties(fastenum.Enum):
    COLLISION = "COLLISION"
    INVALID_ACTION = "INVALID_ACTION"
    TARGET_LATE_ARRIVAL = "TARGET_LATE_ARRIVAL"
    CANCELLATION = "CANCELLATION"
    TARGET_NOT_REACHED = "TARGET_NOT_REACHED"
    TARGET_NOT_REACHED_MINIMUM_PENALTY = "TARGET_NOT_REACHED_MINIMUM_PENALTY"
    INTERMEDIATE_NOT_SERVED = "INTERMEDIATE_NOT_SERVED"
    INTERMEDIATE_LATE_ARRIVAL = "INTERMEDIATE_LATE_ARRIVAL"
    INTERMEDIATE_EARLY_DEPARTURE = "INTERMEDIATE_EARLY_DEPARTURE"


def _is_halt_serving_stop(agent: EnvAgent, agent_transition_data: Optional[AgentTransitionData]) -> bool:
    """
    Whether the agent halted at its current entry point in this step in a way that serves an intermediate stop there: it
    is on the map, not in a malfunction, and its speed is 0 after the step, whatever the action (``STOP_MOVING``,
    ``DO_NOTHING`` while stopped, a denied move). Standing there only during a malfunction does not serve the stop; a
    step with speed 0 after the malfunction ends does. The rule of :class:`BaseDefaultRewards` and
    :class:`PunctualityRewards` alike. Without transition data (hand-built calls), the agent did not halt.

    Not the stop that decides the collision and invalid-action penalties (``is_stopped_now`` in
    :meth:`BaseDefaultRewards.step_reward`): that one is the MOVING -> STOPPED transition only.
    """
    if agent_transition_data is None or agent.current_entry_point is None or agent.malfunction_handler.in_malfunction:
        return False
    # Compare against int 0, not float 0.0: Fraction.__eq__ takes a much slower path for a float operand (profiled at ~13x).
    movement_allowed = agent_transition_data.action_valid and agent_transition_data.resource_check
    return agent_transition_data.candidate_speed == 0 or not movement_allowed


class Visit(NamedTuple):
    """
    An agent's stay at one entry point, from entering it to leaving it. One entry point can have several visits.

    - ``arrival``: the step the agent entered the entry point (for the initial entry point, the step it entered the map).
    - ``departure``: the step it left, dated by :class:`DepartureClock`; ``None`` while it is still there.
    - ``halted``: whether it halted there in a way that serves a stop (:func:`_is_halt_serving_stop`).
    """
    entry_point: object
    arrival: int
    departure: Optional[int]
    halted: bool


class VisitLog:
    """
    Keeps each agent's visits of entry points in time order (:class:`Visit`), the bookkeeping that both
    :class:`BaseDefaultRewards` and :class:`PunctualityRewards` judge the timetable by. Each reward object keeps its own
    log; it is not persisted, as rewards are not.

    Call :meth:`record` once per step and agent. An agent removed from the map at its target stays at its target entry
    point.
    """

    def __init__(self):
        # https://stackoverflow.com/questions/16439301/cant-pickle-defaultdict
        self.visits: Dict[AgentHandle, List[Visit]] = defaultdict(list)
        # dates the departures: the step after the last halt at an entry point, see DepartureClock
        self._departure_clock = DepartureClock()

    def record(self, agent: EnvAgent, agent_transition_data: Optional[AgentTransitionData], elapsed_steps: int) -> None:
        visits = self.visits[agent.handle]
        # off the map after reaching the target (removed there), the agent is at its target entry point
        entry_point = agent.current_entry_point if agent.current_entry_point is not None else agent.target_entry_point
        if entry_point is not None and (len(visits) == 0 or visits[-1].entry_point != entry_point):
            # entered another entry point (or the map): leave the previous one, if any
            if len(visits) > 0 and visits[-1].departure is None:
                visits[-1] = visits[-1]._replace(departure=self._departure_clock.departure(agent, elapsed_steps))
            visits.append(Visit(entry_point, elapsed_steps, None, False))
        elif entry_point is None and agent.old_entry_point is not None and len(visits) > 0 and visits[-1].departure is None:
            # left the map
            visits[-1] = visits[-1]._replace(departure=self._departure_clock.departure(agent, elapsed_steps))
        if _is_halt_serving_stop(agent, agent_transition_data) and len(visits) > 0 and visits[-1].entry_point == agent.current_entry_point:
            visits[-1] = visits[-1]._replace(halted=True)
        # after the departure from the old entry point, so that a halt at the new one does not date it
        self._departure_clock.record(agent, agent_transition_data, elapsed_steps)

    def arrivals(self, handle: AgentHandle, entry_point) -> List[int]:
        """The agent's arrivals at the entry point, one per visit, in time order."""
        return [visit.arrival for visit in self.visits[handle] if visit.entry_point == entry_point]

    def departures(self, handle: AgentHandle, entry_point) -> List[int]:
        """The agent's departures from the entry point, one per visit it left, in time order."""
        return [visit.departure for visit in self.visits[handle] if visit.entry_point == entry_point and visit.departure is not None]

    def halted_entry_points(self, handle: AgentHandle) -> Set:
        """The entry points where the agent halted in a way that serves a stop, in any visit."""
        return {visit.entry_point for visit in self.visits[handle] if visit.halted}


def _match_stops(stops: List[Set], visits: List[Visit], cost: Callable[[int, Visit], Tuple], not_served_cost: Tuple) -> List[Optional[Visit]]:
    """
    Matches an agent's intermediate stops, in timetable order, to its visits, in time order: each stop gets at most one
    visit at one of its entry points, each visit serves at most one stop, and a later stop gets a later visit. Of all
    such matchings, the one with the least total cost counts; ties go to the earliest visits.

    Costs are tuples, summed element-wise and compared lexicographically: ``cost(stop_index, visit)`` for a stop served
    by a visit, ``not_served_cost`` for a stop without one.

    Returns, per stop, its visit or ``None``. Dynamic program over (stop, visit), O(stops x visits).
    """
    m, n = len(stops), len(visits)

    def add(a: Tuple, b: Tuple) -> Tuple:
        return tuple(x + y for x, y in zip(a, b))

    # best[j][i]: least cost of stops j.. with visits i..; choice[j][i]: what stop j does there
    best: List[List[Optional[Tuple]]] = [[None] * (n + 1) for _ in range(m + 1)]
    choice: List[List[Optional[str]]] = [[None] * (n + 1) for _ in range(m + 1)]
    best[m] = [tuple(0 for _ in not_served_cost)] * (n + 1)
    for j in range(m - 1, -1, -1):
        for i in range(n, -1, -1):
            # in order of preference on ties: serve stop j with visit i, skip visit i, leave stop j unserved
            options = []
            if i < n and visits[i].entry_point in stops[j]:
                options.append((add(cost(j, visits[i]), best[j + 1][i + 1]), "serve"))
            if i < n:
                options.append((best[j][i + 1], "skip"))
            options.append((add(not_served_cost, best[j + 1][i]), "not_served"))
            best[j][i], choice[j][i] = min(options, key=lambda option: option[0])

    matching: List[Optional[Visit]] = []
    i = 0
    for j in range(m):
        while choice[j][i] == "skip":
            i += 1
        if choice[j][i] == "serve":
            matching.append(visits[i])
            i += 1
        else:
            matching.append(None)
    return matching


class BaseDefaultRewards(Rewards[Dict[str, float]], Generic[EntryPointT]):
    r"""
    Reward Function.

    This scoring function is designed to capture key operational metrics such as punctuality, efficiency in responding to disruptions, and safety.

    Punctuality and schedule adherence are rewarded based on the difference between actual and target arrival and departure times at each stop respectively,
    as well as penalties for intermediate stops not served or even journeys not started.

    Safety measures are implemented as penalties for env-forced stops which are directly proportional to the train’s speed at impact, ensuring that high-speed operations are managed with extra caution.
    Tracked separately as :attr:`DefaultPenalties.COLLISION` (motion-check conflict with another agent)
    and :attr:`DefaultPenalties.INVALID_ACTION` (invalid action, today only possible at a symmetric
    switch) - both use the same ``collision_factor``, only the attributed category differs.

    Intermediate stops are served in timetable order, each by its own halt at one of its entry points: one halt serves at
    most one stop, and a later stop is served by a later halt. Of all such assignments of the agent's halts to its stops,
    the one that serves the most stops counts, and among those the one with the least penalty; a stop without a halt is
    not served (see :func:`_match_stops`). A halt at the initial entry point before the departure can serve the first
    stop. The env enforces only the initial entry point and the target: a target entry point is never an intermediate
    stop (checked when the agents are created), and an agent whose route passes its target before its last stops is
    done there, with those stops not served.

    Parameters
    ----------
    cancellation_factor : float
        Cancellation factor :math:`\phi \geq 0`. defaults to  1.
    cancellation_time_buffer : float
        Cancellation time buffer :math:`\pi \geq 0`. Defaults to 0.
    target_not_reached_minimum_penalty : float
        Target not reached minimum penalty :math:`\nu \geq 0`. Applied if agent is still on map at end of episode. Defaults to 0.
    intermediate_not_served_penalty : float
        Intermediate stop not served penalty :math:`\mu \geq 0`. Applied if one of the intermediates is not served or only run through without stopping. Defaults to 1.
    intermediate_late_arrival_penalty_factor : float
        Intermediate late arrival penalty factor :math:`\alpha \geq 0`. Defaults to 0.2.
    intermediate_early_departure_penalty_factor : float
        Intermediate early departure penalty factor :math:`\delta \geq 0`. Defaults to 0.5. The departure from a stop
        is the step in which the agent starts moving after its last halt there (speed 0 for any reason, a malfunction
        included), not the step in which it enters the next cell; see :class:`DepartureClock`.
    collision_factor : float
        Crash penalty factor :math:`\kappa \geq 0`. Shared by both the collision and invalid-action
        penalties (see class docstring). Defaults to 0.0.
    """
    # cache enumeration
    _cached_default_penalty_values = tuple(p.value for p in DefaultPenalties)
    # cache the empty reward dict itself, not just its keys - dict.fromkeys(keys, 0) rebuilds the dict
    # from the keys tuple every call; copying an already-built template is ~5x cheaper (profiled).
    _cached_empty_reward_dict = dict.fromkeys(_cached_default_penalty_values, 0)

    def __init__(self,
                 cancellation_factor: float = 1,
                 cancellation_time_buffer: float = 0,
                 target_not_reached_minimum_penalty: float = 0,
                 intermediate_not_served_penalty: float = 1,
                 intermediate_late_arrival_penalty_factor: float = 0.2,
                 intermediate_early_departure_penalty_factor: float = 0.5,
                 collision_factor: float = 0.0
                 ):
        self.collision_factor = collision_factor
        self.intermediate_early_departure_penalty_factor = intermediate_early_departure_penalty_factor
        self.intermediate_late_arrival_penalty_factor = intermediate_late_arrival_penalty_factor
        self.intermediate_not_served_penalty = intermediate_not_served_penalty
        self.cancellation_time_buffer = cancellation_time_buffer
        self.cancellation_factor = cancellation_factor
        self.target_not_reached_minimum_penalty = target_not_reached_minimum_penalty
        assert self.collision_factor >= 0
        assert self.intermediate_early_departure_penalty_factor >= 0
        assert self.intermediate_late_arrival_penalty_factor >= 0
        assert self.intermediate_not_served_penalty >= 0
        assert self.cancellation_time_buffer >= 0
        assert self.cancellation_factor >= 0
        assert self.target_not_reached_minimum_penalty >= 0
        # the agents' visits of entry points, with arrivals, departures and halts, see VisitLog
        self.visit_log = VisitLog()

    def step_reward(self, agent: EnvAgent, agent_transition_data: AgentTransitionData, distance_map: DistanceMap, elapsed_steps: int) -> Dict[str, float]:
        d = self.empty()
        # tests call step_reward() directly with agent_transition_data=None when a scenario doesn't
        # involve any per-step signal (e.g. a hand-constructed already-DONE/off-map agent, or a plain
        # dwelling/departure bookkeeping call) - treat as "normal, unimpeded movement, no special signal
        # this step" (action_valid=True, resource_check=True reproduce movement_allowed=True below,
        # matching the old code's implicit behavior of never actually dereferencing
        # agent_transition_data in those scenarios).
        agent_transition_data = agent_transition_data or AgentTransitionData(None, None, action_valid=True, resource_check=True)
        in_malfunction = agent.malfunction_handler.in_malfunction
        stop_action_given = agent_transition_data.action == RailEnvActions.STOP_MOVING
        # new_speed_zero mirrors rail_env.py's own _is_speed_zero(candidate_speed) - candidate_speed is
        # None for the "no signal" fallback above, and None == 0 is False, matching that fallback's intent.
        # Compare against int 0, not float 0.0: Fraction.__eq__ takes a much slower path for a float
        # operand (must losslessly convert it first) than for an int - profiled at ~13x the cost here.
        new_speed_zero = agent_transition_data.candidate_speed == 0
        # movement_allowed = action_valid and resource_check (rail_env.py's own formula - see
        # RailEnvStateMachineWrapper, which recomputes it identically for the state machine).
        movement_allowed = agent_transition_data.action_valid and agent_transition_data.resource_check
        is_on_map = agent.current_entry_point is not None
        # mirrors TrainStateMachine._handle_moving's own MOVING->STOPPED transition condition, using only
        # this step's signals - a state-machine-independent equivalent of agent.state == TrainState.STOPPED
        # (on-map, not malfunctioning, only MOVING/STOPPED remain).
        is_stopped_now = is_on_map and not in_malfunction and ((stop_action_given and new_speed_zero) or not movement_allowed)
        # agent_transition_data.speed is this step's pre-update speed (see rail_env.py's (8) FETCH
        # CONFLICT RESOLUTION) - a state-machine-independent equivalent of
        # agent.state_machine.previous_state == TrainState.MOVING.
        was_moving = agent_transition_data.speed is not None and agent_transition_data.speed > 0

        # arrivals, departures and halts that serve a stop (see _is_halt_serving_stop - not only the MOVING->STOPPED
        # transition above), per visit
        self.visit_log.record(agent, agent_transition_data, elapsed_steps)

        if was_moving and is_stopped_now:
            # A stop is "voluntary" if the controller issued STOP_MOVING and braking brings the speed to zero this step,
            # and the env did not itself deny movement (invalid action or motion check conflict, see
            # TrainStateMachine._handle_moving: MOVING -> STOPPED on `(stop_action_given and new_speed_zero) or not
            # movement_allowed`). Without the movement_allowed check, an env-forced stop that happens to coincide
            # with a STOP_MOVING action (e.g. STOP_MOVING evaluated as invalid upon facing a symmetric switch) would
            # be misclassified as voluntary and skip the penalty.
            # Only penalize stops imposed by the env (motion check conflict or invalid action), not controlled stops.
            voluntary_stop = stop_action_given and new_speed_zero and movement_allowed
            if not voluntary_stop:
                # agent_transition_data.speed has speed after action is applied at start of step(), not set to 0 upon motion check.
                # - if braking, reduced speed
                # - if not braking, still full speed
                # - if invalid action, speed set to 0
                # design (issue #280): yes, an invalid action is penalized - once per entering
                # attempt (a MOVING->STOPPED transition that denies the crossing). A STOPPED agent given
                # a non-moving retry of the same denied action (e.g. STOP_MOVING resolving to the same
                # invalid look-ahead) never promotes and so never re-charges; but a STOPPED agent given a
                # moving action IS optimistically re-promoted to MOVING (SpeedCounter.is_cell_exit()
                # requires speed > 0, see flatland-book's Design by Contract page,
                # https://flatland-association.github.io/flatland-book/key-concepts/design_by_contract.html, so a STOPPED agent never blocks its
                # own promotion), and if the resulting fresh attempt is denied again, that is itself a new
                # MOVING->STOPPED transition and is charged again.
                # Reaching here guarantees movement_allowed was False (see comment above), and
                # movement_allowed = action_valid and resource_check (rail_env.py) - so exactly one of
                # the two branches below applies. Same penalty value for both, tracked under different
                # keys so an invalid action (today, only possible at a symmetric switch) can be told
                # apart from an actual conflict with another agent.
                penalty = -1 * agent_transition_data.speed * self.collision_factor
                if not agent_transition_data.action_valid:
                    d[DefaultPenalties.INVALID_ACTION.value] = penalty
                elif not agent_transition_data.resource_check:
                    d[DefaultPenalties.COLLISION.value] = penalty

        # agent.arrival_time is set to exactly elapsed_steps by AbstractRailEnv.handle_done_state() the
        # one step an agent reaches its target - a state-machine-independent "just reached this step" check.
        if agent.arrival_time == elapsed_steps:
            self._agent_done_or_max_episode_steps_reward(agent, distance_map, elapsed_steps, d)
        return d

    def end_of_episode_reward(self, agent: EnvAgent, distance_map: DistanceMap, elapsed_steps: int) -> Dict[str, float]:
        d = self.empty()
        # If agent finished during episode, reward already calculated in step_reward() - target_entry_point
        # is set exactly once, permanently, the first step the agent reaches DONE (see
        # AbstractRailEnv.handle_done_state()), so it's a state-machine-independent done proxy.
        if agent.target_entry_point is not None:
            return d
        # Calculate penalty for not reaching target before episode end
        return self._agent_done_or_max_episode_steps_reward(agent, distance_map, elapsed_steps, d)

    def _agent_done_or_max_episode_steps_reward(self, agent, distance_map, elapsed_steps, d: Dict[str, float]):
        """
        Calculate final rewards/penalties for an agent.

        Called in two contexts:
        1. From step_reward(): when agent transitions to DONE during episode
        2. From end_of_episode_reward(): when episode ends and agent didn't finish

        Handles both completed and incomplete journeys.
        """
        if agent.target_entry_point is not None:
            # delay at target
            # if agent arrived earlier or on time = 0
            # if agent arrived later = -ve reward based on how late
            d[DefaultPenalties.TARGET_LATE_ARRIVAL.value] = min(agent.latest_arrival - agent.arrival_time, 0)
        else:
            if agent.current_entry_point is None:
                # journey not started
                d[DefaultPenalties.CANCELLATION.value] = -1 * self.cancellation_factor * \
                                                         (agent.get_travel_time_on_shortest_path(distance_map) + self.cancellation_time_buffer)

            # target not reached
            if agent.current_entry_point is not None:
                d[DefaultPenalties.TARGET_NOT_REACHED.value] = min(-1 * self.target_not_reached_minimum_penalty,
                                                                   agent.get_current_delay(elapsed_steps, distance_map))
        intermediates = list(self._intermediate_waypoints(self._sanitize_waypoints(agent.waypoints), agent))

        def late_and_early(stop_index: int, visit: Visit) -> Tuple[float, float]:
            _, la, ed = intermediates[stop_index]
            late = self.intermediate_late_arrival_penalty_factor * min(la - visit.arrival, 0)
            # a visit not left yet departs too early never
            early = self.intermediate_early_departure_penalty_factor * min(visit.departure - ed, 0) if visit.departure is not None else 0
            return late, early

        # serve as many stops as possible, then with the least penalty (rewards are negative penalties): a stop with a halt
        # pays its full late and early penalty, never capped at the not served penalty
        matching = _match_stops(
            stops=[set(alternatives) for alternatives, _, _ in intermediates],
            visits=[visit for visit in self.visit_log.visits[agent.handle] if visit.halted],
            cost=lambda stop_index, visit: (0, -sum(late_and_early(stop_index, visit))),
            not_served_cost=(1, self.intermediate_not_served_penalty),
        )
        for stop_index, visit in enumerate(matching):
            if visit is None:
                d[DefaultPenalties.INTERMEDIATE_NOT_SERVED.value] += -1 * self.intermediate_not_served_penalty
            else:
                late, early = late_and_early(stop_index, visit)
                d[DefaultPenalties.INTERMEDIATE_LATE_ARRIVAL.value] += late
                d[DefaultPenalties.INTERMEDIATE_EARLY_DEPARTURE.value] += early
        return d

    def cumulate(self, *rewards: Dict[str, float]) -> Dict[str, float]:
        result = self._cached_empty_reward_dict.copy()
        for r in rewards:
            for k, v in r.items():
                result[k] = result.get(k, 0) + v
        return result

    # policy runner calls normalization: normalize sum over all keys instead of per key.
    def normalize(self, *rewards: np.ndarray, num_agents: int, max_episode_steps: int) -> float:
        # https://flatland-association.github.io/flatland-book/challenges/ecml2026/eval.html
        if len(rewards) == num_agents:
            sum_per_agent = np.array(rewards)
        else:
            rewards_by_agent = np.reshape(np.array(rewards), (num_agents, -1), order='F')
            rewards_by_agent = [self.cumulate(*detailled_per_agent) for detailled_per_agent in rewards_by_agent]
            sum_per_agent = [sum(detailled_per_agent.values()) for detailled_per_agent in rewards_by_agent]
        rewards_capped = np.maximum(sum_per_agent, - max_episode_steps)
        return sum(rewards_capped) / (max_episode_steps * num_agents) + 1

    def empty(self) -> Dict[str, float]:
        return self._cached_empty_reward_dict.copy()


class DefaultRewards(Rewards[float]):
    """
    Aggregates `FineDefaultRewards` to single `float`.
    """

    def __init__(self,
                 cancellation_factor: float = 1.0,
                 cancellation_time_buffer: float = 0.0,
                 target_not_reached_minimum_penalty: float = 0.0,
                 intermediate_not_served_penalty: float = 1.0,
                 intermediate_late_arrival_penalty_factor: float = 0.2,
                 intermediate_early_departure_penalty_factor: float = 0.5,
                 collision_factor: float = 0.0
                 ):
        self._proxy = BaseDefaultRewards(
            cancellation_factor=cancellation_factor,
            cancellation_time_buffer=cancellation_time_buffer,
            target_not_reached_minimum_penalty=target_not_reached_minimum_penalty,
            intermediate_not_served_penalty=intermediate_not_served_penalty,
            intermediate_late_arrival_penalty_factor=intermediate_late_arrival_penalty_factor,
            intermediate_early_departure_penalty_factor=intermediate_early_departure_penalty_factor,
            collision_factor=collision_factor
        )

    @property
    def collision_factor(self):
        return self._proxy.collision_factor

    @property
    def intermediate_early_departure_penalty_factor(self):
        return self._proxy.intermediate_early_departure_penalty_factor

    @property
    def intermediate_late_arrival_penalty_factor(self):
        return self._proxy.intermediate_late_arrival_penalty_factor

    @property
    def intermediate_not_served_penalty(self):
        return self._proxy.intermediate_not_served_penalty

    @property
    def cancellation_time_buffer(self):
        return self._proxy.cancellation_time_buffer

    @property
    def cancellation_factor(self):
        return self._proxy.cancellation_factor

    @property
    def target_not_reached_minimum_penalty(self):
        return self._proxy.target_not_reached_minimum_penalty

    @collision_factor.setter
    def collision_factor(self, v):
        self._proxy.collision_factor = v

    @intermediate_early_departure_penalty_factor.setter
    def intermediate_early_departure_penalty_factor(self, v):
        self._proxy.intermediate_early_departure_penalty_factor = v

    @intermediate_late_arrival_penalty_factor.setter
    def intermediate_late_arrival_penalty_factor(self, v):
        self._proxy.intermediate_late_arrival_penalty_factor = v

    @intermediate_not_served_penalty.setter
    def intermediate_not_served_penalty(self, v):
        self._proxy.intermediate_not_served_penalty = v

    @cancellation_time_buffer.setter
    def cancellation_time_buffer(self, v):
        self._proxy.cancellation_time_buffer = v

    @cancellation_factor.setter
    def cancellation_factor(self, v):
        self._proxy.cancellation_factor = v

    @target_not_reached_minimum_penalty.setter
    def target_not_reached_minimum_penalty(self, v):
        self._proxy.target_not_reached_minimum_penalty = v

    def step_reward(self, agent: EnvAgent, agent_transition_data: AgentTransitionData, distance_map: DistanceMap, elapsed_steps: int) -> float:
        return sum(self._proxy.step_reward(agent, agent_transition_data, distance_map, elapsed_steps).values())

    def end_of_episode_reward(self, agent: EnvAgent, distance_map: DistanceMap, elapsed_steps: int) -> float:
        return sum(self._proxy.end_of_episode_reward(agent, distance_map, elapsed_steps).values())

    def cumulate(self, *rewards: float) -> float:
        return sum(rewards)

    def normalize(self, *rewards: np.ndarray, num_agents: int, max_episode_steps: int) -> float:
        # https://flatland-association.github.io/flatland-book/challenges/ecml2026/eval.html
        if len(rewards) == num_agents:
            sum_per_agent = np.array(rewards)
        else:
            rewards_by_agent = np.reshape(np.array(rewards), (num_agents, -1), order='F')
            sum_per_agent = np.sum(rewards_by_agent, axis=1)
        rewards_capped = np.maximum(sum_per_agent, - max_episode_steps)
        return sum(rewards_capped) / (max_episode_steps * num_agents) + 1

    def empty(self) -> float:
        return 0


class BaseECML2026Rewards(BaseDefaultRewards):
    """
    Parametrization of ECML 2026 Competition Rewards.
    """

    def __init__(self):
        super().__init__(
            cancellation_factor=5.0,
            cancellation_time_buffer=0.0,
            target_not_reached_minimum_penalty=100.0,
            intermediate_not_served_penalty=50.0,
            intermediate_late_arrival_penalty_factor=0.5,
            intermediate_early_departure_penalty_factor=0.5,
            collision_factor=250.0,
        )


class ECML2026Rewards(DefaultRewards):
    """
    Parametrization of ECML 2026 Competition Rewards (without details).
    """

    def __init__(self):
        base_rewards = BaseECML2026Rewards()
        super().__init__(
            cancellation_factor=base_rewards.cancellation_factor,
            cancellation_time_buffer=base_rewards.cancellation_time_buffer,
            target_not_reached_minimum_penalty=base_rewards.target_not_reached_minimum_penalty,
            intermediate_not_served_penalty=base_rewards.intermediate_not_served_penalty,
            intermediate_late_arrival_penalty_factor=base_rewards.intermediate_late_arrival_penalty_factor,
            intermediate_early_departure_penalty_factor=base_rewards.intermediate_early_departure_penalty_factor,
            collision_factor=base_rewards.collision_factor,
        )


class BasicMultiObjectiveRewards(DefaultRewards, Rewards[Tuple[float, float, float]]):
    """
    Basic MORL (Multi-Objective Reinforcement Learning) Rewards: with 3 items
     - default score
     - energy efficiency: - square of (speed/max_speed).
     - smoothness: - square of speed differences
    For illustration purposes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._previous_speeds = {}

    def step_reward(self, agent: EnvAgent, agent_transition_data: AgentTransitionData, distance_map: DistanceMap, elapsed_steps: int) -> Tuple[
        float, float, float]:
        default_reward = super().step_reward(agent=agent, agent_transition_data=agent_transition_data, distance_map=distance_map, elapsed_steps=elapsed_steps)

        # N.B. enforces penalization before/after malfunction, off-map and just-arrived (on-map, not
        # malfunctioning and not just done is exactly MOVING or STOPPED - a stopped agent's
        # speed_counter.speed is already 0, so this doesn't need to single out MOVING specifically).
        just_reached_target = agent.arrival_time == elapsed_steps
        current_speed = agent.speed_counter.speed if (
            agent.current_entry_point is not None and not agent.malfunction_handler.in_malfunction and not just_reached_target
        ) else 0

        energy_efficiency = -(current_speed / agent.speed_counter.max_speed) ** 2
        smoothness = 0
        if agent.handle in self._previous_speeds:
            smoothness = -(current_speed - self._previous_speeds[agent.handle]) ** 2
        self._previous_speeds[agent.handle] = current_speed
        return default_reward, float(energy_efficiency), float(smoothness)

    def end_of_episode_reward(self, agent: EnvAgent, distance_map: DistanceMap, elapsed_steps: int) -> Tuple[float, float, float]:
        default_reward = super().end_of_episode_reward(agent=agent, distance_map=distance_map, elapsed_steps=elapsed_steps)
        energy_efficency = 0
        smoothness = 0
        return default_reward, energy_efficency, smoothness

    def cumulate(self, *rewards: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return tuple([sum([r[i] for r in rewards]) for i in range(3)])

    def empty(self) -> Tuple[float, float, float]:
        return 0, 0, 0

    def normalize(self, *rewards: float, num_agents: int, max_episode_steps: int) -> float:
        return None


class PunctualityRewards(Rewards[Tuple[int, int]]):
    """
    Punctuality: n_stops_on_time / n_stops
    An agent is deemed not punctual at a stop if it arrives too late, departs too early or does not serve the stop at all. If an agent is punctual at a stop, n_stops_on_time is increased by 1.
    The departure from a stop is the step in which the agent starts moving after its last halt there (speed 0 for any reason, a malfunction included),
    not the step in which it enters the next cell; see :class:`DepartureClock`.

    The implementation returns the tuple `(n_stops_on_time, n_stops)`.

    An intermediate stop is served only if the agent halts there, as for :class:`BaseDefaultRewards`: speed 0 after a step
    at the stop, not in a malfunction (see :func:`_is_halt_serving_stop`). An agent that passes the stop without halting,
    or stands there only during a malfunction, does not serve it, also within its time window; one step with speed 0
    after the malfunction ends does.

    Intermediate stops are served in timetable order, each by its own halt at one of its entry points: one halt serves at
    most one stop, and a later stop is served by a later halt. Of all such assignments of the agent's halts to its stops,
    the one with the most stops on time counts; a stop without a halt is
    not served (see :func:`_match_stops`). A halt at the initial entry point before the departure can serve the first
    stop. The env enforces only the initial entry point and the target: a target entry point is never an intermediate
    stop (checked when the agents are created), and an agent whose route passes its target before its last stops is
    done there, with those stops not served.
    """

    def __init__(self):
        # the agents' visits of entry points, with arrivals, departures and halts, as BaseDefaultRewards.visit_log
        self.visit_log = VisitLog()

    def step_reward(self, agent: EnvAgent, agent_transition_data: AgentTransitionData, distance_map: DistanceMap, elapsed_steps: int) -> Tuple[int, int]:
        self.visit_log.record(agent, agent_transition_data, elapsed_steps)
        return 0, 0

    def end_of_episode_reward(self, agent: EnvAgent, distance_map: DistanceMap, elapsed_steps: int) -> Tuple[int, int]:
        n_stops_on_time = 0
        agent_waypoints = self._sanitize_waypoints(agent.waypoints)
        # by design, initial waypoint is unique
        assert len(agent_waypoints[0]) == 1

        visits = self.visit_log.visits[agent.handle]
        # the origin: the first visit, at the initial entry point, left at or after the earliest departure
        initial_wp = agent_waypoints[0][0]
        if len(visits) > 0 and visits[0].entry_point == initial_wp and visits[0].departure is not None \
                and visits[0].departure >= agent.waypoints_earliest_departure[0]:
            n_stops_on_time += 1

        intermediates = list(self._intermediate_waypoints(agent_waypoints, agent))

        def on_time(stop_index: int, visit: Visit) -> bool:
            _, la, ed = intermediates[stop_index]
            return visit.departure is not None and visit.arrival <= la and visit.departure >= ed

        # as many stops on time as possible; a stop served late counts as one not served
        matching = _match_stops(
            stops=[set(alternatives) for alternatives, _, _ in intermediates],
            visits=[visit for visit in visits if visit.halted],
            cost=lambda stop_index, visit: (0 if on_time(stop_index, visit) else 1,),
            not_served_cost=(1,),
        )
        n_stops_on_time += sum(1 for stop_index, visit in enumerate(matching) if visit is not None and on_time(stop_index, visit))

        # the target: entering it makes the agent done, so it has at most one visit there
        target_wps = set(agent_waypoints[-1])
        if any(visit.entry_point in target_wps and visit.arrival <= agent.waypoints_latest_arrival[-1] for visit in visits):
            n_stops_on_time += 1
        n_stops = len(agent.waypoints)
        return n_stops_on_time, n_stops

    def cumulate(self, *rewards: Tuple[int, int]) -> Tuple[int, int]:
        return sum([r[0] for r in rewards]), sum([r[1] for r in rewards])

    def empty(self) -> Tuple[int, int]:
        return 0, 0


class DelayRewards(DefaultRewards):
    """
    Delay: total delay time
    This is counted only for the agents' target, not the intermediate stops, to avoid duplicate counting.
    """

    def __init__(self):
        super().__init__(
            cancellation_factor=0.0,
            cancellation_time_buffer=0.0,
            target_not_reached_minimum_penalty=0.0,
            intermediate_not_served_penalty=0.0,
            intermediate_late_arrival_penalty_factor=0.0,
            intermediate_early_departure_penalty_factor=0.0,
            collision_factor=0.0,
        )
