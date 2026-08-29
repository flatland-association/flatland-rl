import sys
import warnings
from typing import Tuple, NamedTuple, List, TypeVar, Generic, Optional, Set, Union

import numpy as np
from attr import attrs, attrib, Factory

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.transition_map import TransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.malfunction_handler import MalfunctionHandler
from flatland.envs.step_utils.speed_counter import SpeedCounter, _pseudo_fractional
from flatland.envs.step_utils.state_machine import TrainStateMachine
from flatland.envs.step_utils.states import TrainState
from flatland.envs.timetable_utils import Line
from flatland.envs.timetable_utils import Timetable


class Agent(NamedTuple):
    initial_position: Tuple[int, int]
    initial_direction: Grid4TransitionsEnum
    direction: Grid4TransitionsEnum
    # set of (position, direction) arrival alternatives, mirroring `EnvAgent.targets` - replaces the legacy
    # single `target` position (which implied "any direction"). Kept at the same tuple index as the old
    # `target` field so envs pickled before the switch still land their value in this slot (see
    # `_agent_tuple_targets`).
    targets: Set[Tuple[Tuple[int, int], Grid4TransitionsEnum]]
    moving: bool
    earliest_departure: int
    latest_arrival: int
    handle: int
    position: Tuple[int, int]
    arrival_time: int
    old_direction: Grid4TransitionsEnum
    old_position: Tuple[int, int]
    speed_counter: SpeedCounter
    # dead field, always None - kept only so envs pickled while `ActionSaver` still existed keep landing
    # their (now-discarded) value in this tuple slot instead of shifting every field after it out of
    # position (`Agent`/`EnvAgent` pickling is purely positional, see `load_env_agent`'s
    # `RailEnvPersister`-registered legacy unpickler for `ActionSaver` itself).
    action_saver: object
    state_machine: TrainStateMachine
    malfunction_handler: MalfunctionHandler
    waypoints: List[List[Waypoint]] = None
    waypoints_earliest_departure: List[int] = None
    waypoints_latest_arrival: List[int] = None
    # the specific target alternative actually reached - see `EnvAgent.target_entry_point`. `None` for
    # envs persisted before this field existed, or for an agent that hasn't reached DONE yet.
    target_position: Tuple[int, int] = None
    target_direction: Grid4TransitionsEnum = None
    # design: actions applied at cell entry.
    next_position: Tuple[int, int] = None
    next_direction: Grid4TransitionsEnum = None


def _normalize_waypoints(waypoints: List[Union[Waypoint, List[Waypoint]]]) -> List[List[Waypoint]]:
    """
    Normalizes a persisted `waypoints` field to the current `List[List[Waypoint]]` shape. Envs persisted
    before routing-flexibility alternatives were introduced store it as a flat `List[Waypoint]` (one bare
    `Waypoint` per stop) rather than one alternatives-group per stop - wrap any such bare entry in a
    single-element list. A no-op for waypoints already in the current shape.
    """
    return [wp if isinstance(wp, list) else [wp] for wp in waypoints]


def _filter_valid_target_entry_points(rail: TransitionMap, waypoint_group: List[Waypoint]) -> List[Waypoint]:
    """
    Keeps only the arrival alternatives in a target waypoint group that are valid entry points on `rail`.
    Envs persisted before routing-flexibility alternatives were introduced store a legacy `None`-direction
    placeholder (meaning "any direction") - explode that into one `Waypoint` per valid direction instead of
    filtering it out (since `None` itself is never a valid entry point). A no-op for waypoints already
    filtered to concrete, rail-valid directions.
    """
    if None in {wp.direction for wp in waypoint_group}:
        position = waypoint_group[0].position
        return [Waypoint(position, d) for d in Grid4TransitionsEnum if rail.is_valid_entry_point((position, d))]
    return [wp for wp in waypoint_group if rail.is_valid_entry_point((wp.position, wp.direction))]


def _agent_tuple_targets(agent_tuple: Agent) -> Set[Tuple[Tuple[int, int], Grid4TransitionsEnum]]:
    """
    Reads the target-entry-point set off a persisted `Agent`. `Agent.targets` replaced the legacy single
    `target` position (which implied "any direction"); envs pickled before that switch reconstruct positionally,
    landing a bare `(row, col)` position in the `targets` slot instead of a set - explode such a position to one
    entry point per direction. Concrete arrival directions are re-filtered against the rail on load anyway
    (see `set_full_state`), so exploding to all four here is safe.
    """
    targets = agent_tuple.targets
    if isinstance(targets, (set, frozenset)):
        return set(targets)
    # legacy: bare (row, col) position from a pre-`targets` pickle
    return {(targets, d) for d in Grid4TransitionsEnum}


def with_direction(entry_point: Optional[Tuple[Tuple[int, int], int]], direction: int) -> Tuple[Tuple[int, int], int]:
    """
    Returns a grid `(position, direction)` entry_point with `direction` replaced, preserving
    `entry_point`'s position - or `(None, direction)` if `entry_point` is `None` (e.g. the agent is
    currently off map).
    """
    return (entry_point[0] if entry_point is not None else None, direction)


def virtual_entry_point(agent: "EnvAgent") -> Optional[Tuple[Tuple[int, int], int]]:
    """
    Returns the effective grid `(position, direction)` for `agent`, regardless of whether it is
    currently on the map - used by observations/predictions that need an entry point to compute
    against even for off-map or arrived agents:
    - off map: `initial_entry_point`.
    - on map: `current_entry_point`.
    - done (arrived, possibly already removed from the map): `agent.target_entry_point` - the
      specific entry point actually reached, so a real, rail-valid direction is returned instead of
      the `None` `direction` that `current_entry_point` would give once the agent is removed from
      the map.
    - any other state (e.g. malfunctioning while still off map): `None`.
    """
    if agent.state.is_off_map_state():
        return agent.initial_entry_point
    elif agent.state.is_on_map_state():
        return agent.current_entry_point
    elif agent.state == TrainState.DONE:
        return agent.target_entry_point
    return None


def load_env_agent(agent_tuple: Agent, rail: TransitionMap):
    # Target entry points are serialised without rail validity (rail is not stored per-agent), so filter
    # them against `rail` here - previously a post-load step in `RailEnvPersister.set_full_state`. The target
    # waypoint group is filtered via `_filter_valid_target_entry_points` (which also explodes a legacy
    # `None`-direction placeholder), and `targets` is kept exactly in sync with it, per the invariant that
    # `EnvAgent.targets` is the last waypoint group.
    if agent_tuple.waypoints is not None:
        waypoints = _normalize_waypoints(agent_tuple.waypoints)
    else:
        waypoints = [
            [Waypoint(agent_tuple.initial_position, agent_tuple.initial_direction)],
            [Waypoint(position, direction) for position, direction in sorted(_agent_tuple_targets(agent_tuple))]]
    waypoints[-1] = _filter_valid_target_entry_points(rail, waypoints[-1])
    targets = {(wp.position, wp.direction) for wp in waypoints[-1]}

    current_entry_point = (agent_tuple.position, agent_tuple.direction) if agent_tuple.position is not None and agent_tuple.direction is not None else None
    next_entry_point = (
        agent_tuple.next_position, agent_tuple.next_direction
    ) if agent_tuple.next_position is not None and agent_tuple.next_direction is not None else None
    # design: actions applied at cell entry.
    assert (current_entry_point is None and next_entry_point is None) or (
        current_entry_point is not None and next_entry_point is not None and next_entry_point != current_entry_point
    ), (
        f"current_entry_point/next_entry_point invariant violated on load for agent {agent_tuple.handle}: "
        f"current_entry_point={current_entry_point}, next_entry_point={next_entry_point}. Only an env at "
        f"step 0 (pre-departure) or with this agent fully done and removed can be loaded; a pickle predating "
        f"`next_entry_point` resumed mid-run is not supported."
    )

    return EnvAgent(
        initial_entry_point=(agent_tuple.initial_position, agent_tuple.initial_direction),
        current_entry_point=current_entry_point,
        old_entry_point=(
            agent_tuple.old_position, agent_tuple.old_direction) if agent_tuple.old_position is not None and agent_tuple.old_direction is not None else None,
        target_entry_point=(
            agent_tuple.target_position, agent_tuple.target_direction
        ) if agent_tuple.target_position is not None and agent_tuple.target_direction is not None else None,
        next_entry_point=next_entry_point,
        targets=targets,
        moving=agent_tuple.moving,
        earliest_departure=agent_tuple.earliest_departure,
        latest_arrival=agent_tuple.latest_arrival,
        handle=agent_tuple.handle,
        arrival_time=agent_tuple.arrival_time,
        speed_counter=agent_tuple.speed_counter,
        state_machine=agent_tuple.state_machine,
        malfunction_handler=agent_tuple.malfunction_handler,
        waypoints=waypoints,
        waypoints_earliest_departure=agent_tuple.waypoints_earliest_departure if agent_tuple.waypoints_earliest_departure is not None else [
            agent_tuple.earliest_departure, None],
        waypoints_latest_arrival=agent_tuple.waypoints_latest_arrival if agent_tuple.waypoints_latest_arrival is not None else [None,
                                                                                                                                agent_tuple.latest_arrival],
    )


EntryPointT = TypeVar('EntryPointT')


def _sanitize_entry_point(entry_point):
    """Coerce a grid (position, direction) entry_point's numeric elements to plain int.

    Rail/line generation code (and various cached grid-transition helpers) can hand back numpy scalars
    (e.g. np.int64) instead of plain int for a cell position/direction. Left unsanitized, that numpy-ness
    gets stored into agent.initial_entry_point/current_entry_point and can later make a position tuple
    compare unequal-via-array-broadcast instead of a clean False against a differently-shaped tuple
    elsewhere (e.g. agent_chains.py's level-free-crossing resources), raising "The truth value of an array
    with more than one element is ambiguous". attrs converters only run in __init__, not on later
    attribute assignment, so this has to be called explicitly at every write site instead.

    A no-op for graph entry points (or anything else that doesn't look like a grid
    ((row, col), direction) tuple), since EntryPointT is generic across grid and graph envs.
    """
    if entry_point is None:
        return None
    try:
        position, direction = entry_point
    except (TypeError, ValueError):
        return entry_point
    if not isinstance(position, tuple) or len(position) != 2:
        return entry_point
    position = tuple(int(c) if isinstance(c, (np.generic, np.ndarray)) else c for c in position)
    if isinstance(direction, (np.generic, np.ndarray)):
        direction = int(direction)
    return (position, direction)


@attrs
class EnvAgent(Generic[EntryPointT]):
    # INIT FROM HERE IN _from_line()
    # converter=_sanitize_entry_point: covers construction (e.g. from rail/line generation code, which is
    # exactly where the numpy-dtype taint described on _sanitize_entry_point has been observed entering) -
    # attrs converters only run in __init__, so later direct assignments (agent.initial_entry_point = ...,
    # agent.current_entry_point = ..., agent.old_entry_point = ..., agent.target_entry_point = ...)
    # still need to call _sanitize_entry_point explicitly themselves, unless the assigned value is already
    # known-sanitized (e.g. copied from another already-sanitized entry point attrib on the same agent).
    initial_entry_point = attrib(type=EntryPointT, converter=_sanitize_entry_point)

    current_entry_point = attrib(type=Optional[EntryPointT], default=Factory(lambda: None),
                                 converter=_sanitize_entry_point)
    targets = attrib(type=Set[EntryPointT], default=Factory(lambda: set()))
    # the specific entry point (a member of `targets`) the agent actually arrived at, once
    # `state == TrainState.DONE` - set exactly once, by `AbstractRailEnv.handle_done_state()`, before
    # `current_entry_point` is possibly cleared to `None` (`remove_agents_at_target`). `None` until
    # the agent reaches DONE. Unlike `next(iter(targets))`, this is deterministic: `targets` may hold
    # several direction alternatives at the same position, only one of which was actually reached.
    target_entry_point = attrib(type=Optional[EntryPointT], default=Factory(lambda: None),
                                converter=_sanitize_entry_point)

    moving = attrib(default=False, type=bool)

    # NEW : EnvAgent - Schedule properties
    earliest_departure = attrib(default=0, type=int)
    latest_arrival = attrib(default=sys.maxsize, type=int)

    # including initial and target, routing flexibility
    waypoints = attrib(type=List[List[Waypoint]], default=Factory(lambda: [[]]))
    # None at target, same for all in routing flexibility
    waypoints_earliest_departure = attrib(type=List[int], default=Factory(lambda: []))
    # None at initial, same for all in routing flexibility
    waypoints_latest_arrival = attrib(type=List[int], default=Factory(lambda: []))

    handle = attrib(default=None)
    # INIT TILL HERE IN _from_line()

    # Env step facelift
    speed_counter = attrib(default=Factory(lambda: SpeedCounter(max_speed=1.0)), type=SpeedCounter)
    state_machine = attrib(default=Factory(lambda: TrainStateMachine(initial_state=TrainState.WAITING)),
                           type=TrainStateMachine)
    malfunction_handler = attrib(default=Factory(lambda: MalfunctionHandler()), type=MalfunctionHandler)

    # NEW : EnvAgent Reward Handling
    arrival_time = attrib(default=None, type=int)

    old_entry_point = attrib(type=Optional[EntryPointT], default=Factory(lambda: None),
                             converter=_sanitize_entry_point)

    # design: actions applied at cell entry.
    next_entry_point = attrib(type=Optional[EntryPointT], default=Factory(lambda: None),
                              converter=_sanitize_entry_point)

    def reset(self):
        """
        Resets the agents to their initial values of the episode. Called after ScheduleTime generation.
        """
        self.current_entry_point = None
        self.old_entry_point = None
        self.target_entry_point = None
        self.next_entry_point = None
        self.moving = False
        self.arrival_time = None

        self.malfunction_handler.reset()

        self.speed_counter.reset()
        self.state_machine.reset()

    def to_agent(self) -> Agent:
        return Agent(initial_position=self.initial_entry_point[0],
                     initial_direction=self.initial_entry_point[1],
                     direction=self.current_entry_point[1] if self.current_entry_point is not None else None,
                     # N.B. the full arrival-entry-point set is serialized, but re-filtered against the rail
                     # on load (see `set_full_state`), since rail validity is not stored with the agent.
                     targets=set(self.targets),
                     moving=self.moving,
                     earliest_departure=self.earliest_departure,
                     latest_arrival=self.latest_arrival,
                     handle=self.handle,
                     position=self.current_entry_point[0] if self.current_entry_point is not None else None,
                     old_direction=self.old_entry_point[1] if self.old_entry_point is not None else None,
                     old_position=self.old_entry_point[0] if self.old_entry_point is not None else None,
                     speed_counter=self.speed_counter,
                     action_saver=None,
                     arrival_time=self.arrival_time,
                     state_machine=self.state_machine,
                     malfunction_handler=self.malfunction_handler,
                     waypoints=self.waypoints,
                     waypoints_earliest_departure=self.waypoints_earliest_departure,
                     waypoints_latest_arrival=self.waypoints_latest_arrival,
                     target_position=self.target_entry_point[0] if self.target_entry_point is not None else None,
                     target_direction=self.target_entry_point[1] if self.target_entry_point is not None else None,
                     next_position=self.next_entry_point[0] if self.next_entry_point is not None else None,
                     next_direction=self.next_entry_point[1] if self.next_entry_point is not None else None,
                     )

    def get_shortest_path(self, distance_map) -> List[Waypoint]:
        return distance_map.get_shortest_paths(agent_handle=self.handle)[self.handle]

    def get_travel_time_on_shortest_path(self, distance_map) -> int:
        shortest_path = self.get_shortest_path(distance_map)
        if shortest_path is not None:
            distance = len(shortest_path)
        else:
            distance = 0
        speed = self.speed_counter.max_speed
        return int(np.ceil(distance / speed))

    def get_time_remaining_until_latest_arrival(self, elapsed_steps: int) -> int:
        return self.latest_arrival - elapsed_steps

    def get_current_delay(self, elapsed_steps: int, distance_map) -> int:
        '''
        +ve if arrival time is projected before latest arrival
        -ve if arrival time is projected after latest arrival
        '''
        return self.get_time_remaining_until_latest_arrival(elapsed_steps) - \
            self.get_travel_time_on_shortest_path(distance_map)

    @classmethod
    def from_line(cls, line: Line):
        """ Create a list of EnvAgent from lists of positions, directions and targets
        """
        num_agents = len(line.agent_waypoints)

        agent_list = []
        for i_agent in range(num_agents):
            speed = line.agent_speeds[i_agent] if line.agent_speeds is not None else 1.0

            agent = EnvAgent(
                initial_entry_point=(line.agent_waypoints[i_agent][0][0].position, line.agent_waypoints[i_agent][0][0].direction),
                # why
                current_entry_point=(line.agent_waypoints[i_agent][0][0].position, line.agent_waypoints[i_agent][0][0].direction),
                old_entry_point=None,
                targets={(line.agent_waypoints[i_agent][-1][0].position, d) for d in Grid4TransitionsEnum},
                waypoints=line.agent_waypoints[i_agent],
                moving=False,
                earliest_departure=None,
                latest_arrival=None,
                waypoints_earliest_departure=None,
                waypoints_latest_arrival=None,
                handle=i_agent,
                speed_counter=SpeedCounter(max_speed=speed))
            agent_list.append(agent)

        return agent_list

    @staticmethod
    def to_line(agents: List["EnvAgent"]):
        return Line(
            agent_waypoints={agent.handle: agent.waypoints for agent in agents},
            agent_speeds={agent.handle: agent.speed_counter.max_speed for agent in agents},
        )

    @classmethod
    def load_legacy_static_agent(cls, static_agents_data: Tuple, rail: TransitionMap = None):
        agents = []
        for i, static_agent in enumerate(static_agents_data):
            initial_entry_point = (static_agent[0], static_agent[1])
            targets = {(static_agent[2], d) for d in Grid4TransitionsEnum}
            if len(static_agent) >= 6:
                speed = static_agent[4]['speed']
                speed = _pseudo_fractional(speed)

                agent = EnvAgent(
                    initial_entry_point=initial_entry_point,
                    current_entry_point=initial_entry_point,
                    old_entry_point=None,
                    # N.B. valid targets cleaned in _agents_from_line
                    targets=targets,
                    moving=static_agent[3],
                    speed_counter=SpeedCounter(max_speed=speed), handle=i,
                    waypoints=[[Waypoint(*initial_entry_point)], [Waypoint(*target) for target in targets]],
                    earliest_departure=0,
                    waypoints_earliest_departure=[0, None],
                    latest_arrival=sys.maxsize,
                    waypoints_latest_arrival=[None, sys.maxsize],
                )
            else:
                agent = EnvAgent(
                    initial_entry_point=initial_entry_point,
                    current_entry_point=initial_entry_point,
                    old_entry_point=None,
                    # N.B. valid targets cleaned in _agents_from_line
                    targets={(static_agent[2], d) for d in Grid4TransitionsEnum},
                    moving=False,
                    speed_counter=SpeedCounter(max_speed=1.0),
                    handle=i,
                    waypoints=[[Waypoint(*initial_entry_point)], [Waypoint(*target) for target in targets]],
                    earliest_departure=0,
                    waypoints_earliest_departure=[0, None],
                    latest_arrival=sys.maxsize,
                    waypoints_latest_arrival=[None, sys.maxsize],
                )
            # Targets are exploded to all four directions above; filter to the rail-valid ones (when a rail
            # is available), keeping `targets` and the target waypoint group in sync. Callers without a rail
            # (deprecated msgpack loaders) leave this to a later step.
            if rail is not None:
                agent.waypoints[-1] = _filter_valid_target_entry_points(rail, agent.waypoints[-1])
                agent.targets = {(wp.position, wp.direction) for wp in agent.waypoints[-1]}
            agents.append(agent)
        return agents

    def __str__(self):
        direction = self.current_entry_point[1] if self.current_entry_point is not None else None
        return (
            f"EnvAgent(\n"
            f"\thandle={self.handle},\n"
            f"\tinitial_position={self.initial_entry_point[0]},\n"
            f"\tinitial_direction={self.initial_entry_point[1]},\n"
            f"\tposition={self.current_entry_point[0] if self.current_entry_point is not None else None},\n"
            f"\tdirection={direction if direction is None else Grid4TransitionsEnum(direction).value},\n"
            f"\ttargets={self.targets},\n"
            f"\told_position={self.old_entry_point[0] if self.old_entry_point is not None else None},\n"
            f"\told_direction={self.old_entry_point[1] if self.old_entry_point is not None else None},\n"
            f"\ttarget_entry_point={self.target_entry_point},\n"
            f"\tearliest_departure={self.earliest_departure},\n"
            f"\tlatest_arrival={self.latest_arrival},\n"
            f"\tstate_machine={str(self.state_machine)},\n"
            f"\tmalfunction_handler={self.malfunction_handler},\n"
            f"\twaypoints={self.waypoints},\n"
            f"\twaypoints_earliest_departure={self.waypoints_earliest_departure},\n"
            f"\twaypoints_latest_arrival={self.waypoints_latest_arrival},\n"
            f")"
        )

    @property
    def state(self):
        return self.state_machine.state

    @state.setter
    def state(self, state):
        self._set_state(state)

    def _set_state(self, state):
        warnings.warn("Not recommended to set the state with this function unless completely required")
        self.state_machine.set_state(state)

    @property
    def malfunction_data(self):
        raise ValueError("agent.malunction_data is deprecated, please use agent.malfunction_hander instead")

    @property
    def speed_data(self):
        raise ValueError("agent.speed_data is deprecated, please use agent.speed_counter instead")

    @classmethod
    def apply_timetable(cls, agents: List["EnvAgent"], timetable: Timetable) -> List["EnvAgent"]:
        for agent_i, agent in enumerate(agents):
            # TODO https://github.com/flatland-association/flatland-rl/issues/280 revise design: should we set state to READY_TO_DEPART if earliest_departure == 0? See `test_known_flatland_bugs.test_earliest_departure_zero_bug`.
            agent.earliest_departure = timetable.earliest_departures[agent_i][0]
            agent.latest_arrival = timetable.latest_arrivals[agent_i][-1]
            agent.waypoints_earliest_departure = timetable.earliest_departures[agent_i]
            agent.waypoints_latest_arrival = timetable.latest_arrivals[agent_i]
        return agents
