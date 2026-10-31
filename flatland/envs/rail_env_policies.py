from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

from flatland.core.env_observation_builder import AgentHandle, ObservationBuilder
from flatland.core.policy import Policy
from flatland.core.transition_map import TransitionMap
from flatland.envs.agent_utils import EntryPointT
from flatland.envs.grid.grid_routing import greedy_grid_route_through_waypoints
from flatland.envs.rail_env import AbstractRailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState

# (rail, source, waypoint groups) -> (route starting at source, per group routed the index in the route of the alternative picked),
# e.g. `greedy_grid_route_through_waypoints()` or `greedy_graph_route_through_waypoints()`.
Router = Callable[[TransitionMap, EntryPointT, List[List[EntryPointT]]], Tuple[List[EntryPointT], List[int]]]


class AgentStaticData(NamedTuple):
    """ An agent's static data for an episode, as in `EnvAgent`: its initial entry point and its waypoint groups with their timetable. """
    initial_entry_point: EntryPointT
    waypoints: Tuple[Tuple[EntryPointT, ...], ...]
    waypoints_earliest_departure: Tuple[Optional[int], ...]
    waypoints_latest_arrival: Tuple[Optional[int], ...]


class RailEnvStaticData(NamedTuple):
    """
    A rail env's static data for an episode, immutable: the transition map (read-only by convention) and per agent
    handle its `AgentStaticData`. Not part of the observations: handed to the policies in the infos (key `"static"`),
    as one object shared by all agents.
    """
    rail: TransitionMap
    agents: Tuple[AgentStaticData, ...]

    @staticmethod
    def from_env(env: AbstractRailEnv) -> "RailEnvStaticData":
        return RailEnvStaticData(rail=env.rail, agents=tuple(
            AgentStaticData(initial_entry_point=agent.initial_entry_point,
                            waypoints=tuple(tuple(group) for group in agent.waypoints),
                            waypoints_earliest_departure=tuple(agent.waypoints_earliest_departure),
                            waypoints_latest_arrival=tuple(agent.waypoints_latest_arrival))
            for agent in env.agents))


@dataclass(frozen=True)
class ShortestPathObservation:
    """
    What `ShortestPathPolicy` observes of an agent: its dynamic state. Its static data (route, timetable) and the map
    come with the infos (`RailEnvStaticData`). After `reset()`, it has no position (`current_entry_point` is `None`).
    TODO move `elapsed_steps` to the infos, once the rail env hands them out.
    """
    handle: int
    elapsed_steps: int
    done: bool
    current_entry_point: Optional[EntryPointT]
    next_entry_point: Optional[EntryPointT]
    is_cell_exit: bool

    @staticmethod
    def from_env(env: AbstractRailEnv, handle: int) -> "ShortestPathObservation":
        agent = env.agents[handle]
        return ShortestPathObservation(handle=handle, elapsed_steps=env._elapsed_steps, done=agent.state == TrainState.DONE,
                                       current_entry_point=agent.current_entry_point, next_entry_point=agent.next_entry_point,
                                       is_cell_exit=agent.speed_counter.is_cell_exit())


class ShortestPathObservationBuilder(ObservationBuilder[AbstractRailEnv, ShortestPathObservation]):
    """ Builds the `ShortestPathObservation`s `ShortestPathPolicy` acts on. """

    def get(self, handle: AgentHandle = 0) -> ShortestPathObservation:
        return ShortestPathObservation.from_env(self.env, handle)


class ShortestPathPolicy(Policy[AbstractRailEnv, Union[ShortestPathObservation, AbstractRailEnv], RailEnvActions]):
    """
    Acts on `ShortestPathObservation`s (`ShortestPathObservationBuilder`), with the rail env's static data
    (`RailEnvStaticData`) in the infos passed to `act_many()` (`infos["static"]`). Backwards compatible: with
    `FullEnvObservation` (the env itself as observation), it derives both from the env.

    Each agent follows its route from its initial entry point, as computed once by the injected `router`, e.g.
    `greedy_grid_route_through_waypoints()` for `RailEnv` or `greedy_graph_route_through_waypoints()` for
    `GraphRailEnv`. Both are greedy, see there. With `router=None` (default), the grid router picks per waypoint group
    the first reachable alternative without `serve_intermediate_stops` (the previous behaviour), and the nearest one
    with it (a new mode, so nothing to stay compatible with).

    Without `serve_intermediate_stops` (default, the previous behaviour), the route goes straight to the target,
    ignoring intermediate stops and their earliest departures.

    With `serve_intermediate_stops`, the route goes through the intermediate stops, and each agent stops at each of
    them and respects its earliest departure there:

    - Stopping: `STOP_MOVING` in the step the agent crosses into the stop's entry point, so it halts there -
      Flatland's rewards count an intermediate stop as served only if the agent stopped there.
    - Departing: `STOP_MOVING` while `elapsed_steps + 1 < waypoints_earliest_departure[i]`, so the agent moves
      again in the step of its earliest departure; with no earliest departure, it stops for one step.
    - Moving on: as soon as `elapsed_steps + 1 >= waypoints_earliest_departure[i]` (with no earliest departure: in
      the step after halting), the policy returns the `MOVE_FORWARD`/`MOVE_LEFT`/`MOVE_RIGHT` towards the next entry
      point on its route again, as anywhere else on the route.

    Limitations:

    - No braking in advance: the agent only brakes in the step it crosses into the stop. It halts at the stop only
      if `braking_delta` stops it within the stop's cell (e.g. the default `braking_delta=-1`); with weaker braking it
      overshoots the stop, which then counts as not served.
    - Conflicts with other agents and malfunctions are not considered. An agent blocked by another one keeps its route
      and waits until the way is free: e.g. two agents heading towards each other on a single track deadlock, as
      neither gives way nor reroutes. A malfunction only delays the agent on its fixed route: e.g. it neither picks an
      alternative route nor skips a stop to catch up, so after a long malfunction it arrives late.
    """

    def __init__(self, router: Optional[Router] = None, serve_intermediate_stops: bool = False):
        super().__init__()
        if router is None:
            router = partial(greedy_grid_route_through_waypoints, nearest_alternative=serve_intermediate_stops)
        self.router = router
        self.serve_intermediate_stops = serve_intermediate_stops
        self._routes: Dict[int, List[EntryPointT]] = {}
        # per agent, route index of each routed waypoint group's alternative picked, starting with the initial entry point
        self._stop_indices: Dict[int, List[int]] = {}
        # per agent, route index of its current entry point - only moves forward, so a route visiting an
        # entry point twice is followed in order
        self._route_index: Dict[int, int] = {}

    def act_many(self, handles: List[int], observations: List[Union[ShortestPathObservation, AbstractRailEnv]],
                 infos: Optional[Dict] = None, **kwargs):
        actions = {}
        static = None
        for handle, observation in zip(handles, observations):
            if isinstance(observation, AbstractRailEnv):
                # `FullEnvObservation`: derive the observation and the static data from the env
                if static is None:
                    static = RailEnvStaticData.from_env(observation)
                observation = ShortestPathObservation.from_env(observation, handle)
            elif static is None:
                static = infos["static"]
            self._update_agent(observation, static)
            actions[handle] = self._act(observation, static)
        return actions

    def _update_agent(self, observation: ShortestPathObservation, static: RailEnvStaticData):
        """
        Update `_routes`, `_stop_indices` and `_route_index`.
        """
        handle = observation.handle
        if handle not in self._routes:
            agent = static.agents[handle]
            # TODO https://github.com/flatland-association/flatland-rl/issues/527 drop the conversion once EnvAgent.waypoints is also of EntryPointT
            #  instead of Waypoint.
            waypoint_groups = [[wp._to_tuple() if isinstance(wp, Waypoint) else wp for wp in group] for group in agent.waypoints[1:]]
            if not self.serve_intermediate_stops:
                waypoint_groups = waypoint_groups[-1:]
            route, stop_indices = self.router(static.rail, agent.initial_entry_point, waypoint_groups)
            self._routes[handle] = route
            self._stop_indices[handle] = [0] + stop_indices
            self._route_index[handle] = 0
        if observation.current_entry_point is None:
            return
        route = self._routes[handle]
        index = self._route_index[handle]
        while route[index] != observation.current_entry_point:
            index += 1
            assert index < len(route), f"Agent {handle} at {observation.current_entry_point} is off its route {route}."
        self._route_index[handle] = index

    def _act(self, observation: ShortestPathObservation, static: RailEnvStaticData) -> RailEnvActions:
        if observation.done:
            return RailEnvActions.DO_NOTHING
        if observation.current_entry_point is None:
            return RailEnvActions.MOVE_FORWARD

        handle = observation.handle
        route = self._routes[handle]
        index = self._route_index[handle]

        if self.serve_intermediate_stops:
            # intermediate stops only: neither the initial entry point nor the target
            intermediate_stops = {stop_index: group for group, stop_index in enumerate(self._stop_indices[handle][1:-1], start=1)}
            if index in intermediate_stops:
                earliest_departure = static.agents[handle].waypoints_earliest_departure[intermediate_stops[index]]
                # the action is applied in step `_elapsed_steps + 1` (`step()` increments `_elapsed_steps` first), so the
                # agent starts moving in the step of its earliest departure - as it enters the map at the initial stop
                if earliest_departure is not None and observation.elapsed_steps + 1 < earliest_departure:
                    return RailEnvActions.STOP_MOVING
            if index + 1 in intermediate_stops and observation.next_entry_point == route[index + 1] and observation.is_cell_exit:
                return RailEnvActions.STOP_MOVING

        # design: actions applied at cell entry -- once agent.next_entry_point holds a
        # pending target (decided one cell ago), this step's action no longer decides *that*
        # transition (already settled) but the look-ahead beyond it, so it must be evaluated from
        # next_entry_point against the route entry *after* it, not from current_entry_point against
        # the immediate next route entry.
        has_pending_target = observation.next_entry_point is not None and observation.next_entry_point != observation.current_entry_point
        lookahead_from = observation.next_entry_point if has_pending_target else observation.current_entry_point
        target_index = index + (2 if has_pending_target else 1)
        if target_index >= len(route):
            return RailEnvActions.MOVE_FORWARD

        for a in [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT]:
            if static.rail.apply_action_independent(a, lookahead_from) == route[target_index]:
                return a
        raise Exception("Invalid state")
