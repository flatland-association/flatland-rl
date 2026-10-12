import math
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Set

from flatland.core.entry_point_distance_map import (
    EntryPointDistanceMap,
    EntryPointT,
    DistanceMapT,
    TransitionMapT,
    WaypointT,
)
from flatland.core.distance_map_walker import DistanceMapWalker
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.step_utils.states import TrainState


class AgentSourceTargetDistanceMap(
    EntryPointDistanceMap[TransitionMapT, DistanceMapT, EntryPointT,
    WaypointT]
):
    """
    Adds agent-handle (target_nr) aware querying on top of `EntryPointDistanceMap`. `geometric_distance`
    returns the minimum distance from a source entry point to any of a given agent's target entry points;
    concrete subclasses provide the underlying per-agent storage via `_set_geometric_distance`.
    """

    def __init__(self, agents: List[EnvAgent], waypoint_init: Callable[[EntryPointT], WaypointT]):
        super().__init__(agents=agents, waypoint_init=waypoint_init)
        self.distance_map = None
        self.agents_previous_computation = None
        self.reset_was_called = False

    def set(self, distance_map: DistanceMapT):
        """
        Set the distance map
        """
        self.distance_map = distance_map

    def get(self) -> DistanceMapT:
        """
        Get the distance map
        """
        if self.reset_was_called:
            self.reset_was_called = False

            compute_distance_map = True
            # Don't compute the distance map if it was loaded
            if self.agents_previous_computation is None and self.distance_map is not None:
                compute_distance_map = False

            if compute_distance_map:
                self._compute(self.agents, self.rail)

        elif self.distance_map is None:
            self._compute(self.agents, self.rail)

        return self.distance_map

    def reset(self, agents: List[EnvAgent], rail: TransitionMapT):
        """
        Reset the distance map
        """
        super().reset(agents=agents, rail=rail)
        self.reset_was_called = True

    def _compute(self, agents: List[EnvAgent], rail: TransitionMapT):
        """
        Computes the distance maps for each unique target. Thus, if several targets are the same we only
        compute the distance for them once and copy to all agents with the same target.
        """
        self.agents_previous_computation = self.agents
        self.distance_map = self._new_distance_map(len(agents))
        distance_map_walker = DistanceMapWalker(self)
        computed_targets = []
        for i, agent in enumerate(agents):
            targets = self._valid_targets(agent, rail)
            if targets not in computed_targets:
                reachable_entry_points = distance_map_walker._distance_map_walker(rail, targets)
                for entry_point in reachable_entry_points:
                    new_distance = min(
                        (self._get_distance(entry_point, target_entry_point) for target_entry_point in targets),
                        default=math.inf
                    )
                    self._set_geometric_distance(entry_point, i, new_distance)
            else:
                # just copy the distance map from other agent with same target (performance)
                self._copy_agent_distance(i, computed_targets.index(targets))
            computed_targets.append(targets)

    def _new_distance_map(self, num_agents: int) -> DistanceMapT:
        raise NotImplementedError()

    def _valid_targets(self, agent: EnvAgent, rail: TransitionMapT) -> List[EntryPointT]:
        raise NotImplementedError()

    def _copy_agent_distance(self, target_nr: int, source_target_nr: int):
        raise NotImplementedError()

    # N.B. get_shortest_paths is not part of distance_map since it refers to RailEnvActions (would lead to circularity!)
    def get_shortest_paths(self, max_depth: Optional[int] = None, agent_handle: Optional[int] = None) -> Dict[int, Optional[List[WaypointT]]]:
        """
        Computes the shortest path for each agent to its target and the action to be taken to do so.
        The paths are derived from a `DistanceMapT`.

        If there is no path (rail disconnected), the path is given as None.
        The agent state (moving or not) and its speed are not taken into account

        example:
                agent_fixed_travel_paths = get_shortest_paths(env.distance_map, None, agent.handle)
                path = agent_fixed_travel_paths[agent.handle]

        Parameters
        ----------
        self : reference to the distance_map
        max_depth : max path length, if the shortest path is longer, it will be cut
        agent_handle : if set, the shortest path for agent.handle will be returned, otherwise for all agents

        Returns
        -------
            Dict[int, Optional[List[WaypointT]]]

        """

        if agent_handle is not None:
            agents = [self.agents[agent_handle]]
        else:
            agents = self.agents

        shortest_paths = dict()
        for agent in agents:
            shortest_paths[agent.handle] = self._shortest_path_for_agent(agent, max_depth)

        return shortest_paths

    def _shortest_path_for_agent(self, agent: EnvAgent, max_depth: Optional[int] = None):
        if agent.derived_state().is_off_map_state():
            entry_point = agent.initial_entry_point
        elif agent.derived_state().is_on_map_state():
            entry_point = agent.current_entry_point
        elif agent.derived_state() == TrainState.DONE:
            return None
        else:
            return None
        handle = agent.handle
        targets = agent.targets

        return self._reconstruct_shortest_path(entry_point, handle, max_depth, targets)

    def _reconstruct_shortest_path(
        self,
        source: EntryPointT,
        handle,
        max_depth: Optional[int],
        targets: Set[EntryPointT]
    ) -> List[WaypointT]:
        """
        Reconstruct shortest path from distance map going forward from source to any of targets.
        """
        agent_shortest_path = []

        distance = math.inf
        depth = 0

        while source not in targets and (max_depth is None or depth < max_depth):
            best_next_entry_point = None
            next_entry_points = self.rail.get_successor_entry_points(source)
            for next_entry_point in next_entry_points:

                next_action_distance = self.geometric_distance(next_entry_point, handle)
                if next_action_distance < distance:
                    distance = next_action_distance
                    best_next_entry_point = next_entry_point
            agent_shortest_path.append(self.waypoint_init(source))
            depth += 1

            # if there is no way to continue, the rail must be disconnected!
            # (or distance map is incorrect)
            if best_next_entry_point is None:
                return None
            source = best_next_entry_point
        if max_depth is None or depth < max_depth:
            agent_shortest_path.append(self.waypoint_init(source))
        return agent_shortest_path

    def geometric_distance(self, source_entry_point: EntryPointT, target_nr: int):
        """
        The minimum number of cells to cross from `source_entry_point` to any of `target_nr`'s targets,
        following the rail - a geometric distance, not a number of steps (see `eta` for that).
        """
        self.get()
        return self._geometric_distance(source_entry_point, target_nr)

    def eta(
        self,
        entry_point: EntryPointT,
        handle: int,
        elapsed_steps: int,
        earliest_departure: int,
        speed: Optional[Fraction],
        max_speed: Fraction,
        acceleration_delta: Fraction,
        distance: Fraction = Fraction(0),
    ) -> Fraction:
        """
        Earliest time of arrival: the `elapsed_steps` value at which `handle` reaches a target, assuming
        it follows the shortest path from `entry_point` unhindered (no other agent, no switch chosen
        wrong) with a continuously-supplied moving action from here on.

        `speed` is the agent's current speed - `None` off map (not yet departed), in which case
        `earliest_departure` decides the departure step (`max(earliest_departure, 1)`, since there is no
        step 0 to depart in) and the ramp to `max_speed` starts from a standing start; `elapsed_steps` is
        then unused, since the predicted arrival step doesn't depend on when during the wait it's asked.
        A concrete `speed` (on map) instead ramps up from wherever it already is, ignoring
        `earliest_departure` (already irrelevant once departed). `distance` is how far the agent has
        already progressed into `entry_point`'s own cell (0 off map, the default) - subtracted from
        `geometric_distance(entry_point, handle)` before converting the remainder to time, since that
        distance is measured from the cell's entry, not the agent's actual position within it.

        Reaching max_speed from the current speed (or from rest, off map) takes `steps_to_max_speed =
        ceil((max_speed - speed) / acceleration_delta)` steps, covering `distance_during_acceleration =
        steps_to_max_speed * speed + acceleration_delta * steps_to_max_speed * (steps_to_max_speed - 1)
        / 2` - less than `steps_to_max_speed * max_speed`, since distance advances by the *pre-step*
        speed each step (see `SpeedCounter.set()`). The remaining geometric distance, after that ramp, is
        covered at max_speed.

        `steps_to_max_speed` is always at least 1, even off map: the departure step itself is the first
        of the ramp (its pre-step speed is 0, so - distance advancing by the pre-step speed - it
        contributes 0 to `distance_during_acceleration` even though its *post*-step speed already jumps
        to `min(acceleration_delta, max_speed)`). Since departure lands exactly on `elapsed_steps ==
        max(earliest_departure, 1)`, treating that value itself as the reference step and then adding
        `steps_to_max_speed` would double-count this step. `reference_step` is offset by `- 1` to the
        step *before* departure (speed conceptually still 0, zero ramp steps taken yet) so that adding
        `steps_to_max_speed` back lands exactly on the departure step, not one past it.
        """
        reference_step = elapsed_steps if speed is not None else max(earliest_departure, 1) - 1
        speed = speed if speed is not None else Fraction(0)
        steps_to_max_speed = math.ceil((max_speed - speed) / acceleration_delta) if speed < max_speed else 0
        distance_during_acceleration = (
            steps_to_max_speed * speed + acceleration_delta * steps_to_max_speed * (steps_to_max_speed - 1) / 2
        )
        # geometric_distance() returns a numpy float (always a whole cell count for a reachable entry
        # point, which this method assumes) - cast to an exact Fraction so it composes with
        # distance/speed/max_speed (also Fractions) without silent float rounding (1/3 isn't exactly
        # representable in float, and that error compounds across the arithmetic below).
        remaining_distance = Fraction(int(self.geometric_distance(entry_point, handle))) - distance - distance_during_acceleration
        return reference_step + steps_to_max_speed + remaining_distance / max_speed

    def _set_geometric_distance(self, source_entry_point: EntryPointT, target_nr: int, new_distance: int):
        raise NotImplementedError()

    def _geometric_distance(self, source_entry_point: EntryPointT, target_nr: int):
        raise NotImplementedError()
