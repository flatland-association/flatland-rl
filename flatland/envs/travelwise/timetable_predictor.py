import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.envs.agent_utils import EnvAgent, EntryPointT
from flatland.envs.graph.graph_routing import greedy_graph_route_through_waypoints
from flatland.envs.graph_rail_env import GraphRailEnv
from flatland.envs.rail_env_policies import Router
from flatland.envs.step_utils.states import TrainState


@dataclass(frozen=True)
class ProjectedEntryPoint:
    """
    An entry point on a vehicle's projected route, the step the vehicle is projected to enter it and, if the vehicle
    stops there, the index of the waypoint group (`EnvAgent.waypoints`) it serves.
    """
    entry_point: EntryPointT
    step: int
    stop: Optional[int] = None


class TimetablePredictor(PredictionBuilder[GraphRailEnv, Dict[int, List[ProjectedEntryPoint]]]):
    """
    Predicts each vehicle's remaining route along its waypoints and timetable: per vehicle handle, its entry points in
    route order, each with the step it is projected to enter it and the waypoint group it stops at, if any.
    """

    def get(self, handle: int = None) -> Dict[int, List[ProjectedEntryPoint]]:
        """
        Call it after each env step: implementations may track the vehicles' progress along their waypoints in it.

        Parameters
        ----------
        handle : int, optional
            the vehicle to predict - all vehicles if `None`

        Returns
        -------
        Dict[int, List[ProjectedEntryPoint]]
            per vehicle handle, its remaining route in order - empty once it has arrived
        """
        raise NotImplementedError()


class UnilateralTimetablePredictor(TimetablePredictor):
    """
    Predicts each vehicle's remaining route along its waypoints and timetable unilaterally, as if no other agent were
    in the environment (as `ShortestPathPredictorForRailEnv`), and it runs on time: the vehicle stops at each intermediate
    stop until its earliest departure there, as `ShortestPathPolicy` with `serve_intermediate_stops` drives it.

    The prediction of a vehicle is its route in order, each entry point with the step it is entered:

    - on the map, starting at its current entry point at the current step;
    - not yet on the map, starting at its initial entry point at `max(earliest departure, elapsed steps + 1)`;
    - empty once it has arrived.

    The route is routed at every `get()` by the injected `router` (default: `greedy_graph_route_through_waypoints()`),
    from the vehicle's current (or initial) entry point through the waypoint groups it has not reached yet. If a group
    cannot be reached, the prediction ends with the last group routed.

    Times:

    - Moving: the vehicle enters the next entry point after `ceil((1 - distance) / max_speed)` steps, `distance` being
      the distance already travelled in its current cell (0 on entering a cell).
    - Stopping: at an intermediate stop, the vehicle halts in the step it enters it, and moves again (with distance
      0) in the step `max(arrival + 1, earliest departure)` - which covers no distance yet. A vehicle halted on the map
      now moves again in the step `elapsed steps + 1`, or at its stop's earliest departure if later.
    - No wait at the target: the vehicle leaves the map in the step it enters it.

    Optimistic: conflicts with other agents and malfunctions are ignored, and the vehicle is assumed to run at
    `max_speed` at once (`acceleration_delta` covering `max_speed`, as the default does). So the prediction is exact
    for a vehicle alone on the map, and else may be too early, never too late.
    """

    def __init__(self, router: Router = greedy_graph_route_through_waypoints):
        super().__init__()
        self.router = router
        # per vehicle, the index of the first waypoint group it has not reached yet
        self._progress: Dict[int, int] = {}

    def reset(self, env: GraphRailEnv):
        super().reset(env)
        self._progress = {}

    def get(self, handle: int = None) -> Dict[int, List[ProjectedEntryPoint]]:
        """
        Advances the progress of all vehicles, then returns the predictions of all vehicles, or of `handle` only.
        Call it after each env step, so no waypoint group reached is missed.
        """
        for agent in self.env.agents:
            self._advance(agent)
        agents = self.env.agents if handle is None else [self.env.agents[handle]]
        return {agent.handle: self._predict(agent) for agent in agents}

    def _advance(self, agent: EnvAgent):
        """
        A waypoint group counts as reached when the vehicle is at one of its entry points - the exact entry point, not
        only its resource: Flatland's default rewards count an intermediate stop as served only at one of the group's
        entry points (`BaseDefaultRewards`), and `ShortestPathPolicy` routes through exactly these. A resource match
        could count a stop as reached when the vehicle crosses its cell in the other direction first.
        """
        progress = self._progress.get(agent.handle, 0)
        while progress < len(agent.waypoints) and agent.current_entry_point in agent.waypoints[progress]:
            progress += 1
        self._progress[agent.handle] = progress

    def _predict(self, agent: EnvAgent) -> List[ProjectedEntryPoint]:
        if agent.state == TrainState.DONE:
            return []
        elapsed_steps = self.env._elapsed_steps
        progress = self._progress[agent.handle]
        if agent.current_entry_point is None:
            source = agent.initial_entry_point
            first_group = 0
        else:
            source = agent.current_entry_point
            # the group the vehicle stands on, so its wait there is projected
            first_group = progress - 1 if progress > 0 and source in agent.waypoints[progress - 1] else progress
        route, stop_indices = self.router(self.env.rail, source, agent.waypoints[first_group:])
        stops = {route_index: first_group + i for i, route_index in enumerate(stop_indices)}

        steps_per_cell = math.ceil(1 / agent.speed_counter.max_speed)
        step, next_step = self._first_steps(agent=agent, first_stop=stops.get(0), elapsed_steps=elapsed_steps, steps_per_cell=steps_per_cell)

        predictions = [ProjectedEntryPoint(route[0], step, stops.get(0))]
        for index in range(1, len(route)):
            step = next_step
            stop = stops.get(index)
            predictions.append(ProjectedEntryPoint(route[index], step, stop))
            earliest_departure = self._earliest_departure(agent, stop)
            if stop is not None and stop < len(agent.waypoints) - 1:
                # halts in the step it enters the stop, moves again with distance 0
                step = max(step + 1, earliest_departure or 0)
            next_step = step + steps_per_cell
        return predictions

    def _first_steps(self, agent: EnvAgent, first_stop: Optional[int], elapsed_steps: int, steps_per_cell: int) -> Tuple[int, int]:
        """
        The step the vehicle is at the first entry point of its route, and the step it enters the second: off the map,
        it enters at `max(earliest departure, elapsed + 1)`; on the map, it is there now, and moves on after its halt
        (speed 0, at the earliest departure of the stop `first_stop` it stands on) and its remaining distance.
        """
        if agent.current_entry_point is None:
            step = max(agent.earliest_departure, elapsed_steps + 1)
            return step, step + steps_per_cell
        moving_from = elapsed_steps
        if agent.speed_counter.speed == 0:
            moving_from = max(elapsed_steps + 1, self._earliest_departure(agent, first_stop) or 0)
        return elapsed_steps, moving_from + math.ceil((1 - agent.speed_counter.distance) / agent.speed_counter.max_speed)

    @staticmethod
    def _earliest_departure(agent: EnvAgent, stop: Optional[int]) -> Optional[int]:
        """ The earliest departure at waypoint group `stop` - `None` if the vehicle doesn't stop there or at the target. """
        if stop is None or stop == len(agent.waypoints) - 1:
            return None
        return agent.waypoints_earliest_departure[stop]
