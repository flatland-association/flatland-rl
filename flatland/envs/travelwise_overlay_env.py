from typing import Dict, List, Optional, Set, Tuple

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.policy import Policy
from flatland.envs.observations import FullEnvObservation
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_policies import ShortestPathPolicy
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.timetable_utils import Line, Timetable

# `RailEnv`'s off-map departure timing has documented/undocumented edge cases at earliest_departure 0
# and 1 (e.g. rail_env.py's own "an earliest_departure=0 agent dispatches directly on the very first
# movement action" comment) where the step an agent actually appears on the map doesn't equal
# earliest_departure exactly. 2 is the smallest value clear of both, so the overlay agent's departure
# step always equals `earliest_departure` and its arrival step always equals `earliest_departure +
# distance`, exactly - the same clean arithmetic the wrapped env's own legs rely on.
OVERLAY_EARLIEST_DEPARTURE = 2


def _derive_overlay_stops(rail_env: RailEnv) -> List[Tuple[Tuple[int, int], Grid4TransitionsEnum]]:
    """
    The overlay agent's route stops at every distinct position used by `rail_env`'s own agents (as a
    start or a target), ordered west to east - so it passes through every station the underlying env's
    own legs use, not just the westmost/eastmost extremes, and stops at any of them that lies strictly
    in between too. Assumes a single-row corridor - every position any agent uses shares the same row.
    """
    positions: Set[Tuple[int, int]] = set()
    for agent in rail_env.agents:
        positions.add(agent.initial_entry_point[0])
        for target_position, _ in agent.targets:
            positions.add(target_position)
    ordered = sorted(positions, key=lambda position: position[1])
    direction = Grid4TransitionsEnum.EAST if ordered[-1][1] > ordered[0][1] else Grid4TransitionsEnum.WEST
    return [(position, direction) for position in ordered]


class TravelwiseOverlayEnv:
    """
    Layers a second, single-agent env over a `RailEnv` ("the underlying env"): the overlay env shares
    the underlying env's rail topology (transition map), but is otherwise wholly independent of it -
    separate agent, separate timetable, separate motion checks. The overlay agent shuttles the full
    span of the underlying env's own route (every distinct station used by its agents, from westmost
    to eastmost - stopping at any of them in between too) on its own timetable, driven by its own
    `overlay_policy` (`ShortestPathPolicy` by default) - the `policy` passed in only ever drives the
    underlying env's own agents.
    """

    def __init__(self, rail_env: RailEnv, policy: Policy, overlay_policy: Optional[Policy] = None):
        self.rail_env = rail_env
        self.policy = policy
        self.overlay_policy = overlay_policy if overlay_policy is not None else ShortestPathPolicy()

        self.overlay_env: Optional[RailEnv] = None
        self.overlay_stops: Optional[List[Tuple[int, int]]] = None
        self.overlay_start: Optional[Tuple[int, int]] = None
        self.overlay_target: Optional[Tuple[int, int]] = None
        self.overlay_earliest_departures: Optional[List[Optional[int]]] = None
        self.overlay_latest_arrivals: Optional[List[Optional[int]]] = None
        self.overlay_earliest_departure: Optional[int] = None
        self.overlay_latest_arrival: Optional[int] = None

        self._rail_env_obs = None
        self._overlay_obs = None
        self._rail_env_done = False
        self._overlay_done = False
        self.rail_env_info: Optional[Dict] = None

    def reset(self):
        self._rail_env_obs, self.rail_env_info = self.rail_env.reset()
        self._rail_env_done = False

        stops = _derive_overlay_stops(self.rail_env)
        self.overlay_stops = [position for position, _ in stops]
        self.overlay_start = stops[0][0]
        self.overlay_target = stops[-1][0]

        # cumulative distance (cells) from the first stop up to each stop, at speed 1 - no dwell at an
        # intermediate stop, so its earliest departure and latest arrival are the same step: the step
        # the train reaches it.
        cumulative_distances = [0]
        for (position, _), (next_position, _) in zip(stops, stops[1:]):
            cumulative_distances.append(cumulative_distances[-1] + abs(next_position[1] - position[1]))
        arrival_steps = [OVERLAY_EARLIEST_DEPARTURE + distance for distance in cumulative_distances]
        self.overlay_earliest_departures = arrival_steps[:-1] + [None]
        self.overlay_latest_arrivals = [None] + arrival_steps[1:]
        self.overlay_earliest_departure = arrival_steps[0]
        self.overlay_latest_arrival = arrival_steps[-1]

        def _line_generator(rail, num_agents, hints, num_resets, np_random) -> Line:
            return Line(agent_waypoints={0: [[Waypoint(position, direction)] for position, direction in stops]}, agent_speeds=[1.0])

        def _timetable_generator(agents, distance_map, hints, np_random) -> Timetable:
            return Timetable(earliest_departures=[self.overlay_earliest_departures],
                             latest_arrivals=[self.overlay_latest_arrivals],
                             max_episode_steps=self.overlay_latest_arrival)

        self.overlay_env = RailEnv(width=self.rail_env.width, height=self.rail_env.height,
                                   rail_generator=rail_from_grid_transition_map(self.rail_env.rail),
                                   line_generator=_line_generator,
                                   timetable_generator=_timetable_generator,
                                   number_of_agents=1,
                                   obs_builder_object=FullEnvObservation())
        self._overlay_obs, overlay_info = self.overlay_env.reset()
        self._overlay_done = False

        return (self._rail_env_obs, self._overlay_obs), (self.rail_env_info, overlay_info)

    def step(self) -> Tuple[Dict, Dict]:
        """
        Advances whichever of the underlying/overlay envs is not yet done - each with its own policy -
        and returns both `dones` dicts. A finished env is left untouched (`RailEnv.step()` itself
        refuses a call once its episode is done), so the other can keep running past it.
        """
        rail_env_dones = {'__all__': self._rail_env_done}
        if not self._rail_env_done:
            actions = self.policy.act_many(self.rail_env.get_agent_handles(), observations=list(self._rail_env_obs.values()))
            self._rail_env_obs, _, rail_env_dones, self.rail_env_info = self.rail_env.step(actions)
            self._rail_env_done = rail_env_dones['__all__']

        overlay_dones = {'__all__': self._overlay_done}
        if not self._overlay_done:
            overlay_actions = self.overlay_policy.act_many(self.overlay_env.get_agent_handles(), observations=list(self._overlay_obs.values()))
            self._overlay_obs, _, overlay_dones, _ = self.overlay_env.step(overlay_actions)
            self._overlay_done = overlay_dones['__all__']

        return rail_env_dones, overlay_dones

    @property
    def done(self) -> bool:
        return self._rail_env_done and self._overlay_done
