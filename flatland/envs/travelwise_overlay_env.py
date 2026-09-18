from typing import Dict, Optional, Set, Tuple

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


def _derive_overlay_route(rail_env: RailEnv) -> Tuple[Tuple[int, int], Tuple[int, int], Grid4TransitionsEnum]:
    """
    The overlay agent's route spans the full corridor used by `rail_env`'s own agents: from the
    westmost position any of them starts at or targets, to the eastmost. Assumes a single-row
    corridor - every position any agent uses shares the same row.
    """
    positions: Set[Tuple[int, int]] = set()
    for agent in rail_env.agents:
        positions.add(agent.initial_entry_point[0])
        for target_position, _ in agent.targets:
            positions.add(target_position)
    start = min(positions, key=lambda position: position[1])
    target = max(positions, key=lambda position: position[1])
    direction = Grid4TransitionsEnum.EAST if target[1] > start[1] else Grid4TransitionsEnum.WEST
    return start, target, direction


class TravelwiseOverlayEnv:
    """
    Layers a second, single-agent env over a `RailEnv` ("the underlying env"): the overlay env shares
    the underlying env's rail topology (transition map), but is otherwise wholly independent of it -
    separate agent, separate timetable, separate motion checks. The overlay agent shuttles the full
    span of the underlying env's own route (its westmost start to its eastmost target) on its own
    timetable, driven by its own `ShortestPathPolicy` - the `policy` passed in only ever drives the
    underlying env's own agents.
    """

    def __init__(self, rail_env: RailEnv, policy: Policy):
        self.rail_env = rail_env
        self.policy = policy
        self.overlay_policy = ShortestPathPolicy()

        self.overlay_env: Optional[RailEnv] = None
        self.overlay_start: Optional[Tuple[int, int]] = None
        self.overlay_target: Optional[Tuple[int, int]] = None
        self.overlay_earliest_departure: Optional[int] = None
        self.overlay_latest_arrival: Optional[int] = None

        self._rail_env_obs = None
        self._overlay_obs = None
        self._rail_env_done = False
        self._overlay_done = False

    def reset(self):
        self._rail_env_obs, rail_env_info = self.rail_env.reset()
        self._rail_env_done = False

        start, target, direction = _derive_overlay_route(self.rail_env)
        self.overlay_start = start
        self.overlay_target = target
        distance = abs(target[1] - start[1])
        self.overlay_earliest_departure = OVERLAY_EARLIEST_DEPARTURE
        self.overlay_latest_arrival = OVERLAY_EARLIEST_DEPARTURE + distance

        def _line_generator(rail, num_agents, hints, num_resets, np_random) -> Line:
            return Line(agent_waypoints={0: [[Waypoint(start, direction)], [Waypoint(target, direction)]]}, agent_speeds=[1.0])

        def _timetable_generator(agents, distance_map, hints, np_random) -> Timetable:
            return Timetable(earliest_departures=[[self.overlay_earliest_departure, None]],
                             latest_arrivals=[[None, self.overlay_latest_arrival]],
                             max_episode_steps=self.overlay_latest_arrival)

        self.overlay_env = RailEnv(width=self.rail_env.width, height=self.rail_env.height,
                                   rail_generator=rail_from_grid_transition_map(self.rail_env.rail),
                                   line_generator=_line_generator,
                                   timetable_generator=_timetable_generator,
                                   number_of_agents=1,
                                   obs_builder_object=FullEnvObservation())
        self._overlay_obs, overlay_info = self.overlay_env.reset()
        self._overlay_done = False

        return (self._rail_env_obs, self._overlay_obs), (rail_env_info, overlay_info)

    def step(self) -> Tuple[Dict, Dict]:
        """
        Advances whichever of the underlying/overlay envs is not yet done - each with its own policy -
        and returns both `dones` dicts. A finished env is left untouched (`RailEnv.step()` itself
        refuses a call once its episode is done), so the other can keep running past it.
        """
        rail_env_dones = {'__all__': self._rail_env_done}
        if not self._rail_env_done:
            actions = self.policy.act_many(self.rail_env.get_agent_handles(), observations=list(self._rail_env_obs.values()))
            self._rail_env_obs, _, rail_env_dones, _ = self.rail_env.step(actions)
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
