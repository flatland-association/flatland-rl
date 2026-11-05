from dataclasses import dataclass
from typing import List, Optional

from flatland.envs.agent_utils import EntryPointT


@dataclass(frozen=True)
class PassengerJourney:
    """
    A passenger's journey, with the same fields and conventions as a vehicle's `EnvAgent`:

    - `waypoints`: the waypoint groups in order, each a list of accepted entry points (e.g. the tracks of a station):
      the origin, the stops where the passenger changes vehicles, and the target. Each pair of consecutive groups is
      one ride. The origin group has exactly one entry point, where the passenger waits.
    - `waypoints_earliest_departure`: per group, the earliest step to leave it - `None` at the target.
    - `waypoints_latest_arrival`: per group, the latest step to reach it - `None` at the origin.
    """
    waypoints: List[List[EntryPointT]]
    waypoints_earliest_departure: List[Optional[int]]
    waypoints_latest_arrival: List[Optional[int]]

    def __post_init__(self):
        assert len(self.waypoints) >= 2, f"at least origin and target, got {self.waypoints}"
        assert len(self.waypoints[0]) == 1, f"the origin must have exactly one entry point, got {self.waypoints[0]}"
        assert len(self.waypoints_earliest_departure) == len(self.waypoints_latest_arrival) == len(self.waypoints)
