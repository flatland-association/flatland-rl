from typing import List, Optional, Tuple

from attr import attrib, attrs

from flatland.core.resource_map import ResourceT
from flatland.envs.agent_utils import EntryPointT
from flatland.envs.travelwise.passenger_journey import PassengerJourney

# A passenger's `mode` tuple - a resource (grid cell, graph node's resource; not a full entry point), then a handle -
# holds in each of its 3 modes:
# - off map:                       (None, None)
# - on map, waiting:               (passenger_resource, None)
# - riding a vehicle:              (passenger_resource, vehicle handle), with the vehicle's own
#   resource always equal to passenger_resource
PassengerMode = Tuple[Optional[ResourceT], Optional[int]]


@attrs
class Passenger:
    """
    A passenger's state in `PassengerEnv`, named as an `EnvAgent`'s:

    - `handle`, `journey`: the passenger's handle and its journey;
    - `mode`: where it is, a `PassengerMode` - its resource, and the handle of the vehicle it rides (once riding, the
      vehicle being ridden);
    - `current_entry_point`: the entry point it waits at or rides through - `None` off the map;
    - `stop_index`: the index of the journey's waypoint group it is at or last left (`PassengerJourney.waypoints`):
      its origin until it has completed its first ride, the target's index once arrived. Counts repeated visits of a
      group;
    - `target_entry_point`, `arrival_time`: where and in which step it arrived at its target - `None` before, as
      `AbstractRailEnv.handle_done_state()` sets them on an `EnvAgent` reaching its target.

    Written by `PassengerEnv` only.
    """
    handle = attrib(type=int)
    journey = attrib(type=PassengerJourney)
    mode = attrib(type=PassengerMode, default=(None, None))
    current_entry_point = attrib(type=Optional[EntryPointT], default=None)
    stop_index = attrib(type=int, default=0)
    target_entry_point = attrib(type=Optional[EntryPointT], default=None)
    arrival_time = attrib(type=Optional[int], default=None)

    def next_stop(self) -> Optional[List[EntryPointT]]:
        """ The waypoint group the passenger travels to next (a change or its target) - `None` once arrived. """
        waypoints = self.journey.waypoints
        return waypoints[self.stop_index + 1] if self.stop_index + 1 < len(waypoints) else None
