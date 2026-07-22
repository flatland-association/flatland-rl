import math
from collections import defaultdict
from typing import Dict, List, Optional, Generic, TypeVar, Callable, Set, Tuple

from flatland.core.transition_map import TransitionMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.step_utils.states import TrainState

UnderlyingTransitionMapType = TypeVar('UnderlyingTransitionMapType', bound=TransitionMap)
UnderlyingDistanceMapType = TypeVar('UnderlyingDistanceMapType')
UnderlyingConfigurationType = TypeVar('UnderlyingConfigurationType')
UnderlyingWaypointType = TypeVar('UnderlyingWaypointType')


def _infinite_distance():
    return math.inf


class AbstractDistanceMap(Generic[UnderlyingTransitionMapType, UnderlyingDistanceMapType, UnderlyingConfigurationType, UnderlyingWaypointType]):
    def __init__(self, agents: List[EnvAgent], waypoint_init: Callable[[UnderlyingConfigurationType], UnderlyingWaypointType]):
        self.distance_map = None
        self.agents_previous_computation = None
        self.reset_was_called = False
        self.agents: List[EnvAgent] = agents
        self.rail: Optional[RailGridTransitionMap] = None
        self.waypoint_init = waypoint_init

    def set(self, distance_map: UnderlyingDistanceMapType):
        """
        Set the distance map
        """
        self.distance_map = distance_map

    def get(self) -> UnderlyingDistanceMapType:
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

    def reset(self, agents: List[EnvAgent], rail: UnderlyingTransitionMapType):
        """
        Reset the distance map
        """
        self.reset_was_called = True
        self.agents: List[EnvAgent] = agents
        self.rail = rail

    # N.B. get_shortest_paths is not part of distance_map since it refers to RailEnvActions (would lead to circularity!)
    def get_shortest_paths(self, max_depth: Optional[int] = None, agent_handle: Optional[int] = None) -> Dict[int, Optional[List[UnderlyingWaypointType]]]:
        """
        Computes the shortest path for each agent to its target and the action to be taken to do so.
        The paths are derived from a `DistanceMap`.

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
            Dict[int, Optional[List[UnderlyingWaypointType]]]

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
        if agent.state.is_off_map_state():
            configuration = agent.initial_configuration
        elif agent.state.is_on_map_state():
            configuration = agent.current_configuration
        elif agent.state == TrainState.DONE:
            return None
        else:
            return None
        handle = agent.handle
        targets = agent.targets

        return self._reconstruct_shortest_path(configuration, handle, max_depth, targets)

    def _reconstruct_shortest_path(
        self,
        source: UnderlyingConfigurationType,
        handle,
        max_depth: Optional[int],
        targets: Set[UnderlyingConfigurationType]
    ) -> List[UnderlyingWaypointType]:
        """
        Reconstruct shortest path from distance map going forward from source to any of targets.
        """
        agent_shortest_path = []

        distance = math.inf
        depth = 0

        while source not in targets and (max_depth is None or depth < max_depth):
            best_next_configuration = None
            next_configurations = self.rail.get_successor_configurations(source)
            for next_configuration in next_configurations:

                next_action_distance = self.get_agent_distance(next_configuration, handle)
                if next_action_distance < distance:
                    distance = next_action_distance
                    best_next_configuration = next_configuration
            agent_shortest_path.append(self.waypoint_init(source))
            depth += 1

            # if there is no way to continue, the rail must be disconnected!
            # (or distance map is incorrect)
            if best_next_configuration is None:
                return None
            source = best_next_configuration
        if max_depth is None or depth < max_depth:
            agent_shortest_path.append(self.waypoint_init(source))
        return agent_shortest_path

    def _compute(self, agents: List[EnvAgent], rail: UnderlyingTransitionMapType):
        raise NotImplementedError()

    def get_agent_distance(self, source_configuration: UnderlyingConfigurationType, target_nr: int):
        raise NotImplementedError()


class ConfigurationDistanceMap(
    AbstractDistanceMap[UnderlyingTransitionMapType, UnderlyingDistanceMapType, UnderlyingConfigurationType,
    UnderlyingWaypointType]
):
    """
    Intermediate distance map collecting the distance from every configuration visited during the BFS walk to
    the effective target configuration reached, keyed by (source_configuration, target_configuration) - agnostic
    of any numeric target_nr (agent handle), which `DistanceMapWalker` has no notion of. `get_agent_distance`
    returns the minimum distance from a source configuration to any of a given agent's target configurations;
    `_compute()` is responsible for using it to fill in the concrete per-agent storage (via `_set_agent_distance`).
    """

    def __init__(self, agents: List[EnvAgent],
                 waypoint_init: Callable[[UnderlyingConfigurationType], UnderlyingWaypointType]):
        super().__init__(agents=agents, waypoint_init=waypoint_init)
        self.distances: Dict[
            Tuple[UnderlyingConfigurationType, UnderlyingConfigurationType], int
        ] = defaultdict(_infinite_distance)

    def _set_distance(self, source_configuration: UnderlyingConfigurationType,
                      target_configuration: UnderlyingConfigurationType, new_distance: int):
        self.distances[(source_configuration, target_configuration)] = new_distance

    def _get_distance(self, source_configuration: UnderlyingConfigurationType,
                      target_configuration: UnderlyingConfigurationType) -> int:
        return self.distances[(source_configuration, target_configuration)]

    def get_agent_distance(self, source_configuration: UnderlyingConfigurationType, target_nr: int):
        return min(
            self._get_distance(source_configuration, target_configuration)
            for target_configuration in self.agents[target_nr].targets
        )

    def _set_agent_distance(self, source_configuration: UnderlyingConfigurationType, target_nr: int, new_distance: int):
        raise NotImplementedError()
