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


class ConfigurationDistanceMap(Generic[UnderlyingTransitionMapType, UnderlyingDistanceMapType, UnderlyingConfigurationType, UnderlyingWaypointType]):
    """
    Base distance map collecting the distance from every configuration visited during the BFS walk to the
    effective target configuration reached, keyed by (source_configuration, target_configuration) - agnostic
    of any numeric target_nr (agent handle), which `DistanceMapWalker` has no notion of.
    """

    def __init__(self, agents: List[EnvAgent], waypoint_init: Callable[[UnderlyingConfigurationType], UnderlyingWaypointType]):
        self.agents: List[EnvAgent] = agents
        self.rail: Optional[RailGridTransitionMap] = None
        self.waypoint_init = waypoint_init
        self.distances: Dict[
            Tuple[UnderlyingConfigurationType, UnderlyingConfigurationType], int
        ] = defaultdict(_infinite_distance)

    def reset(self, agents: List[EnvAgent], rail: UnderlyingTransitionMapType):
        """
        Reset the distance map
        """
        self.agents: List[EnvAgent] = agents
        self.rail = rail
        self.distances: Dict[
            Tuple[UnderlyingConfigurationType, UnderlyingConfigurationType], int
        ] = defaultdict(_infinite_distance)

    def _set_distance(self, source_configuration: UnderlyingConfigurationType,
                      target_configuration: UnderlyingConfigurationType, new_distance: int):
        self.distances[(source_configuration, target_configuration)] = new_distance

    def _get_distance(self, source_configuration: UnderlyingConfigurationType,
                      target_configuration: UnderlyingConfigurationType) -> int:
        return self.distances[(source_configuration, target_configuration)]


class AgentSourceTargetDistanceMap(
    ConfigurationDistanceMap[UnderlyingTransitionMapType, UnderlyingDistanceMapType, UnderlyingConfigurationType,
    UnderlyingWaypointType]
):
    """
    Adds agent-handle (target_nr) aware querying on top of `ConfigurationDistanceMap`. `get_agent_distance`
    returns the minimum distance from a source configuration to any of a given agent's target configurations;
    concrete subclasses provide the underlying per-agent storage via `_set_agent_distance`.
    """

    def __init__(self, agents: List[EnvAgent], waypoint_init: Callable[[UnderlyingConfigurationType], UnderlyingWaypointType]):
        super().__init__(agents=agents, waypoint_init=waypoint_init)
        self.distance_map = None
        self.agents_previous_computation = None
        self.reset_was_called = False

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
        super().reset(agents=agents, rail=rail)
        self.reset_was_called = True

    def _compute(self, agents: List[EnvAgent], rail: UnderlyingTransitionMapType):
        """
        Computes the distance maps for each unique target. Thus, if several targets are the same we only
        compute the distance for them once and copy to all agents with the same target.
        """
        from flatland.core.distance_map_walker import DistanceMapWalker

        self.agents_previous_computation = self.agents
        self.distance_map = self._new_distance_map(len(agents))
        distance_map_walker = DistanceMapWalker(self)
        computed_targets = []
        for i, agent in enumerate(agents):
            targets = self._valid_targets(agent, rail)
            if targets not in computed_targets:
                reachable_configurations = distance_map_walker._distance_map_walker(rail, targets)
                for configuration in reachable_configurations:
                    new_distance = min(
                        (self._get_distance(configuration, target_configuration) for target_configuration in targets),
                        default=math.inf
                    )
                    self._set_agent_distance(configuration, i, new_distance)
            else:
                # just copy the distance map from other agent with same target (performance)
                self._copy_agent_distance(i, computed_targets.index(targets))
            computed_targets.append(targets)

    def _new_distance_map(self, num_agents: int) -> UnderlyingDistanceMapType:
        raise NotImplementedError()

    def _valid_targets(self, agent: EnvAgent, rail: UnderlyingTransitionMapType) -> List[UnderlyingConfigurationType]:
        raise NotImplementedError()

    def _copy_agent_distance(self, target_nr: int, source_target_nr: int):
        raise NotImplementedError()

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

    def get_agent_distance(self, source_configuration: UnderlyingConfigurationType, target_nr: int):
        self.get()
        return self._get_agent_distance(source_configuration, target_nr)

    def _set_agent_distance(self, source_configuration: UnderlyingConfigurationType, target_nr: int, new_distance: int):
        raise NotImplementedError()

    def _get_agent_distance(self, source_configuration: UnderlyingConfigurationType, target_nr: int):
        raise NotImplementedError()
