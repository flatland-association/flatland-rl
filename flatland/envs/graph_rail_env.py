import ast
from typing import List, Any, Optional, Dict

from flatland.core.effects_generator import EffectsGenerator
from flatland.core.env_observation_builder import ObservationBuilder, DummyObservationBuilder
from flatland.core.graph.graph_resource_map import GraphResourceMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import AbstractDistanceMap
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap
from flatland.envs.rail_env import RailEnv, AbstractRailEnv
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.rewards import Rewards
from flatland.envs.timetable_utils import TimetableUtils


class GraphDistanceMap(AbstractDistanceMap[GraphTransitionMap, Any]):
    # TODO implement/generalize distance map for graph
    def _compute(self, agents: List[EnvAgent], rail: GraphTransitionMap):
        pass

    def get_shortest_paths(self, max_depth: Optional[int] = None, agent_handle: Optional[int] = None) -> Dict[int, Optional[List[Waypoint]]]:
        if agent_handle is not None:
            return {agent_handle: []}
        return {a.handle: [] for a in self.agents}


class GraphRailEnv(AbstractRailEnv[GraphTransitionMap, GraphResourceMap, str]):
    @staticmethod
    def from_rail_env(rail_env: RailEnv, observation_builder: ObservationBuilder) -> "GraphRailEnv":
        rail_env.reset(False, False)
        line = EnvAgent.to_line(rail_env.agents)
        timetable = TimetableUtils.from_agents(rail_env.agents, rail_env._max_episode_steps)

        gtm = GraphTransitionMap.from_rail_env(rail_env)
        _resource_map = {}
        for n in gtm.g.nodes:
            # TODO temporary workaround as long as step function relies on tuples
            r, c, d = ast.literal_eval(n)
            if (r, c) in rail_env.resource_map.level_free_positions:
                _resource_map[((r, c), d)] = (r, c, d % 2)
            else:
                _resource_map[((r, c), d)] = (r, c)

        return GraphRailEnv(
            number_of_agents=rail_env.get_num_agents(),
            rail_generator=lambda *args, **kwargs: ({"resource_map": _resource_map}, gtm),
            line_generator=lambda *args, **kwargs: line,
            timetable_generator=lambda *arg, **kwargs: timetable,
            observation_builder=observation_builder,
        )

    def __init__(
        self,
        # TODO fix signature https://github.com/flatland-association/flatland-rl/issues/242
        rail_generator: "RailGenerator" = None,
        line_generator: "LineGenerator" = None,
        number_of_agents=2,
        observation_builder: ObservationBuilder = None,
        malfunction_generator_and_process_data=None,
        malfunction_generator: "MalfunctionGenerator" = None,
        random_seed=None,
        timetable_generator=None,
        acceleration_delta=1.0,
        braking_delta=-1.0,
        rewards: Rewards = None,
        effects_generator: EffectsGenerator["GraphRailEnv"] = None,
        distance_map: GraphDistanceMap = None
    ):
        if observation_builder is None:
            observation_builder = DummyObservationBuilder()
        super().__init__(
            rail_generator=rail_generator,
            line_generator=line_generator,
            number_of_agents=number_of_agents,
            obs_builder_object=observation_builder,
            malfunction_generator_and_process_data=malfunction_generator_and_process_data,
            malfunction_generator=malfunction_generator,
            random_seed=random_seed,
            timetable_generator=timetable_generator,
            acceleration_delta=acceleration_delta,
            braking_delta=braking_delta,
            rewards=rewards,
            effects_generator=effects_generator,
            distance_map=GraphDistanceMap([]) if distance_map is None else distance_map,
        )
        self.agents = [EnvAgent(None, None, None) for i in range(self.get_num_agents())]

    def get_num_agents(self) -> int:
        return self.number_of_agents

    def _extract_resource_map_from_optionals(self, optionals: dict) -> GraphResourceMap:
        if "resource_map" in optionals:
            return GraphResourceMap(optionals["resource_map"])
        else:
            return GraphResourceMap({})

    def _infrastructure_representation(self, configuration: str) -> str:
        return configuration
