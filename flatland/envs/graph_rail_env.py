import ast
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
from numpy.random.mtrand import RandomState

from flatland.core.effects_generator import EffectsGenerator
from flatland.core.env_observation_builder import ObservationBuilder, DummyObservationBuilder
from flatland.core.graph.graph_resource_map import GraphResourceMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.graph.distance_map import GraphDistanceMap
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap
from flatland.envs.malfunction_generators import MalfunctionGenerator, ParamMalfunctionGen
from flatland.envs.rail_env import RailEnv, AbstractRailEnv
from flatland.envs.rewards import Rewards
from flatland.envs.step_utils.speed_counter import SpeedCounter
from flatland.envs.timetable_utils import Line, TimetableUtils
from flatland.utils.seeding import random_state_to_hashablestate, random_state_from_hashablestate


TimetableGenerator = Callable[[List[EnvAgent], GraphDistanceMap, dict, RandomState], "Timetable"]


class GraphRailEnv(AbstractRailEnv[GraphTransitionMap, GraphResourceMap, str]):
    @staticmethod
    def from_rail_env(rail_env: RailEnv, observation_builder: ObservationBuilder, seed: Optional[int] = None,
                      rewards: Rewards = None) -> "GraphRailEnv":
        """
        Parameters
        ----------
        rewards: Rewards, optional
            the `Rewards` instance for `graph_env` to accumulate its own reward state into - must be a
            separate instance from `rail_env.rewards` (never the same object: `Rewards` accumulates
            mutable per-episode state, e.g. `arrivals`/`departures`/`states`, which `rail_env` and
            `graph_env` must not share/corrupt). Defaults to `GraphRailEnv`'s own default (`None`) if
            not given - pass a fresh instance matching `rail_env.rewards`'s type/config for parity. If
            the resulting `graph_env.rewards` type ends up different from `rail_env.rewards`'s, a
            `UserWarning` is raised (e.g. if `rewards` is left `None` while `rail_env.rewards` is not
            `GraphRailEnv`'s own default).
        """
        g = GraphTransitionMap.grid_to_digraph(rail_env.rail)
        resource_map = GraphRailEnv._grid_resource_map(rail_env, g)
        agent_waypoints, agent_speeds = GraphRailEnv._grid_agent_waypoints_and_speeds(rail_env)
        timetable = TimetableUtils.from_agents(rail_env.agents, rail_env._max_episode_steps)

        graph_env = GraphRailEnv.from_graph(
            g=g,
            resource_map=resource_map,
            agent_waypoints=agent_waypoints,
            agent_speeds=agent_speeds,
            observation_builder=observation_builder,
            # TODO https://github.com/flatland-association/flatland-rl/issues/242 generalize malfunction generator injection
            # N.B. ParamMalfunctionGen is not stateless due to cached random nums, see https://github.com/flatland-association/flatland-rl/issues/364.
            malfunction_generator=ParamMalfunctionGen(rail_env.malfunction_generator.MFP),
            timetable_generator=lambda *args, **kwargs: timetable,
            seed=seed,
            rewards=rewards,
        )
        GraphRailEnv._warn_if_rewards_mismatch(rail_env, graph_env)
        # TODO https://github.com/flatland-association/flatland-rl/pull/341 hack while awaiting this pr
        s = random_state_to_hashablestate(rail_env.np_random)
        graph_env.np_random = random_state_from_hashablestate(s)
        return graph_env

    @staticmethod
    def _grid_resource_map(rail_env: RailEnv, g: nx.DiGraph) -> Dict[str, str]:
        """
        Maps each grid-derived graph node to its resource: the underlying `(row, col)` cell, or
        `(row, col, direction % 2)` for a level-free (diamond) crossing so the two crossing axes count
        as distinct resources - mirrors `GridResourceMap.get_resource()`.
        """
        resource_map = {}
        for n in g.nodes:
            r, c, d = ast.literal_eval(n)
            if (r, c) in rail_env.resource_map.level_free_positions:
                resource_map[n] = str((r, c, d % 2))
            else:
                resource_map[n] = str((r, c))
        return resource_map

    @staticmethod
    def _grid_agent_waypoints_and_speeds(rail_env: RailEnv) -> Tuple[Dict[int, List[List[str]]], Dict[int, float]]:
        """Converts `rail_env`'s agents' grid `Waypoint`-based waypoints/speeds into the plain
        string-keyed shape `from_graph` expects."""
        gctgc = GraphTransitionMap.grid_entry_point_to_graph_entry_point
        agent_waypoints = {
            agent.handle: [[gctgc(*wp.position, wp.direction) for wp in group] for group in agent.waypoints]
            for agent in rail_env.agents
        }
        agent_speeds = {agent.handle: agent.speed_counter.max_speed for agent in rail_env.agents}
        return agent_waypoints, agent_speeds

    @staticmethod
    def _warn_if_rewards_mismatch(rail_env: RailEnv, graph_env: "GraphRailEnv") -> None:
        if type(rail_env.rewards) is not type(graph_env.rewards):
            warnings.warn(
                f"rail_env.rewards is {type(rail_env.rewards).__name__}, but graph_env.rewards is "
                f"{type(graph_env.rewards).__name__} (no matching `rewards` was passed to from_rail_env) - "
                f"rewards will not be directly comparable between the two envs."
            )

    @staticmethod
    def from_graph(
        g: nx.DiGraph,
        resource_map: Dict[str, str],
        agent_waypoints: Dict[int, List[List[str]]],
        agent_speeds: Optional[Dict[int, float]] = None,
        observation_builder: ObservationBuilder = None,
        malfunction_generator: "MalfunctionGenerator" = None,
        timetable_generator: Optional[TimetableGenerator] = None,
        seed: Optional[int] = None,
        rewards: Rewards = None,
    ) -> "GraphRailEnv":
        """
        Factory method to create a `GraphRailEnv` directly from a string-node graph and string-based
        agent waypoints - counterpart to `from_rail_env`, but graph-native from the start: `g`'s nodes
        and `agent_waypoints`' leaves are plain entry point strings, never `((row, col), direction)`
        grid tuples or `Waypoint` objects.

        Parameters
        ----------
        g: nx.DiGraph
            the rail topology, with `actions`/`straight` edge attributes and an optional
            `prohibited_actions` node attribute - see `GraphTransitionMap.grid_to_digraph` for the
            shape expected by `RailEnv.step()`.
        resource_map: Dict[str, str]
            maps each node in `g` to the resource (occupancy unit) used for conflict detection.
        agent_waypoints: Dict[int, List[List[str]]]
            per agent handle, the list of waypoint alternative-groups (initial, any intermediate
            stops, target) - mirrors `Line.agent_waypoints`, but with plain node-id strings instead of
            `Waypoint` objects.
        agent_speeds: Dict[int, float], optional
            per agent handle, the agent's speed - defaults to `1.0` for every agent.
        timetable_generator: optional
            `(agents, distance_map, agents_hints, np_random) -> Timetable` - defaults to
            `ttg.ttgen_flatland2` (`earliest_departure=0`/`latest_arrival=1000` for every agent). Pass
            e.g. `lambda *a, **k: TimetableUtils.from_agents(source_agents, max_episode_steps)` to
            reuse an existing timetable instead (mirrors how `from_rail_env` reuses its source env's).
        """
        timetable_generator, agent_speeds = GraphRailEnv._resolve_from_graph_defaults(
            timetable_generator, agent_speeds, agent_waypoints)
        gtm = GraphTransitionMap(g)
        line = Line(agent_waypoints=agent_waypoints, agent_speeds=agent_speeds)

        graph_env = GraphRailEnv(
            number_of_agents=len(agent_waypoints),
            rail_generator=lambda *args, **kwargs: ({"resource_map": resource_map}, gtm),
            line_generator=lambda *args, **kwargs: line,
            timetable_generator=timetable_generator,
            observation_builder=observation_builder,
            malfunction_generator=malfunction_generator,
            rewards=rewards,
        )
        graph_env.reset(random_seed=seed)
        return graph_env

    @staticmethod
    def _resolve_from_graph_defaults(
        timetable_generator: Optional[TimetableGenerator],
        agent_speeds: Optional[Dict[int, float]],
        agent_waypoints: Dict[int, List[List[str]]],
    ) -> Tuple[TimetableGenerator, Dict[int, float]]:
        """Resolves `from_graph`'s optional `timetable_generator`/`agent_speeds` to their concrete
        defaults (`ttg.ttgen_flatland2`, uniform speed `1.0`) - a local import avoids a circular import
        with `flatland.envs.timetable_generators` at module load time."""
        import flatland.envs.timetable_generators as ttg

        if timetable_generator is None:
            timetable_generator = ttg.ttgen_flatland2
        if agent_speeds is None:
            agent_speeds = {handle: 1.0 for handle in agent_waypoints}
        return timetable_generator, agent_speeds

    def __init__(
        self,
        # TODO https://github.com/flatland-association/flatland-rl/issues/242 fix signature
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

    def _infrastructure_representation(self, entry_point: str) -> str:
        return entry_point

    def _agents_from_line(self, line: "Line", rail: GraphTransitionMap) -> List[EnvAgent[str]]:
        """
        Builds `EnvAgent`s directly from a `Line` whose `agent_waypoints` are plain graph node-id
        strings - counterpart to `EnvAgent.from_line` for a graph-native `Line` (no `Waypoint` objects
        or grid `((row, col), direction)` tuples involved at all).
        """
        agents = []
        for handle, waypoints in line.agent_waypoints.items():
            speed = line.agent_speeds[handle] if line.agent_speeds is not None else 1.0
            waypoints = list(waypoints)
            # N.B. only the target's alternatives (last waypoint group) can be invalid - the caller's
            # own routing already guarantees valid entry points everywhere else.
            waypoints[-1] = [t for t in waypoints[-1] if rail.is_valid_entry_point(t)]
            assert len(waypoints[-1]) > 0, (
                f"agent {handle}: none of the target alternatives {list(line.agent_waypoints[handle][-1])} "
                f"are valid entry points in the graph - the agent would end up with an empty `targets`."
            )
            initial_entry_point = waypoints[0][0]
            agents.append(EnvAgent(
                initial_entry_point=initial_entry_point,
                current_entry_point=initial_entry_point,
                old_entry_point=None,
                targets=set(waypoints[-1]),
                waypoints=waypoints,
                moving=False,
                earliest_departure=None,
                latest_arrival=None,
                waypoints_earliest_departure=None,
                waypoints_latest_arrival=None,
                handle=handle,
                speed_counter=SpeedCounter(max_speed=speed)))
        return agents
