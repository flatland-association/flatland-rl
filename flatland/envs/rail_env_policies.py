from typing import List

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_policy import RailEnvPolicy
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths


class ShortestPathPolicy(RailEnvPolicy[RailEnv, RailEnv, RailEnvActions]):
    def __init__(self):
        super().__init__()
        self._shortest_paths = {}
        self._remaining_targets = {}

    def _act(self, env: RailEnv, agent: EnvAgent):
        if agent.position is None:
            return RailEnvActions.MOVE_FORWARD

        if agent.handle not in self._remaining_targets:
            self._remaining_targets[agent.handle] = agent.waypoints

        shortest_path = self._shortest_paths[agent.handle]
        while shortest_path[0].position != agent.position:
            shortest_path = shortest_path[1:]
        assert shortest_path[0].position == agent.position

        if agent.position == self._remaining_targets[agent.handle][0]:
            self._remaining_targets[agent.handle] = self._remaining_targets[agent.handle][1:]
            if len(self._remaining_targets[agent.handle]) > 0:
                self._shortest_paths[agent.handle] = \
                    get_k_shortest_paths(env, agent.position, agent.direction, self._remaining_targets[agent.handle][0].position)[0]

        if len(self._remaining_targets[agent.handle]) == 0:
            return RailEnvActions.DO_NOTHING

        for a in {RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT}:
            new_cell_valid, new_direction, new_position, transition_valid, preprocessed_action = env.rail.check_action_on_agent(
                RailEnvActions.from_value(a),
                agent.position,
                agent.direction
            )
            if new_cell_valid and transition_valid and (
                new_position == self._remaining_targets[agent.handle][0] or (
                new_position == shortest_path[1].position and new_direction == shortest_path[1].direction)):
                return a
        raise Exception("Invalid state")

    def act_many(self, handles: List[int], observations: List[RailEnv], **kwargs):
        actions = {}
        for handle, env in zip(handles, observations):
            agent = env.agents[handle]
            if agent.handle not in self._shortest_paths:
                self._shortest_paths[agent.handle] = get_k_shortest_paths(env, agent.initial_position, agent.initial_direction, agent.target)[0]
            actions[handle] = self._act(env, agent)
        return actions
