from typing import List

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_policy import RailEnvPolicy
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState


class ShortestPathPolicy(RailEnvPolicy[RailEnv, RailEnv, RailEnvActions]):
    def __init__(self):
        super().__init__()
        self._shortest_paths = {}

    def _act(self, env: RailEnv, agent: EnvAgent):
        if agent.current_configuration is None:
            return RailEnvActions.MOVE_FORWARD

        if len(self._shortest_paths[agent.handle]) == 0:
            return RailEnvActions.DO_NOTHING

        for a in {RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT}:
            result = env.rail.apply_action_independent(RailEnvActions.from_value(a), agent.current_configuration)
            if result is not None:
                (new_position, new_direction), _ = result
                next_waypoint = self._shortest_paths[agent.handle][1]
                if new_position == next_waypoint.position and new_direction == next_waypoint.direction:
                    return a
        raise Exception("Invalid state")

    def act_many(self, handles: List[int], observations: List[RailEnv], **kwargs):
        actions = {}
        for handle, env in zip(handles, observations):
            agent = env.agents[handle]
            self._update_agent(agent, env)
            actions[handle] = self._act(env, agent)
        return actions

    def _update_agent(self, agent: EnvAgent, env: RailEnv):
        """
        Update `_shortest_paths`.
        """
        if agent.state == TrainState.DONE:
            self._shortest_paths.pop(agent.handle, None)
            return

        if agent.handle not in self._shortest_paths:
            p = [agent.waypoints[0][0]]
            p_next = None
            for pp2 in agent.waypoints[1:]:
                p1: Waypoint = p[-1]
                for p2 in pp2:
                    pp_next = get_k_shortest_paths(None, p1.position, p1.direction, p2.position, target_direction=p2.direction,rail=env.rail)
                    if len(pp_next) > 0:
                        assert pp_next[0][-1] == p2, (p2, pp_next)
                        p_next = pp_next[0][1:]
                        break
            assert p_next is not None, f"Not found next path from {p1} to {pp_next}, agent.waypoints={agent.waypoints}"
            p += p_next
            self._shortest_paths[agent.handle] = p

        if agent.current_configuration is None:
            return

        position = agent.current_configuration[0]
        while self._shortest_paths[agent.handle][0].position != position:
            self._shortest_paths[agent.handle] = self._shortest_paths[agent.handle][1:]
        assert self._shortest_paths[agent.handle][0].position == position
