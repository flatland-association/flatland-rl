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
        if agent.position is None:
            return RailEnvActions.MOVE_FORWARD

        if len(self._shortest_paths[agent.handle]) == 0:
            return RailEnvActions.DO_NOTHING

        for a in {RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT}:
            result = env.rail.apply_action_independent(RailEnvActions.from_value(a), (agent.position, agent.direction))
            if result is not None:
                (new_position, new_direction), _ = result
                if new_position == self._shortest_paths[agent.handle][1].position and new_direction == self._shortest_paths[agent.handle][1].direction:
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
            p = []
            for pp1, pp2 in zip(agent.waypoints, agent.waypoints[1:]):
                p1: Waypoint = pp1[0]
                p2: Waypoint = pp2[0]
                if len(p) > 0:
                    assert p[-1] == p1, (p[-1], p1)
                pp_next = get_k_shortest_paths(None, p1.position, p1.direction, p2.position, rail=env.rail)
                p_next = None
                if p2.direction is None:
                    p_next = pp_next[0]
                else:
                    for _p_next in pp_next:
                        if _p_next[-1].direction == p2.direction:
                            p_next = _p_next
                            break
                assert p_next is not None, f"Not found next path from {p1} to {p2}"
                if len(p) > 0:
                    p += p_next[1:]
                else:
                    p += p_next
            self._shortest_paths[agent.handle] = p

        if agent.position is None:
            return

        while self._shortest_paths[agent.handle][0].position != agent.position:
            self._shortest_paths[agent.handle] = self._shortest_paths[agent.handle][1:]
        assert self._shortest_paths[agent.handle][0].position == agent.position
