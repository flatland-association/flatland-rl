from typing import Callable

from flatland.action_plan.action_plan import ControllerFromTrainruns
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import Waypoint

ControllerFromTrainrunsReplayerRenderCallback = Callable[[RailEnv], None]


class ControllerFromTrainrunsReplayer():
    """Allows to verify a `DeterministicController` by replaying it against a FLATland env without malfunction."""

    @staticmethod
    def replay_verify(ctl: ControllerFromTrainruns, env: RailEnv,
                      call_back: ControllerFromTrainrunsReplayerRenderCallback = lambda *a, **k: None):
        """Replays this deterministic `ActionPlan` and verifies whether it is feasible.

        Parameters
        ----------
        ctl
        env
        call_back
            Called before/after each step() call. The env is passed to it.
        """
        call_back(env)
        i = 0
        while not env.dones['__all__'] and i <= env._max_episode_steps:
            for agent_id, agent in enumerate(env.agents):
                waypoint: Waypoint = ctl.get_waypoint_before_or_at_step(agent_id, i)
                assert agent.position == waypoint.position, \
                    "before {}, agent {} at {}, expected {}".format(i, agent_id, agent.position,
                                                                    waypoint.position)
            actions = ctl.act(i)
            print("actions for {}: {}".format(i, actions))

            obs, all_rewards, done, _ = env.step(actions)

            call_back(env)

            i += 1
