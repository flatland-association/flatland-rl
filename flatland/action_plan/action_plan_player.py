from flatland.action_plan.action_plan import ControllerFromTrainruns
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.utils.rendertools import RenderTool, AgentRenderVariant


class ControllerFromTrainrunsReplayer():
    """Allows to verify a `DeterministicController` by replaying it against a FLATland env without malfunction."""

    @staticmethod
    def replay_verify(ctl: ControllerFromTrainruns, env: RailEnv, rendering: bool):
        """Replays this deterministic `ActionPlan` and verifies whether it is feasible."""
        if rendering:
            renderer = RenderTool(env, gl="PILSVG",
                                  agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                                  show_debug=True,
                                  clear_debug_text=True,
                                  screen_height=1000,
                                  screen_width=1000)
            renderer.render_env(show=True, show_observations=False, show_predictions=False)
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

            if rendering:
                renderer.render_env(show=True, show_observations=False, show_predictions=False)

            i += 1
