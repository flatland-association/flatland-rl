from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.utils.rendertools import RenderTool
from flatland.utils.simple_rail import make_simple_rail


def test_render_env_one_step_behind_after_agent_done():
    """
    Coverage test: no test previously exercised `RenderTool.render_env()`'s `ONE_STEP_BEHIND`
    variant (the default `agent_render_variant`), including the just-departed-agent transition
    where `current_configuration` becomes `None` while `old_configuration` still holds the last
    on-map position - see `flatland/utils/rendertools.py`.
    """
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1)
    env.reset(False, False)

    # place the agent on the map so `current_configuration` becomes a real (position, direction) tuple
    agent = env.agents[0]
    for _ in range(5):
        env.step({0: RailEnvActions.MOVE_FORWARD})
        if agent.current_configuration is not None:
            break
    assert agent.current_configuration is not None

    # simulate the agent having just reached DONE and been removed from the map this step, as
    # `RailEnv.handle_done_state()` does: `old_configuration` still holds the last on-map position.
    agent.old_configuration = agent.current_configuration
    agent.current_configuration = None

    renderer = RenderTool(env, gl="PILSVG")
    renderer.render_env(show=False, show_observations=False)
