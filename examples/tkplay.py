try:
    from examples.play_model import Player
except ImportError:
    from play_model import Player

from flatland.envs.generators import complex_rail_generator
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool


def tkmain(n_trials=2, n_steps=50, sGL="PIL"):
    # Example generate a random rail
    env = RailEnv(width=15, height=15,
                  rail_generator=complex_rail_generator(nr_start_goal=5, nr_extra=20, min_dist=12),
                  number_of_agents=5)

    env_renderer = RenderTool(env, gl=sGL)

    oPlayer = Player(env)
    n_trials = 1
    for trials in range(1, n_trials + 1):

        # Reset environment8
        oPlayer.reset()
        env_renderer.set_new_rail()

        for step in range(n_steps):
            oPlayer.step()
            env_renderer.renderEnv(show=True, frames=True, iEpisode=trials, iStep=step,
                                   action_dict=oPlayer.action_dict)

    env_renderer.close_window()


if __name__ == "__main__":
    tkmain(sGL="PIL")
    tkmain(sGL="PILSVG")
