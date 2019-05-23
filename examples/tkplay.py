import time
import tkinter as tk

from PIL import ImageTk, Image

from examples.play_model import Player
from flatland.envs.generators import complex_rail_generator
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool


def tkmain(n_trials=2, n_steps=50):
    # This creates the main window of an application
    window = tk.Tk()
    window.title("Join")
    window.configure(background='grey')

    # Example generate a random rail
    env = RailEnv(width=15, height=15,
                  rail_generator=complex_rail_generator(nr_start_goal=5, nr_extra=20, min_dist=12),
                  number_of_agents=5)

    env_renderer = RenderTool(env, gl="PIL")

    oPlayer = Player(env)
    n_trials = 1
    delay = 0
    for trials in range(1, n_trials + 1):

        # Reset environment8
        oPlayer.reset()
        env_renderer.set_new_rail()

        first = True

        for step in range(n_steps):
            oPlayer.step()
            env_renderer.renderEnv(show=True, frames=True, iEpisode=trials, iStep=step,
                                   action_dict=oPlayer.action_dict)
            img = env_renderer.getImage()
            img = Image.fromarray(img)
            tkimg = ImageTk.PhotoImage(img)

            if first:
                panel = tk.Label(window, image=tkimg)
                panel.pack(side="bottom", fill="both", expand="yes")
            else:
                # update the image in situ
                panel.configure(image=tkimg)
                panel.image = tkimg

            window.update()
            if delay > 0:
                time.sleep(delay)
            first = False


if __name__ == "__main__":
    tkmain()
