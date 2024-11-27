import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool

delta = 0.2
offsets = {
    # N
    0: [0.5, -delta],
    # E
    1: [-delta, -0.5],
    # S
    2: [-0.5, delta],
    # W
    3: [delta, 0.5]
}


def get_positions(g: nx.DiGraph):
    """
    Get positions for Flatland 3 DiGraph rendering.

    Parameters
    ----------
    g: The DiGraph of the Graph Transition Map

    Returns
    -------
    Dict[GridNode, Tuple[int,int]]
        The position dict
    """
    return {(r, c, d): (c + offsets[d][1], r + offsets[d][0]) for (r, c, d) in g}


def add_flatland_styling(env: RailEnv, ax: plt.Axes):
    """
    Adds Flatland 3 background image, sets ticks and grid.

    Parameters
    ----------
    env: RailEnv
        The underlying rail env
    ax: plt.Axes
        The ax to style
    """
    env_renderer = RenderTool(env)
    img = env_renderer.render_env(show=False, frames=True, show_observations=False, show_predictions=False, return_image=True)
    ax.set_ylim(env.height - 0.5, -0.5)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_xticks(np.arange(0, env.width, 1))
    ax.set_yticks(np.arange(0, env.height, 1))
    # TODO image does not fill extent entirely - why?
    ax.imshow(np.fliplr(np.rot90(np.rot90(img))), extent=[-0.5, env.width - 0.5, -0.5, env.height - 0.5])
    ax.set_xticks(np.arange(-0.5, env.width + 0.5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height + 0.5, 1), minor=True)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.grid(which="minor")
