import time
import warnings
from collections import deque
from enum import IntEnum

import numpy as np
from numpy import array
from recordtype import recordtype

from flatland.utils.graphics_pil import PILGL, PILSVG


# TODO: suggested renaming to RailEnvRenderTool, as it will only work with RailEnv!

class AgentRenderVariant(IntEnum):
    BOX_ONLY = 0
    ONE_STEP_BEHIND = 1
    AGENT_SHOWS_OPTIONS = 2
    ONE_STEP_BEHIND_AND_BOX = 3
    AGENT_SHOWS_OPTIONS_AND_BOX = 4


class RenderTool(object):
    """ Class to render the RailEnv and agents.
        Uses two layers, layer 0 for rails (mostly static), layer 1 for agents etc (dynamic)
        The lower / rail layer 0 is only redrawn after set_new_rail() has been called.
        Created with a "GraphicsLayer" or gl - now either PIL or PILSVG
    """
    visit = recordtype("visit", ["rc", "iDir", "iDepth", "prev"])

    color_list = list("brgcmyk")
    # \delta RC for NESW
    transitions_row_col = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
    pix_per_cell = 1  # misnomer...
    half_pix_per_cell = pix_per_cell / 2
    x_y_half = array([half_pix_per_cell, -half_pix_per_cell])
    row_col_to_xy = array([[0, -pix_per_cell], [pix_per_cell, 0]])
    grid = array(np.meshgrid(np.arange(10), -np.arange(10))) * array([[[pix_per_cell]], [[pix_per_cell]]])
    theta = np.linspace(0, np.pi / 2, 5)
    arc = array([np.cos(theta), np.sin(theta)]).T  # from [1,0] to [0,1]

    def __init__(self, env, gl="PILSVG", jupyter=False, agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND):
        self.env = env
        self.frame_nr = 0
        self.start_time = time.time()
        self.times_list = deque()

        self.agent_render_variant = agent_render_variant

        if gl == "PIL":
            self.gl = PILGL(env.width, env.height, jupyter)
        elif gl == "PILSVG":
            self.gl = PILSVG(env.width, env.height, jupyter)
        else:
            print("[", gl, "] not found, switch to PILSVG")
            self.gl = PILSVG(env.width, env.height, jupyter)

        self.new_rail = True
        self.update_background()

    def reset(self):
        """
        Resets the environment
        :return:
        """
        self.set_new_rail()
        self.frame_nr = 0
        self.start_time = time.time()
        self.times_list = deque()
        return

    def update_background(self):
        # create background map
        targets = {}
        for agent_idx, agent in enumerate(self.env.agents_static):
            if agent is None:
                continue
            targets[tuple(agent.target)] = agent_idx
        self.gl.build_background_map(targets)

    def resize(self):
        self.gl.resize(self.env)

    def set_new_rail(self):
        """ Tell the renderer that the rail has changed.
            eg when the rail has been regenerated, or updated in the editor.
        """
        self.new_rail = True

    def plot_agents(self, targets=True, selected_agent=None):
        color_map = self.gl.get_cmap('hsv',
                                     lut=max(len(self.env.agents), len(self.env.agents_static) + 1))

        for agent_idx, agent in enumerate(self.env.agents_static):
            if agent is None:
                continue
            color = color_map(agent_idx)
            self.plot_single_agent(agent.position, agent.direction, color, target=agent.target if targets else None,
                                   static=True, selected=agent_idx == selected_agent)

        for agent_idx, agent in enumerate(self.env.agents):
            if agent is None:
                continue
            color = color_map(agent_idx)
            self.plot_single_agent(agent.position, agent.direction, color, target=agent.target if targets else None)

    def get_transition_row_col(self, row_col_pos, direction, bgiTrans=False):
        """
        Get the available transitions for row_col_pos in direction direction,
        as row & col deltas.

        If bgiTrans is True, return a grid of indices of available transitions.

        eg for a cell row_col_pos = (4,5), in direction direction = 0 (N),
        where the available transitions are N and E, returns:
        [[-1,0], [0,1]] ie N=up one row, and E=right one col.
        and if bgiTrans is True, returns a tuple:
        (
            [[-1,0], [0,1]], # deltas as before
            [0, 1] #  available transition indices, ie N, E
        )
        """

        transitions = self.env.rail.get_transitions(*row_col_pos, direction)
        transition_list = np.where(transitions)[0]  # RC list of transitions

        # HACK: workaround dead-end transitions
        if len(transition_list) == 0:
            reverse_direciton = (direction + 2) % 4
            transitions = tuple(int(tmp_dir == reverse_direciton) for tmp_dir in range(4))
            transition_list = np.where(transitions)[0]  # RC list of transitions

        transition_grid = self.__class__.transitions_row_col[transition_list]

        if bgiTrans:
            return transition_grid, transition_list
        else:
            return transition_grid

    def plot_single_agent(self, position_row_col, direction, color="r", target=None, static=False, selected=False):
        """
        Plot a simple agent.
        Assumes a working graphics layer context (cf a MPL figure).
        """
        rt = self.__class__

        direction_row_col = rt.transitions_row_col[direction]  # agent direction in RC
        direction_xy = np.matmul(direction_row_col, rt.row_col_to_xy)  # agent direction in xy

        xyPos = np.matmul(position_row_col - direction_row_col / 2, rt.row_col_to_xy) + rt.x_y_half

        if static:
            color = self.gl.adapt_color(color, lighten=True)

        color = color

        self.gl.scatter(*xyPos, color=color, layer=1, marker="o", s=100)  # agent location
        xy_dir_line = array([xyPos, xyPos + direction_xy / 2]).T  # line for agent orient.
        self.gl.plot(*xy_dir_line, color=color, layer=1, lw=5, ms=0, alpha=0.6)
        if selected:
            self._draw_square(xyPos, 1, color)

        if target is not None:
            target_row_col = array(target)
            target_xy = np.matmul(target_row_col, rt.row_col_to_xy) + rt.x_y_half
            self._draw_square(target_xy, 1 / 3, color, layer=1)

    def plot_transition(self, position_row_col, transition_row_col, color="r", depth=None):
        """
        plot the transitions in transition_row_col at position position_row_col.
        transition_row_col is a 2d numpy array containing a list of RC transitions,
        eg [[-1,0], [0,1]] means N, E.

        """

        rt = self.__class__
        position_xy = np.matmul(position_row_col, rt.row_col_to_xy) + rt.x_y_half
        transition_xy = position_xy + np.matmul(transition_row_col, rt.row_col_to_xy / 2.4)
        self.gl.scatter(*transition_xy.T, color=color, marker="o", s=50, alpha=0.2)
        if depth is not None:
            for x, y in transition_xy:
                self.gl.text(x, y, depth)

    def draw_transition(self,
                        line,
                        center,
                        rotation,
                        dead_end=False,
                        curves=False,
                        color="gray",
                        arrow=True,
                        spacing=0.1):
        """
        gLine is a numpy 2d array of points,
        in the plotting space / coords.
        eg:
        [[0,.5],[1,0.2]] means a line
        from x=0, y=0.5
        to   x=1, y=0.2
        """

        if not curves and not dead_end:
            # just a straigt line, no curve nor dead_end included in this basic rail element
            self.gl.plot(
                [line[0][0], line[1][0]],  # x
                [line[0][1], line[1][1]],  # y
                color=color
            )
        else:
            # it was not a simple line to draw: the rail has a curve or dead_end included.
            rt = self.__class__
            straight = rotation in [0, 2]
            dx, dy = np.squeeze(np.diff(line, axis=0)) * spacing / 2

            if straight:

                if color == "auto":
                    if dx > 0 or dy > 0:
                        color = "C1"  # N or E
                    else:
                        color = "C2"  # S or W

                if dead_end:
                    line_xy = array([
                        line[1] + [dy, dx],
                        center,
                        line[1] - [dy, dx],
                    ])
                    self.gl.plot(*line_xy.T, color=color)
                else:
                    line_xy = line + [-dy, dx]
                    self.gl.plot(*line_xy.T, color=color)

                    if arrow:
                        middle_xy = np.sum(line_xy * [[1 / 4], [3 / 4]], axis=0)

                        arrow_xy = array([
                            middle_xy + [-dx - dy, +dx - dy],
                            middle_xy,
                            middle_xy + [-dx + dy, -dx - dy]])
                        self.gl.plot(*arrow_xy.T, color=color)

            else:

                middle_xy = np.mean(line, axis=0)
                dxy = middle_xy - center
                corner = middle_xy + dxy
                if rotation == 1:
                    arc_factor = 1 - spacing
                    color_auto = "C1"
                else:
                    arc_factor = 1 + spacing
                    color_auto = "C2"
                dxy2 = (center - corner) * arc_factor  # for scaling the arc

                if color == "auto":
                    color = color_auto

                self.gl.plot(*(rt.arc * dxy2 + corner).T, color=color)

                if arrow:
                    dx, dy = np.squeeze(np.diff(line, axis=0)) / 20
                    iArc = int(len(rt.arc) / 2)
                    middle_xy = corner + rt.arc[iArc] * dxy2
                    arrow_xy = array([
                        middle_xy + [-dx - dy, +dx - dy],
                        middle_xy,
                        middle_xy + [-dx + dy, -dx - dy]])
                    self.gl.plot(*arrow_xy.T, color=color)

    def render_observation(self, agent_handles, observation_dict):
        """
        Render the extent of the observation of each agent. All cells that appear in the agent
        observation will be highlighted.
        :param agent_handles: List of agent indices to adapt color and get correct observation
        :param observation_dict: dictionary containing sets of cells of the agent observation

        """
        rt = self.__class__

        # Check if the observation builder provides an observation
        if len(observation_dict) < 1:
            warnings.warn(
                "Predictor did not provide any predicted cells to render. \
                Observaiton builder needs to populate: env.dev_obs_dict")
        else:
            for agent in agent_handles:
                color = self.gl.get_agent_color(agent)
                for visited_cell in observation_dict[agent]:
                    cell_coord = array(visited_cell[:2])
                    cell_coord_trans = np.matmul(cell_coord, rt.row_col_to_xy) + rt.x_y_half
                    self._draw_square(cell_coord_trans, 1 / (agent + 1.1), color, layer=1, opacity=100)

    def render_prediction(self, agent_handles, prediction_dict):
        """
        Render the extent of the observation of each agent. All cells that appear in the agent
        observation will be highlighted.
        :param agent_handles: List of agent indices to adapt color and get correct observation
        :param observation_dict: dictionary containing sets of cells of the agent observation

        """
        rt = self.__class__
        if len(prediction_dict) < 1:
            warnings.warn(
                "Predictor did not provide any predicted cells to render. \
                Predictors builder needs to populate: env.dev_pred_dict")




        else:
            for agent in agent_handles:
                color = self.gl.get_agent_color(agent)
                for visited_cell in prediction_dict[agent]:
                    cell_coord = array(visited_cell[:2])
                    if type(self.gl) is PILSVG:
                        # TODO : Track highlighting (Adrian)
                        r = cell_coord[0]
                        c = cell_coord[1]
                        transitions = self.env.rail.grid[r, c]
                        self.gl.set_predicion_path_at(r, c, transitions, agent_rail_color=color)
                    else:
                        cell_coord_trans = np.matmul(cell_coord, rt.row_col_to_xy) + rt.x_y_half
                        self._draw_square(cell_coord_trans, 1 / (agent + 1.1), color, layer=1, opacity=100)

    def render_rail(self, spacing=False, rail_color="gray", curves=True, arrows=False):

        cell_size = 1  # TODO: remove cell_size
        env = self.env

        # Draw cells grid
        grid_color = [0.95, 0.95, 0.95]
        for row in range(env.height + 1):
            self.gl.plot([0, (env.width + 1) * cell_size],
                         [-row * cell_size, -row * cell_size],
                         color=grid_color, linewidth=2)
        for col in range(env.width + 1):
            self.gl.plot([col * cell_size, col * cell_size],
                         [0, -(env.height + 1) * cell_size],
                         color=grid_color, linewidth=2)

        # Draw each cell independently
        for row in range(env.height):
            for col in range(env.width):

                # bounding box of the grid cell
                x0 = cell_size * col  # left
                x1 = cell_size * (col + 1)  # right
                y0 = cell_size * -row  # top
                y1 = cell_size * -(row + 1)  # bottom

                # centres of cell edges
                coords = [
                    ((x0 + x1) / 2.0, y0),  # N middle top
                    (x1, (y0 + y1) / 2.0),  # E middle right
                    ((x0 + x1) / 2.0, y1),  # S middle bottom
                    (x0, (y0 + y1) / 2.0)  # W middle left
                ]

                # cell centre
                center_xy = array([x0, y1]) + cell_size / 2

                # cell transition values
                cell = env.rail.get_full_transitions(row, col)

                cell_valid = env.rail.cell_neighbours_valid((row, col), check_this_cell=True)

                # Special Case 7, with a single bit; terminate at center
                nbits = 0
                tmp = cell

                while tmp > 0:
                    nbits += (tmp & 1)
                    tmp = tmp >> 1

                # as above - move the from coord to the centre
                # it's a dead env.
                is_dead_end = nbits == 1

                if not cell_valid:
                    self.gl.scatter(*center_xy, color="r", s=30)

                for orientation in range(4):  # ori is where we're heading
                    from_ori = (orientation + 2) % 4  # 0123=NESW -> 2301=SWNE
                    from_xy = coords[from_ori]

                    moves = env.rail.get_transitions(row, col, orientation)

                    for to_ori in range(4):
                        to_xy = coords[to_ori]
                        rotation = (to_ori - from_ori) % 4
                        if (moves[to_ori]):  # if we have this transition
                            self.draw_transition(
                                array([from_xy, to_xy]), center_xy,
                                rotation, dead_end=is_dead_end, curves=curves and not is_dead_end, spacing=spacing,
                                color=rail_color)

    def render_env(self,
                   show=False,  # whether to call matplotlib show() or equivalent after completion
                   agents=True,  # whether to include agents
                   show_observations=True,  # whether to include observations
                   show_predictions=False,  # whether to include predictions
                   frames=False,  # frame counter to show (intended since invocation)
                   episode=None,  # int episode number to show
                   step=None,  # int step number to show in image
                   selected_agent=None):  # indicate which agent is "selected" in the editor
        """ Draw the environment using the GraphicsLayer this RenderTool was created with.
            (Use show=False from a Jupyter notebook with %matplotlib inline)
        """
        if type(self.gl) is PILSVG:
            self.render_env_svg(show=show,
                                show_observations=show_observations,
                                show_predictions=show_predictions,
                                selected_agent=selected_agent
                                )
        else:
            self.render_env_pil(show=show,
                                agents=agents,
                                show_observations=show_observations,
                                show_predictions=show_predictions,
                                frames=frames,
                                episode=episode,
                                step=step,
                                selected_agent=selected_agent
                                )

    def _draw_square(self, center, size, color, opacity=255, layer=0):
        x0 = center[0] - size / 2
        x1 = center[0] + size / 2
        y0 = center[1] - size / 2
        y1 = center[1] + size / 2
        self.gl.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=color, layer=layer, opacity=opacity)

    def get_image(self):
        return self.gl.get_image()

    def render_env_pil(self,
                       show=False,  # whether to call matplotlib show() or equivalent after completion
                       # use false when calling from Jupyter.  (and matplotlib no longer supported!)
                       agents=True,  # whether to include agents
                       show_observations=True,  # whether to include observations
                       show_predictions=False,  # whether to include predictions
                       frames=False,  # frame counter to show (intended since invocation)
                       episode=None,  # int episode number to show
                       step=None,  # int step number to show in image
                       selected_agent=None  # indicate which agent is "selected" in the editor
                       ):

        if type(self.gl) is PILGL:
            self.gl.begin_frame()

        env = self.env

        self.render_rail()

        # Draw each agent + its orientation + its target
        if agents:
            self.plot_agents(targets=True, selected_agent=selected_agent)
        if show_observations:
            self.render_observation(range(env.get_num_agents()), env.dev_obs_dict)
        if show_predictions and len(env.dev_pred_dict) > 0:
            self.render_prediction(range(env.get_num_agents()), env.dev_pred_dict)
        # Draw some textual information like fps
        text_y = [-0.3, -0.6, -0.9]
        if frames:
            self.gl.text(0.1, text_y[2], "Frame:{:}".format(self.frame_nr))
        self.frame_nr += 1

        if episode is not None:
            self.gl.text(0.1, text_y[1], "Ep:{}".format(episode))

        if step is not None:
            self.gl.text(0.1, text_y[0], "Step:{}".format(step))

        time_now = time.time()
        self.gl.text(2, text_y[2], "elapsed:{:.2f}s".format(time_now - self.start_time))
        self.times_list.append(time_now)
        if len(self.times_list) > 20:
            self.times_list.popleft()
        if len(self.times_list) > 1:
            rFps = (len(self.times_list) - 1) / (self.times_list[-1] - self.times_list[0])
            self.gl.text(2, text_y[1], "fps:{:.2f}".format(rFps))

        self.gl.prettify2(env.width, env.height, self.pix_per_cell)

        # TODO: for MPL, we don't want to call clf (called by endframe)
        # if not show:

        if show and type(self.gl) is PILGL:
            self.gl.show()

        self.gl.pause(0.00001)

        return

    def render_env_svg(
        self, show=False, show_observations=True, show_predictions=False, selected_agent=None
    ):
        """
        Renders the environment with SVG support (nice image)
        """

        env = self.env

        self.gl.begin_frame()

        if self.new_rail:
            self.new_rail = False
            self.gl.clear_rails()

            # store the targets
            targets = {}
            selected = {}
            for agent_idx, agent in enumerate(self.env.agents_static):
                if agent is None:
                    continue
                targets[tuple(agent.target)] = agent_idx
                selected[tuple(agent.target)] = (agent_idx == selected_agent)

            # Draw each cell independently
            for r in range(env.height):
                for c in range(env.width):
                    transitions = env.rail.grid[r, c]
                    if (r, c) in targets:
                        target = targets[(r, c)]
                        is_selected = selected[(r, c)]
                    else:
                        target = None
                        is_selected = False

                    self.gl.set_rail_at(r, c, transitions, target=target, is_selected=is_selected,
                                        rail_grid=env.rail.grid)

            self.gl.build_background_map(targets)

        for agent_idx, agent in enumerate(self.env.agents):

            if agent is None:
                continue

            if self.agent_render_variant == AgentRenderVariant.BOX_ONLY:
                self.gl.set_cell_occupied(agent_idx, *(agent.position))
            elif self.agent_render_variant == AgentRenderVariant.ONE_STEP_BEHIND or \
                self.agent_render_variant == AgentRenderVariant.ONE_STEP_BEHIND_AND_BOX:  # noqa: E125
                if agent.old_position is not None:
                    position = agent.old_position
                    direction = agent.direction
                    old_direction = agent.old_direction
                else:
                    position = agent.position
                    direction = agent.direction
                    old_direction = agent.direction

                # set_agent_at uses the agent index for the color
                if self.agent_render_variant == AgentRenderVariant.ONE_STEP_BEHIND_AND_BOX:
                    self.gl.set_cell_occupied(agent_idx, *(agent.position))
                self.gl.set_agent_at(agent_idx, *position, old_direction, direction, selected_agent == agent_idx)
            else:
                position = agent.position
                direction = agent.direction
                for possible_directions in range(4):
                    # Is a transition along movement `desired_movement_from_new_cell' to the current cell possible?
                    isValid = env.rail.get_transition((*agent.position, agent.direction), possible_directions)
                    if isValid:
                        direction = possible_directions

                        # set_agent_at uses the agent index for the color
                        self.gl.set_agent_at(agent_idx, *position, agent.direction, direction,
                                             selected_agent == agent_idx)

                # set_agent_at uses the agent index for the color
                if self.agent_render_variant == AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX:
                    self.gl.set_cell_occupied(agent_idx, *(agent.position))
                self.gl.set_agent_at(agent_idx, *position, agent.direction, direction, selected_agent == agent_idx)

        if show_observations:
            self.render_observation(range(env.get_num_agents()), env.dev_obs_dict)
        if show_predictions:
            self.render_prediction(range(env.get_num_agents()), env.dev_pred_dict)
        if show:
            self.gl.show()
        for i in range(3):
            self.gl.process_events()

        self.frame_nr += 1
        return

    def close_window(self):
        self.gl.close_window()
