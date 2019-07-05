import time
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
    Visit = recordtype("Visit", ["rc", "iDir", "iDepth", "prev"])

    lColors = list("brgcmyk")
    # \delta RC for NESW
    gTransRC = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
    nPixCell = 1  # misnomer...
    nPixHalf = nPixCell / 2
    xyHalf = array([nPixHalf, -nPixHalf])
    grc2xy = array([[0, -nPixCell], [nPixCell, 0]])
    gGrid = array(np.meshgrid(np.arange(10), -np.arange(10))) * array([[[nPixCell]], [[nPixCell]]])
    gTheta = np.linspace(0, np.pi / 2, 5)
    gArc = array([np.cos(gTheta), np.sin(gTheta)]).T  # from [1,0] to [0,1]

    def __init__(self, env, gl="PILSVG", jupyter=False, agentRenderVariant=AgentRenderVariant.ONE_STEP_BEHIND):
        self.env = env
        self.iFrame = 0
        self.time1 = time.time()
        self.lTimes = deque()

        self.agentRenderVariant = agentRenderVariant

        if gl == "PIL":
            self.gl = PILGL(env.width, env.height, jupyter)
        elif gl == "PILSVG":
            self.gl = PILSVG(env.width, env.height, jupyter)
        else:
            print("[", gl, "] not found, switch to PILSVG")
            self.gl = PILSVG(env.width, env.height, jupyter)

        self.new_rail = True
        self.update_background()

    def update_background(self):
        # create background map
        dTargets = {}
        for iAgent, agent in enumerate(self.env.agents_static):
            if agent is None:
                continue
            dTargets[tuple(agent.target)] = iAgent
        self.gl.build_background_map(dTargets)

    def resize(self):
        self.gl.resize(self.env)

    def set_new_rail(self):
        """ Tell the renderer that the rail has changed.
            eg when the rail has been regenerated, or updated in the editor.
        """
        self.new_rail = True

    def plotTreeOnRail(self, lVisits, color="r"):
        """
        DEFUNCT
        Derives and plots a tree of transitions starting at position rcPos
        in direction iDir.
        Returns a list of Visits which are the nodes / vertices in the tree.
        """
        rt = self.__class__

        for visit in lVisits:
            # transition for next cell
            tbTrans = self.env.rail.get_transitions((*visit.rc, visit.iDir))
            giTrans = np.where(tbTrans)[0]  # RC list of transitions
            gTransRCAg = rt.gTransRC[giTrans]
            self.plotTrans(visit.rc, gTransRCAg, depth=str(visit.iDepth), color=color)

    def plotAgents(self, targets=True, iSelectedAgent=None):
        cmap = self.gl.get_cmap('hsv',
                                lut=max(len(self.env.agents), len(self.env.agents_static) + 1))

        for iAgent, agent in enumerate(self.env.agents_static):
            if agent is None:
                continue
            oColor = cmap(iAgent)
            self.plotAgent(agent.position, agent.direction, oColor, target=agent.target if targets else None,
                           static=True, selected=iAgent == iSelectedAgent)

        for iAgent, agent in enumerate(self.env.agents):
            if agent is None:
                continue
            oColor = cmap(iAgent)
            self.plotAgent(agent.position, agent.direction, oColor, target=agent.target if targets else None)

    def getTransRC(self, rcPos, iDir, bgiTrans=False):
        """
        Get the available transitions for rcPos in direction iDir,
        as row & col deltas.

        If bgiTrans is True, return a grid of indices of available transitions.

        eg for a cell rcPos = (4,5), in direction iDir = 0 (N),
        where the available transitions are N and E, returns:
        [[-1,0], [0,1]] ie N=up one row, and E=right one col.
        and if bgiTrans is True, returns a tuple:
        (
            [[-1,0], [0,1]], # deltas as before
            [0, 1] #  available transition indices, ie N, E
        )
        """

        tbTrans = self.env.rail.get_transitions((*rcPos, iDir))
        giTrans = np.where(tbTrans)[0]  # RC list of transitions

        # HACK: workaround dead-end transitions
        if len(giTrans) == 0:
            iDirReverse = (iDir + 2) % 4
            tbTrans = tuple(int(iDir2 == iDirReverse) for iDir2 in range(4))
            giTrans = np.where(tbTrans)[0]  # RC list of transitions

        gTransRCAg = self.__class__.gTransRC[giTrans]

        if bgiTrans:
            return gTransRCAg, giTrans
        else:
            return gTransRCAg

    def plotAgent(self, rcPos, iDir, color="r", target=None, static=False, selected=False):
        """
        Plot a simple agent.
        Assumes a working graphics layer context (cf a MPL figure).
        """
        rt = self.__class__

        rcDir = rt.gTransRC[iDir]  # agent direction in RC
        xyDir = np.matmul(rcDir, rt.grc2xy)  # agent direction in xy

        xyPos = np.matmul(rcPos - rcDir / 2, rt.grc2xy) + rt.xyHalf

        if static:
            color = self.gl.adaptColor(color, lighten=True)

        color = color

        self.gl.scatter(*xyPos, color=color, layer=1, marker="o", s=100)  # agent location
        xyDirLine = array([xyPos, xyPos + xyDir / 2]).T  # line for agent orient.
        self.gl.plot(*xyDirLine, color=color, layer=1, lw=5, ms=0, alpha=0.6)
        if selected:
            self._draw_square(xyPos, 1, color)

        if target is not None:
            rcTarget = array(target)
            xyTarget = np.matmul(rcTarget, rt.grc2xy) + rt.xyHalf
            self._draw_square(xyTarget, 1 / 3, color, layer=1)

    def plotTrans(self, rcPos, gTransRCAg, color="r", depth=None):
        """
        plot the transitions in gTransRCAg at position rcPos.
        gTransRCAg is a 2d numpy array containing a list of RC transitions,
        eg [[-1,0], [0,1]] means N, E.

        """

        rt = self.__class__
        xyPos = np.matmul(rcPos, rt.grc2xy) + rt.xyHalf
        gxyTrans = xyPos + np.matmul(gTransRCAg, rt.grc2xy / 2.4)
        self.gl.scatter(*gxyTrans.T, color=color, marker="o", s=50, alpha=0.2)
        if depth is not None:
            for x, y in gxyTrans:
                self.gl.text(x, y, depth)

    def getTreeFromRail(self, rcPos, iDir, nDepth=10, bBFS=True, bPlot=False):
        """
        DEFUNCT
        Generate a tree from the env starting at rcPos, iDir.
        """
        rt = self.__class__
        print(rcPos, iDir)
        iPos = 0 if bBFS else -1  # BF / DF Search

        iDepth = 0
        visited = set()
        lVisits = []
        stack = [rt.Visit(rcPos, iDir, iDepth, None)]
        while stack:
            visit = stack.pop(iPos)
            rcd = (visit.rc, visit.iDir)
            if visit.iDepth > nDepth:
                continue
            lVisits.append(visit)

            if rcd not in visited:
                visited.add(rcd)

                gTransRCAg, giTrans = self.getTransRC(visit.rc,
                                                      visit.iDir,
                                                      bgiTrans=True)
                # enqueue the next nodes (ie transitions from this node)
                for gTransRC2, iTrans in zip(gTransRCAg, giTrans):
                    visitNext = rt.Visit(tuple(visit.rc + gTransRC2),
                                         iTrans,
                                         visit.iDepth + 1,
                                         visit)
                    stack.append(visitNext)

                # plot the available transitions from this node
                if bPlot:
                    self.plotTrans(
                        visit.rc, gTransRCAg,
                        depth=str(visit.iDepth))

        return lVisits

    def plotTree(self, lVisits, xyTarg):
        '''
        Plot a vertical tree of transitions.
        Returns the "visit" to the destination
        (ie where euclidean distance is near zero) or None if absent.
        '''

        dPos = {}
        iPos = 0

        visitDest = None

        for iVisit, visit in enumerate(lVisits):

            if visit.rc in dPos:
                xLoc = dPos[visit.rc]
            else:
                xLoc = dPos[visit.rc] = iPos
                iPos += 1

            rDist = np.linalg.norm(array(visit.rc) - array(xyTarg))

            xLoc = rDist + visit.iDir / 4

            # point labelled with distance
            self.gl.scatter(xLoc, visit.iDepth, color="k", s=2)
            self.gl.text(xLoc, visit.iDepth, visit.rc, color="k", rotation=45)

            # if len(dPos)>1:
            if visit.prev:
                xLocPrev = dPos[visit.prev.rc]

                rDistPrev = np.linalg.norm(array(visit.prev.rc) -
                                           array(xyTarg))

                xLocPrev = rDistPrev + visit.prev.iDir / 4

                # line from prev node
                self.gl.plot([xLocPrev, xLoc],
                             [visit.iDepth - 1, visit.iDepth],
                             color="k", alpha=0.5, lw=1)

            if rDist < 0.1:
                visitDest = visit

        # Walk backwards from destination to origin, plotting in red
        if visitDest is not None:
            visit = visitDest
            xLocPrev = None
            while visit is not None:
                rDist = np.linalg.norm(array(visit.rc) - array(xyTarg))
                xLoc = rDist + visit.iDir / 4
                if xLocPrev is not None:
                    self.gl.plot([xLoc, xLocPrev], [visit.iDepth, visit.iDepth + 1],
                                 color="r", alpha=0.5, lw=2)
                xLocPrev = xLoc
                visit = visit.prev

        self.gl.prettify()
        return visitDest

    def plotPath(self, visitDest):
        """
        Given a "final" visit visitDest, plotPath recurses back through the path
        using the visit.prev field (previous) to get back to the start of the path.
        The path of transitions is plotted with arrows at 3/4 along the line.
        The transition is plotted slightly to one side of the rail, so that
        transitions in opposite directions are separate.
        Currently, no attempt is made to make the transition arrows coincide
        at corners, and they are straight only.
        """

        rt = self.__class__
        # Walk backwards from destination to origin
        if visitDest is not None:
            visit = visitDest
            xyPrev = None
            while visit is not None:
                xy = np.matmul(visit.rc, rt.grc2xy) + rt.xyHalf
                if xyPrev is not None:
                    dx, dy = (xyPrev - xy) / 20
                    xyLine = array([xy, xyPrev]) + array([dy, dx])

                    self.gl.plot(*xyLine.T, color="r", alpha=0.5, lw=1)

                    xyMid = np.sum(xyLine * [[1 / 4], [3 / 4]], axis=0)

                    xyArrow = array([
                        xyMid + [-dx - dy, +dx - dy],
                        xyMid,
                        xyMid + [-dx + dy, -dx - dy]])
                    self.gl.plot(*xyArrow.T, color="r")

                visit = visit.prev
                xyPrev = xy

    def drawTrans(self, oFrom, oTo, sColor="gray"):
        self.gl.plot(
            [oFrom[0], oTo[0]],  # x
            [oFrom[1], oTo[1]],  # y
            color=sColor
        )

    def drawTrans2(self,
                   xyLine, xyCentre,
                   rotation, bDeadEnd=False,
                   sColor="gray",
                   bArrow=True,
                   spacing=0.1):
        """
        gLine is a numpy 2d array of points,
        in the plotting space / coords.
        eg:
        [[0,.5],[1,0.2]] means a line
        from x=0, y=0.5
        to   x=1, y=0.2
        """
        rt = self.__class__
        bStraight = rotation in [0, 2]
        dx, dy = np.squeeze(np.diff(xyLine, axis=0)) * spacing / 2

        if bStraight:

            if sColor == "auto":
                if dx > 0 or dy > 0:
                    sColor = "C1"  # N or E
                else:
                    sColor = "C2"  # S or W

            if bDeadEnd:
                xyLine2 = array([
                    xyLine[1] + [dy, dx],
                    xyCentre,
                    xyLine[1] - [dy, dx],
                ])
                self.gl.plot(*xyLine2.T, color=sColor)
            else:
                xyLine2 = xyLine + [-dy, dx]
                self.gl.plot(*xyLine2.T, color=sColor)

                if bArrow:
                    xyMid = np.sum(xyLine2 * [[1 / 4], [3 / 4]], axis=0)

                    xyArrow = array([
                        xyMid + [-dx - dy, +dx - dy],
                        xyMid,
                        xyMid + [-dx + dy, -dx - dy]])
                    self.gl.plot(*xyArrow.T, color=sColor)

        else:

            xyMid = np.mean(xyLine, axis=0)
            dxy = xyMid - xyCentre
            xyCorner = xyMid + dxy
            if rotation == 1:
                rArcFactor = 1 - spacing
                sColorAuto = "C1"
            else:
                rArcFactor = 1 + spacing
                sColorAuto = "C2"
            dxy2 = (xyCentre - xyCorner) * rArcFactor  # for scaling the arc

            if sColor == "auto":
                sColor = sColorAuto

            self.gl.plot(*(rt.gArc * dxy2 + xyCorner).T, color=sColor)

            if bArrow:
                dx, dy = np.squeeze(np.diff(xyLine, axis=0)) / 20
                iArc = int(len(rt.gArc) / 2)
                xyMid = xyCorner + rt.gArc[iArc] * dxy2
                xyArrow = array([
                    xyMid + [-dx - dy, +dx - dy],
                    xyMid,
                    xyMid + [-dx + dy, -dx - dy]])
                self.gl.plot(*xyArrow.T, color=sColor)

    def renderObs(self, agent_handles, observation_dict):
        """
        Render the extent of the observation of each agent. All cells that appear in the agent
        observation will be highlighted.
        :param agent_handles: List of agent indices to adapt color and get correct observation
        :param observation_dict: dictionary containing sets of cells of the agent observation

        """
        rt = self.__class__

        for agent in agent_handles:
            color = self.gl.getAgentColor(agent)
            for visited_cell in observation_dict[agent]:
                cell_coord = array(visited_cell[:2])
                cell_coord_trans = np.matmul(cell_coord, rt.grc2xy) + rt.xyHalf
                self._draw_square(cell_coord_trans, 1 / (agent + 1.1), color, layer=1, opacity=100)

    def renderRail(self, spacing=False, sRailColor="gray", curves=True, arrows=False):

        cell_size = 1  # TODO: remove cell_size
        env = self.env

        # Draw cells grid
        grid_color = [0.95, 0.95, 0.95]
        for r in range(env.height + 1):
            self.gl.plot([0, (env.width + 1) * cell_size],
                         [-r * cell_size, -r * cell_size],
                         color=grid_color, linewidth=2)
        for c in range(env.width + 1):
            self.gl.plot([c * cell_size, c * cell_size],
                         [0, -(env.height + 1) * cell_size],
                         color=grid_color, linewidth=2)

        # Draw each cell independently
        for r in range(env.height):
            for c in range(env.width):

                # bounding box of the grid cell
                x0 = cell_size * c  # left
                x1 = cell_size * (c + 1)  # right
                y0 = cell_size * -r  # top
                y1 = cell_size * -(r + 1)  # bottom

                # centres of cell edges
                coords = [
                    ((x0 + x1) / 2.0, y0),  # N middle top
                    (x1, (y0 + y1) / 2.0),  # E middle right
                    ((x0 + x1) / 2.0, y1),  # S middle bottom
                    (x0, (y0 + y1) / 2.0)  # W middle left
                ]

                # cell centre
                xyCentre = array([x0, y1]) + cell_size / 2

                # cell transition values
                oCell = env.rail.get_transitions((r, c))

                bCellValid = env.rail.cell_neighbours_valid((r, c), check_this_cell=True)

                # Special Case 7, with a single bit; terminate at center
                nbits = 0
                tmp = oCell

                while tmp > 0:
                    nbits += (tmp & 1)
                    tmp = tmp >> 1

                # as above - move the from coord to the centre
                # it's a dead env.
                bDeadEnd = nbits == 1

                if not bCellValid:
                    self.gl.scatter(*xyCentre, color="r", s=30)

                for orientation in range(4):  # ori is where we're heading
                    from_ori = (orientation + 2) % 4  # 0123=NESW -> 2301=SWNE
                    from_xy = coords[from_ori]

                    tMoves = env.rail.get_transitions((r, c, orientation))

                    for to_ori in range(4):
                        to_xy = coords[to_ori]
                        rotation = (to_ori - from_ori) % 4

                        if (tMoves[to_ori]):  # if we have this transition

                            if bDeadEnd:
                                self.drawTrans2(
                                    array([from_xy, to_xy]), xyCentre,
                                    rotation, bDeadEnd=True, spacing=spacing,
                                    sColor=sRailColor)

                            else:

                                if curves:
                                    self.drawTrans2(
                                        array([from_xy, to_xy]), xyCentre,
                                        rotation, spacing=spacing, bArrow=arrows,
                                        sColor=sRailColor)
                                else:
                                    self.drawTrans(self, from_xy, to_xy, sRailColor)

                            if False:
                                print(
                                    "r,c,ori: ", r, c, orientation,
                                    "cell:", "{0:b}".format(oCell),
                                    "moves:", tMoves,
                                    "from:", from_ori, from_xy,
                                    "to: ", to_ori, to_xy,
                                    "cen:", *xyCentre,
                                    "rot:", rotation,
                                )

    def renderEnv(self,
                  show=False,  # whether to call matplotlib show() or equivalent after completion
                  # use false when calling from Jupyter.  (and matplotlib no longer supported!)
                  curves=True,  # draw turns as curves instead of straight diagonal lines
                  spacing=False,  # defunct - size of spacing between rails
                  arrows=False,  # defunct - draw arrows on rail lines
                  agents=True,  # whether to include agents
                  show_observations=True,  # whether to include observations
                  sRailColor="gray",  # color to use in drawing rails (not used with SVG)
                  frames=False,  # frame counter to show (intended since invocation)
                  iEpisode=None,  # int episode number to show
                  iStep=None,  # int step number to show in image
                  iSelectedAgent=None,  # indicate which agent is "selected" in the editor
                  action_dict=None):  # defunct - was used to indicate agent intention to turn
        """ Draw the environment using the GraphicsLayer this RenderTool was created with.
            (Use show=False from a Jupyter notebook with %matplotlib inline)
        """

        if not self.gl.is_raster():
            self.renderEnv2(show=show, curves=curves, spacing=spacing,
                            arrows=arrows, agents=agents, show_observations=show_observations,
                            sRailColor=sRailColor,
                            frames=frames, iEpisode=iEpisode, iStep=iStep,
                            iSelectedAgent=iSelectedAgent, action_dict=action_dict)
            return

        if type(self.gl) is PILGL:
            self.gl.beginFrame()

        env = self.env

        self.renderRail()

        # Draw each agent + its orientation + its target
        if agents:
            self.plotAgents(targets=True, iSelectedAgent=iSelectedAgent)
        if show_observations:
            self.renderObs(range(env.get_num_agents()), env.dev_obs_dict)
        # Draw some textual information like fps
        yText = [-0.3, -0.6, -0.9]
        if frames:
            self.gl.text(0.1, yText[2], "Frame:{:}".format(self.iFrame))
        self.iFrame += 1

        if iEpisode is not None:
            self.gl.text(0.1, yText[1], "Ep:{}".format(iEpisode))

        if iStep is not None:
            self.gl.text(0.1, yText[0], "Step:{}".format(iStep))

        tNow = time.time()
        self.gl.text(2, yText[2], "elapsed:{:.2f}s".format(tNow - self.time1))
        self.lTimes.append(tNow)
        if len(self.lTimes) > 20:
            self.lTimes.popleft()
        if len(self.lTimes) > 1:
            rFps = (len(self.lTimes) - 1) / (self.lTimes[-1] - self.lTimes[0])
            self.gl.text(2, yText[1], "fps:{:.2f}".format(rFps))

        self.gl.prettify2(env.width, env.height, self.nPixCell)

        # TODO: for MPL, we don't want to call clf (called by endframe)
        # if not show:

        if show and type(self.gl) is PILGL:
            self.gl.show()

        self.gl.pause(0.00001)

        return

    def _draw_square(self, center, size, color, opacity=255, layer=0):
        x0 = center[0] - size / 2
        x1 = center[0] + size / 2
        y0 = center[1] - size / 2
        y1 = center[1] + size / 2
        self.gl.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=color, layer=layer, opacity=opacity)

    def getImage(self):
        return self.gl.getImage()

    def plotTreeObs(self, gObs):
        nBranchFactor = 4

        gP0 = array([[0, 0, 0]]).T
        nDepth = 2
        for i in range(nDepth):
            nDepthNodes = nBranchFactor ** i
            rShrinkDepth = 1 / (i + 1)

            gX1 = np.linspace(-(nDepthNodes - 1), (nDepthNodes - 1), nDepthNodes) * rShrinkDepth
            gY1 = np.ones((nDepthNodes)) * i
            gZ1 = np.zeros((nDepthNodes))

            gP1 = array([gX1, gY1, gZ1])
            gP01 = np.append(gP0, gP1, axis=1)

            if nDepthNodes > 1:
                nDepthNodesPrev = nDepthNodes / nBranchFactor
                giP0 = np.repeat(np.arange(nDepthNodesPrev), nBranchFactor)
                giP1 = np.arange(0, nDepthNodes) + nDepthNodesPrev
                giLinePoints = np.stack([giP0, giP1]).ravel("F")
                self.gl.plot(gP01[0], -gP01[1], lines=giLinePoints, color="gray")

            gP0 = array([gX1, gY1, gZ1])

    def renderEnv2(
        self, show=False, curves=True, spacing=False, arrows=False, agents=True,
        show_observations=True, sRailColor="gray",
        frames=False, iEpisode=None, iStep=None, iSelectedAgent=None,
        action_dict=dict()
    ):
        """
        Draw the environment using matplotlib.
        Draw into the figure if provided.

        Call pyplot.show() if show==True.
        (Use show=False from a Jupyter notebook with %matplotlib inline)
        """

        env = self.env

        self.gl.beginFrame()

        if self.new_rail:
            self.new_rail = False
            self.gl.clear_rails()

            # store the targets
            dTargets = {}
            dSelected = {}
            for iAgent, agent in enumerate(self.env.agents_static):
                if agent is None:
                    continue
                dTargets[tuple(agent.target)] = iAgent
                dSelected[tuple(agent.target)] = (iAgent == iSelectedAgent)

            # Draw each cell independently
            for r in range(env.height):
                for c in range(env.width):
                    binTrans = env.rail.grid[r, c]
                    if (r, c) in dTargets:
                        target = dTargets[(r, c)]
                        isSelected = dSelected[(r, c)]
                    else:
                        target = None
                        isSelected = False

                    self.gl.setRailAt(r, c, binTrans, iTarget=target, isSelected=isSelected, rail_grid=env.rail.grid)

            self.gl.build_background_map(dTargets)

        for iAgent, agent in enumerate(self.env.agents):

            if agent is None:
                continue

            if self.agentRenderVariant == AgentRenderVariant.BOX_ONLY:
                self.gl.setCellOccupied(iAgent, *(agent.position))
            elif self.agentRenderVariant == AgentRenderVariant.ONE_STEP_BEHIND or \
                self.agentRenderVariant == AgentRenderVariant.ONE_STEP_BEHIND_AND_BOX:  # noqa: E125
                if agent.old_position is not None:
                    position = agent.old_position
                    direction = agent.direction
                    old_direction = agent.old_direction
                else:
                    position = agent.position
                    direction = agent.direction
                    old_direction = agent.direction

                # setAgentAt uses the agent index for the color
                if self.agentRenderVariant == AgentRenderVariant.ONE_STEP_BEHIND_AND_BOX:
                    self.gl.setCellOccupied(iAgent, *(agent.position))
                self.gl.setAgentAt(iAgent, *position, old_direction, direction, iSelectedAgent == iAgent)
            else:
                position = agent.position
                direction = agent.direction
                for possible_directions in range(4):
                    # Is a transition along movement `desired_movement_from_new_cell' to the current cell possible?
                    isValid = env.rail.get_transition((*agent.position, agent.direction), possible_directions)
                    if isValid:
                        direction = possible_directions

                        # setAgentAt uses the agent index for the color
                        self.gl.setAgentAt(iAgent, *position, agent.direction, direction, iSelectedAgent == iAgent)

                # setAgentAt uses the agent index for the color
                if self.agentRenderVariant == AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX:
                    self.gl.setCellOccupied(iAgent, *(agent.position))
                self.gl.setAgentAt(iAgent, *position, agent.direction, direction, iSelectedAgent == iAgent)

        if show_observations:
            self.renderObs(range(env.get_num_agents()), env.dev_obs_dict)

        if show:
            self.gl.show()
        for i in range(3):
            self.gl.processEvents()

        self.iFrame += 1
        return

    def close_window(self):
        self.gl.close_window()
