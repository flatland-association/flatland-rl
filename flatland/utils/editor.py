import numpy as np
from numpy import array
import time
from collections import deque
from matplotlib import pyplot as plt
from contextlib import redirect_stdout
import os

# import io
# from PIL import Image
# from ipywidgets import IntSlider, link, VBox

# from flatland.envs.rail_env import RailEnv, random_rail_generator
# from flatland.core.transitions import RailEnvTransitions
# from flatland.core.env_observation_builder import TreeObsForRailEnv
import flatland.utils.rendertools as rt


class JupEditor(object):
    def __init__(self, env):
        self.env = env
        self.qEvents = deque()

        # TODO: These are currently estimated values
        self.yxBase = array([6, 21])  # pixel offset
        self.nPixCell = 35

        self.rcHistory = []
        self.iTransLast = -1
        self.gRCTrans = array([[-1, 0], [0, 1], [1, 0], [0, -1]])  # NESW in RC
        self.oRT = rt.RenderTool(env)

    def event_handler(self, wid, event):
        """Mouse motion event handler
        """
        x = event['canvasX']
        y = event['canvasY']
        env = self.env
        qEvents = self.qEvents
        rcHistory = self.rcHistory
        bRedrawn = False
        writableData = None

        # If the mouse is held down, enqueue an event in our own queue
        if event["buttons"] > 0:
            qEvents.append((time.time(), x, y))
        
        if len(qEvents) > 0:
            tNow = time.time()
            if tNow - qEvents[0][0] > 0.1:   # wait before trying to draw
                height, width = wid.data.shape[:2]
                writableData = np.copy(wid.data)  # writable copy of image - wid.data is somehow readonly
                
                with wid.hold_sync():
                    while len(qEvents) > 0:
                        t, x, y = qEvents.popleft()  # get events from our queue

                        # Draw a black square
                        if x > 10 and x < width and y > 10 and y < height:
                            writableData[y-2:y+2, x-2:x+2, :] = 0
                        
                        # Translate and scale from x,y to integer row,col (note order change)
                        rcCell = ((array([y, x]) - self.yxBase) / self.nPixCell).astype(int)

                        if len(rcHistory) > 1:
                            rcLast = rcHistory[-1]
                            if not np.array_equal(rcLast, rcCell):  # only save at transition
                                # print(y, x, rcCell)
                                rcHistory.append(rcCell)
                        else:
                            rcHistory.append(rcCell)

        # If we have already touched 3 cells
        # We have a transition into a cell, and out of it.
        if len(rcHistory) >= 3:
            rc3Cells = array(rcHistory[:3])  # the 3 cells
            rcMiddle = rc3Cells[1]  # the middle cell which we will update
            # get the 2 row, col deltas between the 3 cells, eg [-1,0] = North
            rc2Trans = np.diff(rc3Cells, axis=0)
            
            # get the direction index for the 2 transitions
            liTrans = []
            for rcTrans in rc2Trans:
                iTrans = np.argwhere(np.all(self.gRCTrans - rcTrans == 0, axis=1))
                if len(iTrans) > 0:
                    iTrans = iTrans[0][0]
                    liTrans.append(iTrans)

            if len(liTrans) == 2:
                # Set the transition
                # oEnv.rail.set_transition((*rcLast, iTransLast), iTrans, True) # does nothing
                iValCell = env.rail.transitions.set_transition(
                    env.rail.grid[tuple(rcMiddle)], liTrans[0], liTrans[1], True)

                # Also set the reverse transition
                iValCell = env.rail.transitions.set_transition(
                    iValCell,
                    (liTrans[1] + 2) % 4,
                    (liTrans[0] + 2) % 4,
                    True)

                # Write the cell transition value back into the grid
                env.rail.grid[tuple(rcMiddle)] = iValCell
                
                # TODO: bit of a hack - can we suppress the console messages from MPL at source?
                with redirect_stdout(os.devnull):
                    plt.figure(figsize=(10, 10))
                    self.oRT.renderEnv(spacing=False, arrows=False, sRailColor="gray", show=False)
                    img = self.oRT.getImage()
                    plt.clf()
                    plt.close()

                # This updates the image in the browser with the new rendered image
                wid.data = img
                bRedrawn = True
        
            rcHistory.pop(0)  # remove the last-but-one
            
        if not bRedrawn and writableData is not None:
            # This updates the image in the browser to be the new edited version
            wid.data = writableData

