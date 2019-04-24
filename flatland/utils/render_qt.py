from flatland.utils.graphics_qt import QtRenderer
from numpy import array
from flatland.utils.graphics_layer import GraphicsLayer
from matplotlib import pyplot as plt
import numpy as np


class QTGL(GraphicsLayer):
    def __init__(self, width, height):
        self.cell_pixels = 50
        self.tile_size = self.cell_pixels

        self.width = width
        self.height = height

        # Total grid size at native scale
        self.widthPx = self.width * self.cell_pixels
        self.heightPx = self.height * self.cell_pixels
        self.qtr = QtRenderer(self.widthPx, self.heightPx, ownWindow=True)

        self.qtr.beginFrame()
        self.qtr.push()

        # This comment comes from minigrid.  Not sure if it's still true. Jeremy.
        # Internally, we draw at the "large" full-grid resolution, but we
        # use the renderer to scale back to the desired size
        self.qtr.scale(self.tile_size / self.cell_pixels, self.tile_size / self.cell_pixels)

        self.tColBg = (255, 255, 255)     # white background
        # self.tColBg = (220, 120, 40)    # background color
        self.tColRail = (0, 0, 0)         # black rails
        self.tColGrid = (230,) * 3        # light grey for grid

        # Draw the background of the in-world cells
        self.qtr.fillRect(0, 0, self.widthPx, self.heightPx, *self.tColBg)
        self.qtr.pop()
        self.qtr.endFrame()

    def adaptColor(self, color):
        if color == "red" or color == "r":
            color = (255, 0, 0)
        elif color == "gray":
            color = (128, 128, 128)
        elif type(color) is list:
            color = array(color) * 255
        elif type(color) is tuple:
            gcolor = array(color)
            color = gcolor[:3] * 255
        else:
            color = self.tColGrid
        return color

    def plot(self, gX, gY, color=None, linewidth=2, **kwargs):
        color = self.adaptColor(color)

        self.qtr.setLineColor(*color)
        lastx = lasty = None

        if False:
            for x, y in zip(gX, gY):
                if lastx is not None:
                    # print("line", lastx, lasty, x, y)
                    self.qtr.drawLine(
                        lastx*self.cell_pixels, -lasty*self.cell_pixels,
                        x*self.cell_pixels, -y*self.cell_pixels)
                lastx = x
                lasty = y
        else:
            # print(gX, gY)
            gPoints = np.stack([array(gX), -array(gY)]).T * self.cell_pixels
            self.qtr.drawPolyline(gPoints)

    def scatter(self, gX, gY, color=None, marker="o", size=5, *args, **kwargs):
        color = self.adaptColor(color)
        self.qtr.setColor(*color)
        r = np.sqrt(size)
        gPoints = np.stack([np.atleast_1d(gX), -np.atleast_1d(gY)]).T * self.cell_pixels
        for x, y in gPoints:
            self.qtr.drawCircle(x, y, r)

    def text(self, x, y, sText):
        self.qtr.drawText(x*self.cell_pixels, -y*self.cell_pixels, sText)
    
    def prettify(self, *args, **kwargs):
        pass

    def prettify2(self, width, height, cell_size):
        pass
    
    def show(self, block=False):
        pass

    def pause(self, seconds=0.00001):
        pass
    
    def clf(self):
        pass

    def get_cmap(self, *args, **kwargs):
        return plt.get_cmap(*args, **kwargs)

    def beginFrame(self):
        self.qtr.beginFrame()
        self.qtr.push()
        self.qtr.fillRect(0, 0, self.widthPx, self.heightPx, *self.tColBg)
    
    def endFrame(self):
        self.qtr.pop()
        self.qtr.endFrame()


