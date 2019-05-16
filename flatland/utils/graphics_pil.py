
from flatland.utils.graphics_layer import GraphicsLayer
from PIL import Image, ImageDraw   # , ImageFont
from numpy import array
import numpy as np


class PILGL(GraphicsLayer):
    def __init__(self, width, height, nPixCell=60):
        self.nPixCell = 60
        self.yxBase = (0, 0)
        self.linewidth = 4
        # self.tile_size = self.nPixCell

        self.width = width
        self.height = height

        # Total grid size at native scale
        self.widthPx = self.width * self.nPixCell + self.linewidth
        self.heightPx = self.height * self.nPixCell + self.linewidth
        self.beginFrame()

        self.tColBg = (255, 255, 255)     # white background
        # self.tColBg = (220, 120, 40)    # background color
        self.tColRail = (0, 0, 0)         # black rails
        self.tColGrid = (230,) * 3        # light grey for grid

    def plot(self, gX, gY, color=None, linewidth=3, **kwargs):
        color = self.adaptColor(color)

        # print(gX, gY)
        gPoints = np.stack([array(gX), -array(gY)]).T * self.nPixCell
        gPoints = list(gPoints.ravel())
        # print(gPoints, color)
        self.draw.line(gPoints, fill=color, width=self.linewidth)

    def scatter(self, gX, gY, color=None, marker="o", s=50, *args, **kwargs):
        color = self.adaptColor(color)
        r = np.sqrt(s)
        gPoints = np.stack([np.atleast_1d(gX), -np.atleast_1d(gY)]).T * self.nPixCell
        for x, y in gPoints:
            self.draw.rectangle([(x - r, y - r), (x + r, y + r)], fill=color, outline=color)

    def text(self, *args, **kwargs):
        pass

    def prettify(self, *args, **kwargs):
        pass

    def prettify2(self, width, height, cell_size):
        pass

    def beginFrame(self):
        self.img = Image.new("RGBA", (self.widthPx, self.heightPx), (255, 255, 255, 255))
        self.draw = ImageDraw.Draw(self.img)

    def show(self, block=False):
        pass
        # plt.show(block=block)

    def pause(self, seconds=0.00001):
        pass
        # plt.pause(seconds)

    def getImage(self):
        return array(self.img)
