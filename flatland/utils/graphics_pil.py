
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
        self.layers = []
        self.draws = []

        self.tColBg = (255, 255, 255)     # white background
        # self.tColBg = (220, 120, 40)    # background color
        self.tColRail = (0, 0, 0)         # black rails
        self.tColGrid = (230,) * 3        # light grey for grid

        self.beginFrame()

    def plot(self, gX, gY, color=None, linewidth=3, layer=0, opacity=255, **kwargs):
        color = self.adaptColor(color)
        if len(color) == 3:
            color += (opacity,)
        elif len(color) == 4:
            color = color[:3] + (opacity,)
        gPoints = np.stack([array(gX), -array(gY)]).T * self.nPixCell
        gPoints = list(gPoints.ravel())
        self.draws[layer].line(gPoints, fill=color, width=self.linewidth)

    def scatter(self, gX, gY, color=None, marker="o", s=50, layer=0, opacity=255, *args, **kwargs):
        color = self.adaptColor(color)
        r = np.sqrt(s)
        gPoints = np.stack([np.atleast_1d(gX), -np.atleast_1d(gY)]).T * self.nPixCell
        for x, y in gPoints:
            self.draws[layer].rectangle([(x - r, y - r), (x + r, y + r)], fill=color, outline=color)

    def text(self, *args, **kwargs):
        pass

    def prettify(self, *args, **kwargs):
        pass

    def prettify2(self, width, height, cell_size):
        pass

    def beginFrame(self):
        self.create_layer(0)
        self.create_layer(1)

    def show(self, block=False):
        pass
        # plt.show(block=block)

    def pause(self, seconds=0.00001):
        pass
        # plt.pause(seconds)

    def alpha_composite_layers(self):
        img = self.layers[0]
        for img2 in self.layers[1:]:
            img = Image.alpha_composite(img, img2)
        return img

    def getImage(self):
        """ return a blended / alpha composited image composed of all the layers,
            with layer 0 at the "back".
        """
        img = self.alpha_composite_layers()
        return array(img)

    def create_image(self, opacity=255):
        img = Image.new("RGBA", (self.widthPx, self.heightPx), (255, 255, 255, opacity))
        return img

    def create_layer(self, iLayer=0):
        if len(self.layers) <= iLayer:
            for i in range(len(self.layers), iLayer+1):
                if i==0:
                    opacity = 255  # "bottom" layer is opaque (for rails)
                else:
                    opacity = 0   # subsequent layers are transparent
                img = self.create_image(opacity)
                self.layers.append(img)
                self.draws.append(ImageDraw.Draw(img))
        else:
            opacity = 0 if iLayer > 0 else 255
            self.layers[iLayer] = img = self.create_image(opacity)
            self.draws[iLayer] = ImageDraw.Draw(img)

