
import matplotlib.pyplot as plt
from numpy import array

import os
if os.name=='nt':
    import site
    import ctypes.util
    default_os_path = os.environ['PATH']
    os.environ['PATH'] = ''
    for s in site.getsitepackages():
        os.environ['PATH'] = os.environ['PATH'] + ';' + s + '\\cairo'
    os.environ['PATH'] = os.environ['PATH'] + ';' + default_os_path
    if ctypes.util.find_library('cairo') is not None:
        print("cairo installed: OK")

class GraphicsLayer(object):
    def __init__(self):
        pass

    def open_window(self):
        pass

    def is_raster(self):
        return True

    def plot(self, *args, **kwargs):
        pass

    def scatter(self, *args, **kwargs):
        pass

    def text(self, *args, **kwargs):
        pass

    def prettify(self, *args, **kwargs):
        pass

    def show(self, block=False):
        pass

    def pause(self, seconds=0.00001):
        pass

    def clf(self):
        pass

    def beginFrame(self):
        pass

    def endFrame(self):
        pass

    def getImage(self):
        pass

    def saveImage(self,filename):
        pass

    def adaptColor(self, color, lighten=False):
        if type(color) is str:
            if color == "red" or color == "r":
                color = (255, 0, 0)
            elif color == "gray":
                color = (128, 128, 128)
        elif type(color) is list:
            color = tuple((array(color) * 255).astype(int))
        elif type(color) is tuple:
            if type(color[0]) is not int:
                gcolor = array(color)
                color = tuple((gcolor[:3] * 255).astype(int))
        else:
            color = self.tColGrid

        if lighten:
            color = tuple([int(255 - (255 - iRGB) / 3) for iRGB in color])

        return color

    def get_cmap(self, *args, **kwargs):
        return plt.get_cmap(*args, **kwargs)

    def setRailAt(self, row, col, binTrans, iTarget=None):
        """ Set the rail at cell (row, col) to have transitions binTrans.
            The target argument can contain the index of the agent to indicate
            that agent's target is at that cell, so that a station can be
            rendered in the static rail layer.
        """
        pass

    def setAgentAt(self, iAgent, row, col, iDirIn, iDirOut):
        pass

    def resize(self, env):
        pass
