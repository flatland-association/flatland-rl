from flatland.utils.graphics_layer import GraphicsLayer
from PIL import Image, ImageDraw, ImageTk   # , ImageFont
import tkinter as tk
from numpy import array
import numpy as np
# from flatland.utils.svg import Track, Zug
import time
import io
import os
import site

def enable_windows_cairo_support():
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
enable_windows_cairo_support()
from cairosvg import svg2png
from screeninfo import get_monitors

# from copy import copy
from flatland.core.transitions import RailEnvTransitions




class PILGL(GraphicsLayer):
    def __init__(self, width, height, jupyter=False):
        self.yxBase = (0, 0)
        self.linewidth = 4
        self.nAgentColors = 1  # overridden in loadAgent
        # self.tile_size = self.nPixCell

        self.width = width
        self.height = height


        if jupyter == False:
            self.screen_width = 99999
            self.screen_height = 99999
            for m in get_monitors():
                self.screen_height = min(self.screen_height,m.height)
                self.screen_width = min(self.screen_width,m.width)

            w = (self.screen_width-self.width-10)/(self.width + 1 + self.linewidth)
            h = (self.screen_height-self.height-10)/(self.height + 1 + self.linewidth)
            self.nPixCell = int(max(1,np.ceil(min(w,h))))
        else:
            self.nPixCell = 40

        # Total grid size at native scale
        self.widthPx = self.width * self.nPixCell + self.linewidth
        self.heightPx = self.height * self.nPixCell + self.linewidth

        self.xPx = int((self.screen_width - self.widthPx) / 2.0)
        self.yPx = int((self.screen_height - self.heightPx) / 2.0)

        self.layers = []
        self.draws = []

        self.tColBg = (255, 255, 255)     # white background
        # self.tColBg = (220, 120, 40)    # background color
        self.tColRail = (0, 0, 0)         # black rails
        self.tColGrid = (230,) * 3        # light grey for grid

        sColors = "d50000#c51162#aa00ff#6200ea#304ffe#2962ff#0091ea#00b8d4#00bfa5#00c853" + \
            "#64dd17#aeea00#ffd600#ffab00#ff6d00#ff3d00#5d4037#455a64"
            
        self.ltAgentColors = [self.rgb_s2i(sColor) for sColor in sColors.split("#")]
        self.nAgentColors = len(self.ltAgentColors)

        self.window_open = False
        # self.bShow = show
        self.firstFrame = True
        self.create_layers()
        # self.beginFrame()

    def rgb_s2i(self, sRGB):
        """ convert a hex RGB string like 0091ea to 3-tuple of ints """
        return tuple(int(sRGB[iRGB * 2:iRGB * 2 + 2], 16) for iRGB in [0, 1, 2])

    def getAgentColor(self, iAgent):
        return self.ltAgentColors[iAgent % self.nAgentColors]

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

    def drawImageXY(self, pil_img, xyPixLeftTop, layer=0):
        # self.layers[layer].alpha_composite(pil_img, offset=xyPixLeftTop)
        if (pil_img.mode == "RGBA"): 
            pil_mask = pil_img
        else:
            pil_mask = None
            # print(pil_img, pil_img.mode, xyPixLeftTop, layer)
        
        self.layers[layer].paste(pil_img, xyPixLeftTop, pil_mask)

    def drawImageRC(self, pil_img, rcTopLeft, layer=0):
        xyPixLeftTop = tuple((array(rcTopLeft) * self.nPixCell)[[1, 0]])
        self.drawImageXY(pil_img, xyPixLeftTop, layer=layer)

    def open_window(self):
        assert self.window_open is False, "Window is already open!"
        self.window = tk.Tk()
        self.window.title("Flatland")
        self.window.configure(background='grey')
        #self.window.geometry('%dx%d+%d+%d' % (self.widthPx, self.heightPx, self.xPx, self.yPx))
        self.window_open = True

    def close_window(self):
        self.panel.destroy()
        self.window.quit()
        self.window.destroy()

    def text(self, *args, **kwargs):
        pass

    def prettify(self, *args, **kwargs):
        pass

    def prettify2(self, width, height, cell_size):
        pass

    def beginFrame(self):
        # Create a new agent layer
        self.create_layer(iLayer=1, clear=True)

    def show(self, block=False):
        img = self.alpha_composite_layers()

        if not self.window_open:
            self.open_window()
        
        tkimg = ImageTk.PhotoImage(img)
        
        if self.firstFrame:
            # Do TK actions for a new panel (not sure what they really do)
            self.panel = tk.Label(self.window, image=tkimg)
            self.panel.pack(side="bottom", fill="both", expand="yes")
        else:
            # update the image in situ
            self.panel.configure(image=tkimg)
            self.panel.image = tkimg

        self.window.update()
        self.firstFrame = False

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


    def saveImage(self,filename):
        print(filename)
        img = self.alpha_composite_layers()
        img.save(filename)


    def create_image(self, opacity=255):
        img = Image.new("RGBA", (self.widthPx, self.heightPx), (255, 255, 255, opacity))
        return img

    def create_layer(self, iLayer=0, clear=True):
        # If we don't have the layers already, create them
        if len(self.layers) <= iLayer:
            for i in range(len(self.layers), iLayer+1):
                if i == 0:
                    opacity = 255  # "bottom" layer is opaque (for rails)
                else:
                    opacity = 0   # subsequent layers are transparent
                img = self.create_image(opacity)
                self.layers.append(img)
                self.draws.append(ImageDraw.Draw(img))
        else:
            # We do already have this iLayer.  Clear it if requested.
            if clear:
                opacity = 0 if iLayer > 0 else 255
                self.layers[iLayer] = img = self.create_image(opacity)
                # We also need to maintain a Draw object for each layer
                self.draws[iLayer] = ImageDraw.Draw(img)

    def create_layers(self, clear=True):        
        self.create_layer(0, clear=clear)
        self.create_layer(1, clear=clear)


class PILSVG(PILGL):
    def __init__(self, width, height,jupyter=False):
        print(self, type(self))
        oSuper = super()
        print(oSuper, type(oSuper))
        oSuper.__init__(width, height,jupyter)

        # self.track = self.track = Track()
        # self.lwTrack = []
        # self.zug = Zug()

        self.lwAgents = []
        self.agents_prev = []

        self.loadRailSVGs()
        self.loadAgentSVGs()

    def is_raster(self):
        return False

    def processEvents(self):
        # self.app.processEvents()
        time.sleep(0.001)

    def clear_rails(self):
        print("Clear rails")
        self.create_layers()
        self.clear_agents()

    def clear_agents(self):
        # print("Clear Agents: ", len(self.lwAgents))
        for wAgent in self.lwAgents:
            self.layout.removeWidget(wAgent)
        self.lwAgents = []
        self.agents_prev = []

    def pilFromSvgFile(self, sfPath):
        try:
            with open(sfPath, "r") as fIn:
                bytesPNG = svg2png(file_obj=fIn, output_height=self.nPixCell, output_width=self.nPixCell)
        except:
            newList=''
            for directory in site.getsitepackages():
                x = [word for word in os.listdir(directory) if word.startswith('flatland')]
                if len(x) > 0 :
                    newList = directory+'/'+x[0]
            with open(newList+'/'+sfPath, "r") as fIn:
                bytesPNG = svg2png(file_obj=fIn, output_height=self.nPixCell, output_width=self.nPixCell)
        with io.BytesIO(bytesPNG) as fIn:
            pil_img = Image.open(fIn)
            pil_img.load()
            # print(pil_img.mode)
        
        return pil_img

    def pilFromSvgBytes(self, bytesSVG):
        bytesPNG = svg2png(bytesSVG, output_height=self.nPixCell, output_width=self.nPixCell)
        with io.BytesIO(bytesPNG) as fIn:
            pil_img = Image.open(fIn)
            return pil_img

    def loadRailSVGs(self):
        """ Load the rail SVG images, apply rotations, and store as PIL images.
        """
        dRailFiles = {
            "": "Background_#91D1DD.svg",
            "WE": "Gleis_Deadend.svg",
            "WW EE NN SS": "Gleis_Diamond_Crossing.svg",
            "WW EE": "Gleis_horizontal.svg",
            "EN SW": "Gleis_Kurve_oben_links.svg",
            "WN SE": "Gleis_Kurve_oben_rechts.svg",
            "ES NW": "Gleis_Kurve_unten_links.svg",
            "NE WS": "Gleis_Kurve_unten_rechts.svg",
            "NN SS": "Gleis_vertikal.svg",
            "NN SS EE WW ES NW SE WN": "Weiche_Double_Slip.svg",
            "EE WW EN SW": "Weiche_horizontal_oben_links.svg",
            "EE WW SE WN": "Weiche_horizontal_oben_rechts.svg",
            "EE WW ES NW": "Weiche_horizontal_unten_links.svg",
            "EE WW NE WS": "Weiche_horizontal_unten_rechts.svg",
            "NN SS EE WW NW ES": "Weiche_Single_Slip.svg",
            "NE NW ES WS": "Weiche_Symetrical.svg",
            "NN SS EN SW": "Weiche_vertikal_oben_links.svg",
            "NN SS SE WN": "Weiche_vertikal_oben_rechts.svg",
            "NN SS NW ES": "Weiche_vertikal_unten_links.svg",
            "NN SS NE WS": "Weiche_vertikal_unten_rechts.svg"}

        dTargetFiles = {
            "EW": "Bahnhof_#d50000_Deadend_links.svg",
            "NS": "Bahnhof_#d50000_Deadend_oben.svg",
            "WE": "Bahnhof_#d50000_Deadend_rechts.svg",
            "SN": "Bahnhof_#d50000_Deadend_unten.svg",
            "EE WW": "Bahnhof_#d50000_Gleis_horizontal.svg",
            "NN SS": "Bahnhof_#d50000_Gleis_vertikal.svg"}

        # Dict of rail cell images indexed by binary transitions
        self.dPilRail = self.loadSVGs(dRailFiles, rotate=True)

        # Load the target files (which have rails and transitions of their own)
        # They are indexed by (binTrans, iAgent), ie a tuple of the binary transition and the agent index
        dPilRail2 = self.loadSVGs(dTargetFiles, rotate=False, agent_colors=self.ltAgentColors)
        # Merge them with the regular rails.
        # https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression
        self.dPilRail = {**self.dPilRail, **dPilRail2}
        
    def loadSVGs(self, dDirFile, rotate=False, agent_colors=False):
        dPil = {}

        transitions = RailEnvTransitions()

        lDirs = list("NESW")

        # svgBG = SVG("./svg/Background_#91D1DD.svg")

        for sTrans, sFile in dDirFile.items():
            sPathSvg = "./svg/" + sFile

            # Translate the ascii transition description in the format  "NE WS" to the 
            # binary list of transitions as per RailEnv - NESW (in) x NESW (out)
            lTrans16 = ["0"] * 16
            for sTran in sTrans.split(" "):
                if len(sTran) == 2:
                    iDirIn = lDirs.index(sTran[0])
                    iDirOut = lDirs.index(sTran[1])
                    iTrans = 4 * iDirIn + iDirOut
                    lTrans16[iTrans] = "1"
            sTrans16 = "".join(lTrans16)
            binTrans = int(sTrans16, 2)
            # print(sTrans, sTrans16, sFile)

            # Merge the transition svg image with the background colour.
            # This is a shortcut / hack and will need re-working.
            # if binTrans > 0:
            #    svg = svg.merge(svgBG)

            pilRail = self.pilFromSvgFile(sPathSvg)
            
            if rotate:
                # For rotations, we also store the base image
                dPil[binTrans] = pilRail
                # Rotate both the transition binary and the image and save in the dict
                for nRot in [90, 180, 270]:
                    binTrans2 = transitions.rotate_transition(binTrans, nRot)

                    # PIL rotates anticlockwise for positive theta
                    pilRail2 = pilRail.rotate(-nRot)
                    dPil[binTrans2] = pilRail2
            
            if agent_colors:
                # For recoloring, we don't store the base image.
                a3BaseColor = self.rgb_s2i("d50000")
                lPils = self.recolorImage(pilRail, a3BaseColor, self.ltAgentColors)
                for iColor, pilRail2 in enumerate(lPils):
                    dPil[(binTrans, iColor)] = lPils[iColor]

        return dPil

    def setRailAt(self, row, col, binTrans, iTarget=None):
        if iTarget is None:
            if binTrans in self.dPilRail:
                pilTrack = self.dPilRail[binTrans]
                self.drawImageRC(pilTrack, (row, col))
            else:
                print("Illegal rail:", row, col, format(binTrans, "#018b")[2:])
        else:
            if (binTrans, iTarget) in self.dPilRail:
                pilTrack = self.dPilRail[(binTrans, iTarget)]
                self.drawImageRC(pilTrack, (row, col))
            else:
                print("Illegal target rail:", row, col, format(binTrans, "#018b")[2:])

    def recolorImage(self, pil, a3BaseColor, ltColors):
        rgbaImg = array(pil)
        lPils = []

        for iColor, tnColor in enumerate(ltColors):
            # find the pixels which match the base paint color
            xy_color_mask = np.all(rgbaImg[:, :, 0:3] - a3BaseColor == 0, axis=2)
            rgbaImg2 = np.copy(rgbaImg)

            # Repaint the base color with the new color
            rgbaImg2[xy_color_mask, 0:3] = tnColor
            pil2 = Image.fromarray(rgbaImg2)
            lPils.append(pil2)
        return lPils

    def loadAgentSVGs(self):

        # Seed initial train/zug files indexed by tuple(iDirIn, iDirOut):
        dDirsFile = {
            (0, 0): "svg/Zug_Gleis_#0091ea.svg",
            (1, 2): "svg/Zug_1_Weiche_#0091ea.svg",
            (0, 3): "svg/Zug_2_Weiche_#0091ea.svg"
            }

        # "paint" color of the train images we load
        a3BaseColor = self.rgb_s2i("0091ea")

        self.dPilZug = {}

        for tDirs, sPathSvg in dDirsFile.items():
            iDirIn, iDirOut = tDirs
            
            pilZug = self.pilFromSvgFile(sPathSvg)

            # Rotate both the directions and the image and save in the dict
            for iDirRot in range(4):
                nDegRot = iDirRot * 90
                iDirIn2 = (iDirIn + iDirRot) % 4
                iDirOut2 = (iDirOut + iDirRot) % 4

                # PIL rotates anticlockwise for positive theta
                pilZug2 = pilZug.rotate(-nDegRot)

                # Save colored versions of each rotation / variant
                lPils = self.recolorImage(pilZug2, a3BaseColor, self.ltAgentColors)
                for iColor, pilZug3 in enumerate(lPils):
                    self.dPilZug[(iDirIn2, iDirOut2, iColor)] = lPils[iColor]

    def setAgentAt(self, iAgent, row, col, iDirIn, iDirOut):
        delta_dir = (iDirOut - iDirIn) % 4
        iColor = iAgent % self.nAgentColors
        # when flipping direction at a dead end, use the "iDirOut" direction.
        if delta_dir == 2:
            iDirIn = iDirOut
        pilZug = self.dPilZug[(iDirIn % 4, iDirOut % 4, iColor)]
        self.drawImageRC(pilZug, (row, col), layer=1)


def main2():
    gl = PILSVG(10, 10)
    for i in range(10):
        gl.beginFrame()
        gl.plot([3 + i, 4], [-4 - i, -5], color="r")
        gl.endFrame()
        time.sleep(1)


def main():
    gl = PILSVG(width=10, height=10)

    for i in range(1000):
        gl.processEvents()
        time.sleep(0.1)
    time.sleep(1)


if __name__ == "__main__":
    main()

