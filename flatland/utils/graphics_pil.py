import io
import os
import platform
import time
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageTk  # , ImageFont
from numpy import array
from pkg_resources import resource_string as resource_bytes

from flatland.utils.graphics_layer import GraphicsLayer


def enable_windows_cairo_support():
    if os.name == 'nt':
        import site
        import ctypes.util
        default_os_path = os.environ['PATH']
        os.environ['PATH'] = ''
        for s in site.getsitepackages():
            os.environ['PATH'] = os.environ['PATH'] + ';' + s + '\\cairo'
        os.environ['PATH'] = os.environ['PATH'] + ';' + default_os_path
        if ctypes.util.find_library('cairo') is None:
            print("Error: cairo not installed")


enable_windows_cairo_support()
from cairosvg import svg2png  # noqa: E402
from screeninfo import get_monitors  # noqa: E402

from flatland.core.grid.rail_env_grid import RailEnvTransitions  # noqa: E402


class PILGL(GraphicsLayer):
    # tk.Tk() must be a singleton!
    # https://stackoverflow.com/questions/26097811/image-pyimage2-doesnt-exist
    window = tk.Tk()

    def __init__(self, width, height, jupyter=False):
        self.yxBase = (0, 0)
        self.linewidth = 4
        self.nAgentColors = 1  # overridden in loadAgent

        self.width = width
        self.height = height

        self.background_grid = np.zeros(shape=(self.width, self.height))

        if jupyter is False:
            self.screen_width = 800
            self.screen_height = 600

            if platform.system() == "Windows" or platform.system() == "Linux":
                self.screen_width = 9999
                self.screen_height = 9999
                for m in get_monitors():
                    self.screen_height = min(self.screen_height, m.height)
                    self.screen_width = min(self.screen_width, m.width)
                # Note: screeninfo doesnot have proper support for 
                # OSX yet, hence the default values of 800,600
                # will be used for the same.

            w = (self.screen_width - self.width - 10) / (self.width + 1 + self.linewidth)
            h = (self.screen_height - self.height - 10) / (self.height + 1 + self.linewidth)
            self.nPixCell = int(max(1, np.ceil(min(w, h))))
        else:
            self.nPixCell = 40

        # Total grid size at native scale
        self.widthPx = self.width * self.nPixCell + self.linewidth
        self.heightPx = self.height * self.nPixCell + self.linewidth

        self.xPx = int((self.screen_width - self.widthPx) / 2.0)
        self.yPx = int((self.screen_height - self.heightPx) / 2.0)

        self.layers = []
        self.draws = []

        self.tColBg = (255, 255, 255)  # white background
        self.tColRail = (0, 0, 0)  # black rails
        self.tColGrid = (230,) * 3  # light grey for grid

        sColors = "d50000#c51162#aa00ff#6200ea#304ffe#2962ff#0091ea#00b8d4#00bfa5#00c853" + \
                  "#64dd17#aeea00#ffd600#ffab00#ff6d00#ff3d00#5d4037#455a64"

        self.ltAgentColors = [self.rgb_s2i(sColor) for sColor in sColors.split("#")]
        self.nAgentColors = len(self.ltAgentColors)

        self.window_open = False
        self.firstFrame = True
        self.old_background_image = (None, None, None)
        self.create_layers()

    def build_background_map(self, dTargets):
        x = self.old_background_image
        rebuild = False
        if x[0] is None:
            rebuild = True
        else:
            if len(x[0]) != len(dTargets):
                rebuild = True
            else:
                if x[0] != dTargets:
                    rebuild = True
                if x[1] != self.width:
                    rebuild = True
                if x[2] != self.height:
                    rebuild = True

        if rebuild:
            self.background_grid = np.zeros(shape=(self.width, self.height))
            for x in range(self.width):
                for y in range(self.height):
                    distance = int(np.ceil(np.sqrt(self.width ** 2.0 + self.height ** 2.0)))
                    for rc in dTargets:
                        r = rc[1]
                        c = rc[0]
                        d = int(np.floor(np.sqrt((x - r) ** 2 + (y - c) ** 2)))
                        distance = min(d, distance)
                    self.background_grid[x][y] = distance

            self.old_background_image = (dTargets, self.width, self.height)

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
        if (pil_img.mode == "RGBA"):
            pil_mask = pil_img
        else:
            pil_mask = None

        self.layers[layer].paste(pil_img, xyPixLeftTop, pil_mask)

    def drawImageRC(self, pil_img, rcTopLeft, layer=0):
        xyPixLeftTop = tuple((array(rcTopLeft) * self.nPixCell)[[1, 0]])
        self.drawImageXY(pil_img, xyPixLeftTop, layer=layer)

    def open_window(self):
        assert self.window_open is False, "Window is already open!"
        self.__class__.window.title("Flatland")
        self.__class__.window.configure(background='grey')
        self.window_open = True

    def close_window(self):
        self.panel.destroy()
        # quit but not destroy!
        self.__class__.window.quit()

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

        self.__class__.window.update()
        self.firstFrame = False

    def pause(self, seconds=0.00001):
        pass

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

    def saveImage(self, filename):
        """
        Renders the current scene into a image file
        :param filename: filename where to store the rendering output (supported image format *.bmp , .. , *.png)
        """
        img = self.alpha_composite_layers()
        img.save(filename)

    def create_image(self, opacity=255):
        img = Image.new("RGBA", (self.widthPx, self.heightPx), (255, 255, 255, opacity))
        return img

    def clear_layer(self, iLayer=0, opacity=None):
        if opacity is None:
            opacity = 0 if iLayer > 0 else 255
        self.layers[iLayer] = img = self.create_image(opacity)
        # We also need to maintain a Draw object for each layer
        self.draws[iLayer] = ImageDraw.Draw(img)

    def create_layer(self, iLayer=0, clear=True):
        # If we don't have the layers already, create them
        if len(self.layers) <= iLayer:
            for i in range(len(self.layers), iLayer + 1):
                if i == 0:
                    opacity = 255  # "bottom" layer is opaque (for rails)
                else:
                    opacity = 0  # subsequent layers are transparent
                img = self.create_image(opacity)
                self.layers.append(img)
                self.draws.append(ImageDraw.Draw(img))
        else:
            # We do already have this iLayer.  Clear it if requested.
            if clear:
                self.clear_layer(iLayer)

    def create_layers(self, clear=True):
        self.create_layer(0, clear=clear)  # rail / background (scene)
        self.create_layer(1, clear=clear)  # agents
        self.create_layer(2, clear=clear)  # drawing layer for selected agent
        self.create_layer(3, clear=clear)  # drawing layer for selected agent's target


class PILSVG(PILGL):
    def __init__(self, width, height, jupyter=False):
        oSuper = super()
        oSuper.__init__(width, height, jupyter)

        self.lwAgents = []
        self.agents_prev = []

        self.loadBuildingSVGs()
        self.loadScenerySVGs()
        self.loadRailSVGs()
        self.loadAgentSVGs()

    def is_raster(self):
        return False

    def processEvents(self):
        time.sleep(0.001)

    def clear_rails(self):
        self.create_layers()
        self.clear_agents()

    def clear_agents(self):
        for wAgent in self.lwAgents:
            self.layout.removeWidget(wAgent)
        self.lwAgents = []
        self.agents_prev = []

    def pilFromSvgFile(self, package, resource):
        bytestring = resource_bytes(package, resource)
        bytesPNG = svg2png(bytestring=bytestring, output_height=self.nPixCell, output_width=self.nPixCell)
        with io.BytesIO(bytesPNG) as fIn:
            pil_img = Image.open(fIn)
            pil_img.load()

        return pil_img

    def pilFromSvgBytes(self, bytesSVG):
        bytesPNG = svg2png(bytesSVG, output_height=self.nPixCell, output_width=self.nPixCell)
        with io.BytesIO(bytesPNG) as fIn:
            pil_img = Image.open(fIn)
            return pil_img

    def loadBuildingSVGs(self):
        dBuildingFiles = [
            "Buildings/Bank.svg",
            "Buildings/Bar.svg",
            "Buildings/Wohnhaus.svg",
            "Buildings/Hochhaus.svg",
            "Buildings/Hotel.svg",
            "Buildings/Office.svg",
            "Buildings/Polizei.svg",
            "Buildings/Post.svg",
            "Buildings/Supermarkt.svg",
            "Buildings/Tankstelle.svg",
            "Buildings/Fabrik_A.svg",
            "Buildings/Fabrik_B.svg",
            "Buildings/Fabrik_C.svg",
            "Buildings/Fabrik_D.svg",
            "Buildings/Fabrik_E.svg",
            "Buildings/Fabrik_F.svg",
            "Buildings/Fabrik_G.svg",
            "Buildings/Fabrik_H.svg",
            "Buildings/Fabrik_I.svg",
        ]

        imgBg = self.pilFromSvgFile('svg', "Background_city.svg")

        self.dBuildings = []
        for sFile in dBuildingFiles:
            img = self.pilFromSvgFile('svg', sFile)
            img = Image.alpha_composite(imgBg, img)
            self.dBuildings.append(img)

    def loadScenerySVGs(self):
        dSceneryFiles = [
            "Scenery/Laubbaume_A.svg",
            "Scenery/Laubbaume_B.svg",
            "Scenery/Laubbaume_C.svg",
            "Scenery/Nadelbaume_A.svg",
            "Scenery/Nadelbaume_B.svg",
            "Scenery/Bergwelt_B.svg"
        ]

        dSceneryFilesDim2 = [
            "Scenery/Bergwelt_C_Teil_1_links.svg",
            "Scenery/Bergwelt_C_Teil_2_rechts.svg"
        ]

        dSceneryFilesDim3 = [
            "Scenery/Bergwelt_A_Teil_3_rechts.svg",
            "Scenery/Bergwelt_A_Teil_2_mitte.svg",
            "Scenery/Bergwelt_A_Teil_1_links.svg"
        ]

        imgBg = self.pilFromSvgFile('svg', "Background_Light_green.svg")

        self.dScenery = []
        for sFile in dSceneryFiles:
            img = self.pilFromSvgFile('svg', sFile)
            img = Image.alpha_composite(imgBg, img)
            self.dScenery.append(img)

        self.dSceneryDim2 = []
        for sFile in dSceneryFilesDim2:
            img = self.pilFromSvgFile('svg', sFile)
            img = Image.alpha_composite(imgBg, img)
            self.dSceneryDim2.append(img)

        self.dSceneryDim3 = []
        for sFile in dSceneryFilesDim3:
            img = self.pilFromSvgFile('svg', sFile)
            img = Image.alpha_composite(imgBg, img)
            self.dSceneryDim3.append(img)

    def loadRailSVGs(self):
        """ Load the rail SVG images, apply rotations, and store as PIL images.
        """
        dRailFiles = {
            "": "Background_Light_green.svg",
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
            "NN SS NE WS": "Weiche_vertikal_unten_rechts.svg",
            "NE NW ES WS SS NN": "Weiche_Symetrical_gerade.svg",
            "NE EN SW WS": "Gleis_Kurve_oben_links_unten_rechts.svg"
        }

        dTargetFiles = {
            "EW": "Bahnhof_#d50000_Deadend_links.svg",
            "NS": "Bahnhof_#d50000_Deadend_oben.svg",
            "WE": "Bahnhof_#d50000_Deadend_rechts.svg",
            "SN": "Bahnhof_#d50000_Deadend_unten.svg",
            "EE WW": "Bahnhof_#d50000_Gleis_horizontal.svg",
            "NN SS": "Bahnhof_#d50000_Gleis_vertikal.svg"}

        # Dict of rail cell images indexed by binary transitions
        dPilRailFiles = self.loadSVGs(dRailFiles, rotate=True, backgroundImage="Background_rail.svg",
                                      whitefilter="Background_white_filter.svg")

        # Load the target files (which have rails and transitions of their own)
        # They are indexed by (binTrans, iAgent), ie a tuple of the binary transition and the agent index
        dPilTargetFiles = self.loadSVGs(dTargetFiles, rotate=False, agent_colors=self.ltAgentColors,
                                        backgroundImage="Background_rail.svg",
                                        whitefilter="Background_white_filter.svg")

        # Load station and recolorize them
        station = self.pilFromSvgFile("svg", "Bahnhof_#d50000_target.svg")
        self.ltStationColors = self.recolorImage(station, [0, 0, 0], self.ltAgentColors, False)

        cellOccupied = self.pilFromSvgFile("svg", "Cell_occupied.svg")
        self.ltCellOccupied = self.recolorImage(cellOccupied, [0, 0, 0], self.ltAgentColors, False)

        # Merge them with the regular rails.
        # https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression
        self.dPilRail = {**dPilRailFiles, **dPilTargetFiles}

    def loadSVGs(self, dDirFile, rotate=False, agent_colors=False, backgroundImage=None, whitefilter=None):
        dPil = {}

        transitions = RailEnvTransitions()

        lDirs = list("NESW")

        for sTrans, sFile in dDirFile.items():

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

            pilRail = self.pilFromSvgFile('svg', sFile)

            if backgroundImage is not None:
                imgBg = self.pilFromSvgFile('svg', backgroundImage)
                pilRail = Image.alpha_composite(imgBg, pilRail)

            if whitefilter is not None:
                imgBg = self.pilFromSvgFile('svg', whitefilter)
                pilRail = Image.alpha_composite(pilRail, imgBg)

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

    def setRailAt(self, row, col, binTrans, iTarget=None, isSelected=False, rail_grid=None):
        if binTrans in self.dPilRail:
            pilTrack = self.dPilRail[binTrans]
            if iTarget is not None:
                pilTrack = Image.alpha_composite(pilTrack, self.ltStationColors[iTarget % len(self.ltStationColors)])

            if binTrans == 0:
                if self.background_grid[col][row] <= 4:
                    a = int(self.background_grid[col][row])
                    a = a % len(self.dBuildings)
                    if (col + row + col * row) % 13 > 11:
                        pilTrack = self.dScenery[a % len(self.dScenery)]
                    else:
                        if (col + row + col * row) % 3 == 0:
                            a = (a + (col + row + col * row)) % len(self.dBuildings)
                        pilTrack = self.dBuildings[a]
                elif (self.background_grid[col][row] > 4) or ((col ** 3 + row ** 2 + col * row) % 10 == 0):
                    a = int(self.background_grid[col][row]) - 4
                    a2 = (a + (col + row + col * row + col ** 3 + row ** 4))
                    if a2 % 17 > 11:
                        a = a2
                    pilTrack = self.dScenery[a % len(self.dScenery)]

            self.drawImageRC(pilTrack, (row, col))
        else:
            print("Illegal rail:", row, col, format(binTrans, "#018b")[2:], binTrans)

        if iTarget is not None:
            if isSelected:
                svgBG = self.pilFromSvgFile("svg", "Selected_Target.svg")
                self.clear_layer(3, 0)
                self.drawImageRC(svgBG, (row, col), layer=3)

    def recolorImage(self, pil, a3BaseColor, ltColors, invert=False):
        rgbaImg = array(pil)
        lPils = []

        for iColor, tnColor in enumerate(ltColors):
            # find the pixels which match the base paint color
            if invert:
                xy_color_mask = np.all(rgbaImg[:, :, 0:3] - a3BaseColor != 0, axis=2)
            else:
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
            (0, 0): "Zug_Gleis_#0091ea.svg",
            (1, 2): "Zug_1_Weiche_#0091ea.svg",
            (0, 3): "Zug_2_Weiche_#0091ea.svg"
        }

        # "paint" color of the train images we load - this is the color we will change.
        # a3BaseColor = self.rgb_s2i("0091ea") \#  noqa: E800
        # temporary workaround for trains / agents renamed with different colour:
        a3BaseColor = self.rgb_s2i("d50000")

        self.dPilZug = {}

        for tDirs, sPathSvg in dDirsFile.items():
            iDirIn, iDirOut = tDirs

            pilZug = self.pilFromSvgFile("svg", sPathSvg)

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

    def setAgentAt(self, iAgent, row, col, iDirIn, iDirOut, isSelected):
        delta_dir = (iDirOut - iDirIn) % 4
        iColor = iAgent % self.nAgentColors
        # when flipping direction at a dead end, use the "iDirOut" direction.
        if delta_dir == 2:
            iDirIn = iDirOut
        pilZug = self.dPilZug[(iDirIn % 4, iDirOut % 4, iColor)]
        self.drawImageRC(pilZug, (row, col), layer=1)

        if isSelected:
            svgBG = self.pilFromSvgFile("svg", "Selected_Agent.svg")
            self.clear_layer(2, 0)
            self.drawImageRC(svgBG, (row, col), layer=2)

    def setCellOccupied(self, iAgent, row, col):
        occIm = self.ltCellOccupied[iAgent % len(self.ltCellOccupied)]
        self.drawImageRC(occIm, (row, col), 1)


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
