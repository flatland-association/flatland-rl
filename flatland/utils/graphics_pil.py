import io
import os
import time
#import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from numpy import array
from pkg_resources import resource_string as resource_bytes

from flatland.utils.graphics_layer import GraphicsLayer

from flatland.core.grid.rail_env_grid import RailEnvTransitions  # noqa: E402


class PILGL(GraphicsLayer):
    # tk.Tk() must be a singleton!
    # https://stackoverflow.com/questions/26097811/image-pyimage2-doesnt-exist
    # window = tk.Tk()

    RAIL_LAYER = 0
    PREDICTION_PATH_LAYER = 1
    TARGET_LAYER = 2
    AGENT_LAYER = 3
    SELECTED_AGENT_LAYER = 4
    SELECTED_TARGET_LAYER = 5

    def __init__(self, width, height, jupyter=False, screen_width=800, screen_height=600):
        self.yxBase = (0, 0)
        self.linewidth = 4
        self.n_agent_colors = 1  # overridden in loadAgent

        self.width = width
        self.height = height

        self.background_grid = np.zeros(shape=(self.width, self.height))

        if jupyter is False:
            # NOTE: Currently removed the dependency on
            #       screeninfo. We have to find an alternate
            #       way to compute the screen width and height
            #       In the meantime, we are harcoding the 800x600
            #       assumption
            self.screen_width = screen_width
            self.screen_height = screen_height
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
        self.agent_colors = [self.rgb_s2i(sColor) for sColor in sColors.split("#")]
        self.n_agent_colors = len(self.agent_colors)

        self.firstFrame = True
        self.old_background_image = (None, None, None)
        self.create_layers()

        self.font = ImageFont.load_default()

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
            # rebuild background_grid to control the visualisation of buildings, trees, mountains, lakes and river
            self.background_grid = np.zeros(shape=(self.width, self.height))

            # build base distance map (distance to targets)
            for x in range(self.width):
                for y in range(self.height):
                    distance = int(np.ceil(np.sqrt(self.width ** 2.0 + self.height ** 2.0)))
                    for rc in dTargets:
                        r = rc[1]
                        c = rc[0]
                        d = int(np.floor(np.sqrt((x - r) ** 2 + (y - c) ** 2)) / 0.5)
                        distance = min(d, distance)
                    self.background_grid[x][y] = distance

            self.old_background_image = (dTargets, self.width, self.height)

    def rgb_s2i(self, sRGB):
        """ convert a hex RGB string like 0091ea to 3-tuple of ints """
        return tuple(int(sRGB[iRGB * 2:iRGB * 2 + 2], 16) for iRGB in [0, 1, 2])

    def get_agent_color(self, iAgent):
        return self.agent_colors[iAgent % self.n_agent_colors]

    def plot(self, gX, gY, color=None, linewidth=3, layer=RAIL_LAYER, opacity=255, **kwargs):
        """ Draw a line joining the points in gX, GY - each an"""
        color = self.adapt_color(color)
        if len(color) == 3:
            color += (opacity,)
        elif len(color) == 4:
            color = color[:3] + (opacity,)
        gPoints = np.stack([array(gX), -array(gY)]).T * self.nPixCell
        gPoints = list(gPoints.ravel())
        # the width here was self.linewidth - not really sure of the implications
        self.draws[layer].line(gPoints, fill=color, width=linewidth)

    def scatter(self, gX, gY, color=None, marker="o", s=50, layer=RAIL_LAYER, opacity=255, *args, **kwargs):
        color = self.adapt_color(color)
        r = np.sqrt(s)
        gPoints = np.stack([np.atleast_1d(gX), -np.atleast_1d(gY)]).T * self.nPixCell
        for x, y in gPoints:
            self.draws[layer].rectangle([(x - r, y - r), (x + r, y + r)], fill=color, outline=color)

    def draw_image_xy(self, pil_img, xyPixLeftTop, layer=RAIL_LAYER, ):

        # Resize all PIL images just before drawing them
        # to ensure that resizing doesnt affect the 
        # recolorizing strategies in place
        # 
        # That said : All the code in this file needs 
        # some serious refactoring -_- to ensure the 
        # code style and structure is consitent.
        #                               - Mohanty
        pil_img = pil_img.resize(
            (self.nPixCell, self.nPixCell)
        )

        if (pil_img.mode == "RGBA"):
            pil_mask = pil_img
        else:
            pil_mask = None
        
        self.layers[layer].paste(pil_img, xyPixLeftTop, pil_mask)

    def draw_image_row_col(self, pil_img, rcTopLeft, layer=RAIL_LAYER, ):
        xyPixLeftTop = tuple((array(rcTopLeft) * self.nPixCell)[[1, 0]])
        self.draw_image_xy(pil_img, xyPixLeftTop, layer=layer)

    def open_window(self):
        pass

    def close_window(self):
        pass

    def text(self, xPx, yPx, strText, layer=RAIL_LAYER):
        xyPixLeftTop = (xPx, yPx)
        self.draws[layer].text(xyPixLeftTop, strText, font=self.font, fill=(0, 0, 0, 255))

    def text_rowcol(self, rcTopLeft, strText, layer=AGENT_LAYER):
        xyPixLeftTop = tuple((array(rcTopLeft) * self.nPixCell)[[1, 0]])
        self.text(*xyPixLeftTop, strText, layer)

    def prettify(self, *args, **kwargs):
        pass

    def prettify2(self, width, height, cell_size):
        pass

    def begin_frame(self):
        # Create a new agent layer
        self.create_layer(iLayer=PILGL.AGENT_LAYER, clear=True)
        self.create_layer(iLayer=PILGL.PREDICTION_PATH_LAYER, clear=True)

    def show(self, block=False):
        #print("show() - ", self.__class__)
        pass

    def pause(self, seconds=0.00001):
        pass

    def idle(self, seconds=0.00001):
        pass

    def alpha_composite_layers(self):
        img = self.layers[0]
        for img2 in self.layers[1:]:
            img = Image.alpha_composite(img, img2)
        return img

    def get_image(self):
        """ return a blended / alpha composited image composed of all the layers,
            with layer 0 at the "back".
        """
        img = self.alpha_composite_layers()
        return array(img)

    def save_image(self, filename):
        """
        Renders the current scene into a image file
        :param filename: filename where to store the rendering output_generator
        (supported image format *.bmp , .. , *.png)
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
        self.create_layer(PILGL.RAIL_LAYER, clear=clear)  # rail / background (scene)
        self.create_layer(PILGL.AGENT_LAYER, clear=clear)  # agents
        self.create_layer(PILGL.TARGET_LAYER, clear=clear)  # agents
        self.create_layer(PILGL.PREDICTION_PATH_LAYER, clear=clear)  # drawing layer for agent's prediction path
        self.create_layer(PILGL.SELECTED_AGENT_LAYER, clear=clear)  # drawing layer for selected agent
        self.create_layer(PILGL.SELECTED_TARGET_LAYER, clear=clear)  # drawing layer for selected agent's target


class PILSVG(PILGL):
    """
    Note : This class should now ideally be called as PILPNG,
    but for backward compatibility, and to not introduce any breaking changes at this point
    we are sticking to the legacy name of PILSVG (when in practice we are not using SVG anymore)
    """
    def __init__(self, width, height, jupyter=False, screen_width=800, screen_height=600):
        oSuper = super()
        oSuper.__init__(width, height, jupyter, screen_width, screen_height)

        self.lwAgents = []
        self.agents_prev = []

        self.load_buildings()
        self.load_scenery()
        self.load_rail()
        self.load_agent()

    def process_events(self):
        time.sleep(0.001)

    def clear_rails(self):
        self.create_layers()
        self.clear_agents()

    def clear_agents(self):
        for wAgent in self.lwAgents:
            self.layout.removeWidget(wAgent)
        self.lwAgents = []
        self.agents_prev = []

    def pil_from_png_file(self, package, resource):
        bytestring = resource_bytes(package, resource)
        with io.BytesIO(bytestring) as fIn:
            pil_img = Image.open(fIn)
            pil_img.load()
        return pil_img

    def load_buildings(self):
        lBuildingFiles = [
            "Buildings-Bank.png",
            "Buildings-Bar.png",
            "Buildings-Wohnhaus.png",
            "Buildings-Hochhaus.png",
            "Buildings-Hotel.png",
            "Buildings-Office.png",
            "Buildings-Polizei.png",
            "Buildings-Post.png",
            "Buildings-Supermarkt.png",
            "Buildings-Tankstelle.png",
            "Buildings-Fabrik_A.png",
            "Buildings-Fabrik_B.png",
            "Buildings-Fabrik_C.png",
            "Buildings-Fabrik_D.png",
            "Buildings-Fabrik_E.png",
            "Buildings-Fabrik_F.png",
            "Buildings-Fabrik_G.png",
            "Buildings-Fabrik_H.png",
            "Buildings-Fabrik_I.png"
        ]

        imgBg = self.pil_from_png_file('flatland.png', "Background_city.png")
        imgBg = imgBg.convert("RGBA")

        self.lBuildings = []
        for sFile in lBuildingFiles:
            img = self.pil_from_png_file('flatland.png', sFile)
            img = Image.alpha_composite(imgBg, img)
            self.lBuildings.append(img)

    def load_scenery(self):
        scenery_files = [
            "Scenery-Laubbaume_A.png",
            "Scenery-Laubbaume_B.png",
            "Scenery-Laubbaume_C.png",
            "Scenery-Nadelbaume_A.png",
            "Scenery-Nadelbaume_B.png",
            "Scenery-Bergwelt_B.png"
        ]

        scenery_files_d2 = [
            "Scenery-Bergwelt_C_Teil_1_links.png",
            "Scenery-Bergwelt_C_Teil_2_rechts.png"
        ]

        scenery_files_d3 = [
            "Scenery-Bergwelt_A_Teil_1_links.png",
            "Scenery-Bergwelt_A_Teil_2_mitte.png",
            "Scenery-Bergwelt_A_Teil_3_rechts.png"
        ]

        scenery_files_water = [
            "Scenery_Water.png"
        ]

        img_back_ground = self.pil_from_png_file('flatland.png', "Background_Light_green.png").convert("RGBA")

        self.scenery_background_white = self.pil_from_png_file('flatland.png', "Background_white.png").convert("RGBA")

        self.scenery = []
        for file in scenery_files:
            img = self.pil_from_png_file('flatland.png', file)
            img = Image.alpha_composite(img_back_ground, img)
            self.scenery.append(img)

        self.scenery_d2 = []
        for file in scenery_files_d2:
            img = self.pil_from_png_file('flatland.png', file)
            img = Image.alpha_composite(img_back_ground, img)
            self.scenery_d2.append(img)

        self.scenery_d3 = []
        for file in scenery_files_d3:
            img = self.pil_from_png_file('flatland.png', file)
            img = Image.alpha_composite(img_back_ground, img)
            self.scenery_d3.append(img)

        self.scenery_water = []
        for file in scenery_files_water:
            img = self.pil_from_png_file('flatland.png', file)
            img = Image.alpha_composite(img_back_ground, img)
            self.scenery_water.append(img)

    def load_rail(self):
        """ Load the rail SVG images, apply rotations, and store as PIL images.
        """
        rail_files = {
            "": "Background_Light_green.png",
            "WE": "Gleis_Deadend.png",
            "WW EE NN SS": "Gleis_Diamond_Crossing.png",
            "WW EE": "Gleis_horizontal.png",
            "EN SW": "Gleis_Kurve_oben_links.png",
            "WN SE": "Gleis_Kurve_oben_rechts.png",
            "ES NW": "Gleis_Kurve_unten_links.png",
            "NE WS": "Gleis_Kurve_unten_rechts.png",
            "NN SS": "Gleis_vertikal.png",
            "NN SS EE WW ES NW SE WN": "Weiche_Double_Slip.png",
            "EE WW EN SW": "Weiche_horizontal_oben_links.png",
            "EE WW SE WN": "Weiche_horizontal_oben_rechts.png",
            "EE WW ES NW": "Weiche_horizontal_unten_links.png",
            "EE WW NE WS": "Weiche_horizontal_unten_rechts.png",
            "NN SS EE WW NW ES": "Weiche_Single_Slip.png",
            "NE NW ES WS": "Weiche_Symetrical.png",
            "NN SS EN SW": "Weiche_vertikal_oben_links.png",
            "NN SS SE WN": "Weiche_vertikal_oben_rechts.png",
            "NN SS NW ES": "Weiche_vertikal_unten_links.png",
            "NN SS NE WS": "Weiche_vertikal_unten_rechts.png",
            "NE NW ES WS SS NN": "Weiche_Symetrical_gerade.png",
            "NE EN SW WS": "Gleis_Kurve_oben_links_unten_rechts.png"
        }

        target_files = {
            "EW": "Bahnhof_#d50000_Deadend_links.png",
            "NS": "Bahnhof_#d50000_Deadend_oben.png",
            "WE": "Bahnhof_#d50000_Deadend_rechts.png",
            "SN": "Bahnhof_#d50000_Deadend_unten.png",
            "EE WW": "Bahnhof_#d50000_Gleis_horizontal.png",
            "NN SS": "Bahnhof_#d50000_Gleis_vertikal.png"}

        # Dict of rail cell images indexed by binary transitions
        pil_rail_files_org = self.load_pngs(rail_files, rotate=True)
        pil_rail_files = self.load_pngs(rail_files, rotate=True, background_image="Background_rail.png",
                                        whitefilter="Background_white_filter.png")

        # Load the target files (which have rails and transitions of their own)
        # They are indexed by (binTrans, iAgent), ie a tuple of the binary transition and the agent index
        pil_target_files_org = self.load_pngs(target_files, rotate=False, agent_colors=self.agent_colors)
        pil_target_files = self.load_pngs(target_files, rotate=False, agent_colors=self.agent_colors,
                                          background_image="Background_rail.png",
                                          whitefilter="Background_white_filter.png")

        # Load station and recolorize them
        station = self.pil_from_png_file('flatland.png', "Bahnhof_#d50000_target.png")
        self.station_colors = self.recolor_image(station, [0, 0, 0], self.agent_colors, False)

        cell_occupied = self.pil_from_png_file('flatland.png', "Cell_occupied.png")
        self.cell_occupied = self.recolor_image(cell_occupied, [0, 0, 0], self.agent_colors, False)

        # Merge them with the regular rails.
        # https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression
        self.pil_rail = {**pil_rail_files, **pil_target_files}
        self.pil_rail_org = {**pil_rail_files_org, **pil_target_files_org}

    def load_pngs(self, file_directory, rotate=False, agent_colors=False, background_image=None, whitefilter=None):
        pil = {}

        transitions = RailEnvTransitions()

        directions = list("NESW")

        for transition, file in file_directory.items():

            # Translate the ascii transition description in the format  "NE WS" to the
            # binary list of transitions as per RailEnv - NESW (in) x NESW (out)
            transition_16_bit = ["0"] * 16
            for sTran in transition.split(" "):
                if len(sTran) == 2:
                    in_direction = directions.index(sTran[0])
                    out_direction = directions.index(sTran[1])
                    transition_idx = 4 * in_direction + out_direction
                    transition_16_bit[transition_idx] = "1"
            transition_16_bit_string = "".join(transition_16_bit)
            binary_trans = int(transition_16_bit_string, 2)

            pil_rail = self.pil_from_png_file('flatland.png', file).convert("RGBA")

            if background_image is not None:
                img_bg = self.pil_from_png_file('flatland.png', background_image).convert("RGBA")
                pil_rail = Image.alpha_composite(img_bg, pil_rail)

            if whitefilter is not None:
                img_bg = self.pil_from_png_file('flatland.png', whitefilter).convert("RGBA")
                pil_rail = Image.alpha_composite(pil_rail, img_bg)

            if rotate:
                # For rotations, we also store the base image
                pil[binary_trans] = pil_rail
                # Rotate both the transition binary and the image and save in the dict
                for nRot in [90, 180, 270]:
                    binary_trans_2 = transitions.rotate_transition(binary_trans, nRot)

                    # PIL rotates anticlockwise for positive theta
                    pil_rail_2 = pil_rail.rotate(-nRot)
                    pil[binary_trans_2] = pil_rail_2

            if agent_colors:
                # For recoloring, we don't store the base image.
                base_color = self.rgb_s2i("d50000")
                pils = self.recolor_image(pil_rail, base_color, self.agent_colors)
                for color_idx, pil_rail_2 in enumerate(pils):
                    pil[(binary_trans, color_idx)] = pils[color_idx]

        return pil

    def set_predicion_path_at(self, row, col, binary_trans, agent_rail_color):
        colored_rail = self.recolor_image(self.pil_rail_org[binary_trans],
                                          [61, 61, 61], [agent_rail_color],
                                          False)[0]
        self.draw_image_row_col(colored_rail, (row, col), layer=PILGL.PREDICTION_PATH_LAYER)

    def set_rail_at(self, row, col, binary_trans, target=None, is_selected=False, rail_grid=None, num_agents=None,
                    show_debug=True):

        if binary_trans in self.pil_rail:
            pil_track = self.pil_rail[binary_trans]
            if target is not None:
                target_img = self.station_colors[target % len(self.station_colors)]
                target_img = Image.alpha_composite(pil_track, target_img)
                self.draw_image_row_col(target_img, (row, col), layer=PILGL.TARGET_LAYER)
                if show_debug:
                    self.text_rowcol((row + 0.8, col + 0.0), strText=str(target), layer=PILGL.TARGET_LAYER)

            city_size = 1
            if num_agents is not None:
                city_size = max(1, np.log(1 + num_agents) / 2.5)

            if binary_trans == 0:
                if self.background_grid[col][row] <= 4 + np.ceil(((col * row + col) % 10) / city_size):
                    a = int(self.background_grid[col][row])
                    a = a % len(self.lBuildings)
                    if (col + row + col * row) % 13 > 11:
                        pil_track = self.scenery[a % len(self.scenery)]
                    else:
                        if (col + row + col * row) % 3 == 0:
                            a = (a + (col + row + col * row)) % len(self.lBuildings)
                        pil_track = self.lBuildings[a]
                elif ((self.background_grid[col][row] > 5 + ((col * row + col) % 3)) or
                      ((col ** 3 + row ** 2 + col * row) % 10 == 0)):
                    a = int(self.background_grid[col][row]) - 4
                    a2 = (a + (col + row + col * row + col ** 3 + row ** 4))
                    if a2 % 64 > 11:
                        a = a2
                    a_l = a % len(self.scenery)
                    if a2 % 50 == 49:
                        pil_track = self.scenery_water[0]
                    else:
                        pil_track = self.scenery[a_l]
                    if rail_grid is not None:
                        if a2 % 11 > 3:
                            if a_l == len(self.scenery) - 1:
                                # mountain
                                if col > 1 and row % 7 == 1:
                                    if rail_grid[row, col - 1] == 0:
                                        self.draw_image_row_col(self.scenery_d2[0], (row, col - 1),
                                                                layer=PILGL.RAIL_LAYER)
                                        pil_track = self.scenery_d2[1]
                        else:
                            if a_l == len(self.scenery) - 1:
                                # mountain
                                if col > 2 and not (row % 7 == 1):
                                    if rail_grid[row, col - 2] == 0 and rail_grid[row, col - 1] == 0:
                                        self.draw_image_row_col(self.scenery_d3[0], (row, col - 2),
                                                                layer=PILGL.RAIL_LAYER)
                                        self.draw_image_row_col(self.scenery_d3[1], (row, col - 1),
                                                                layer=PILGL.RAIL_LAYER)
                                        pil_track = self.scenery_d3[2]

            self.draw_image_row_col(pil_track, (row, col), layer=PILGL.RAIL_LAYER)
        else:
            print("Illegal rail:", row, col, format(binary_trans, "#018b")[2:], binary_trans)

        if target is not None:
            if is_selected:
                svgBG = self.pil_from_png_file('flatland.png', "Selected_Target.png")
                self.clear_layer(PILGL.SELECTED_TARGET_LAYER, 0)
                self.draw_image_row_col(svgBG, (row, col), layer=PILGL.SELECTED_TARGET_LAYER)

    def recolor_image(self, pil, a3BaseColor, ltColors, invert=False):
        rgbaImg = array(pil)
        pils = []
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
            pils.append(pil2)
        return pils

    def load_agent(self):

        # Seed initial train/zug files indexed by tuple(iDirIn, iDirOut):
        file_directory = {
            (0, 0): "Zug_Gleis_#0091ea.png",
            (1, 2): "Zug_1_Weiche_#0091ea.png",
            (0, 3): "Zug_2_Weiche_#0091ea.png"
        }

        # "paint" color of the train images we load - this is the color we will change.
        # base_color = self.rgb_s2i("0091ea") \#  noqa: E800
        # temporary workaround for trains / agents renamed with different colour:
        base_color = self.rgb_s2i("d50000")

        self.pil_zug = {}

        for directions, path_svg in file_directory.items():
            in_direction, out_direction = directions

            pil_zug = self.pil_from_png_file('flatland.png', path_svg)

            # Rotate both the directions and the image and save in the dict
            for rot_direction in range(4):
                rotation_degree = rot_direction * 90
                in_direction_2 = (in_direction + rot_direction) % 4
                out_direction_2 = (out_direction + rot_direction) % 4

                # PIL rotates anticlockwise for positive theta
                pil_zug_2 = pil_zug.rotate(-rotation_degree)

                # Save colored versions of each rotation / variant
                pils = self.recolor_image(pil_zug_2, base_color, self.agent_colors)
                for color_idx, pil_zug_3 in enumerate(pils):
                    self.pil_zug[(in_direction_2, out_direction_2, color_idx)] = pils[color_idx]

    def set_agent_at(self, agent_idx, row, col, in_direction, out_direction, is_selected,
                     rail_grid=None, show_debug=False, clear_debug_text=True, malfunction=False):
        delta_dir = (out_direction - in_direction) % 4
        color_idx = agent_idx % self.n_agent_colors
        # when flipping direction at a dead end, use the "out_direction" direction.
        if delta_dir == 2:
            in_direction = out_direction
        pil_zug = self.pil_zug[(in_direction % 4, out_direction % 4, color_idx)]
        self.draw_image_row_col(pil_zug, (row, col), layer=PILGL.AGENT_LAYER)
        if rail_grid is not None:
            if rail_grid[row, col] == 0.0:
                self.draw_image_row_col(self.scenery_background_white, (row, col), layer=PILGL.RAIL_LAYER)

        if is_selected:
            bg_svg = self.pil_from_png_file('flatland.png', "Selected_Agent.png")
            self.clear_layer(PILGL.SELECTED_AGENT_LAYER, 0)
            self.draw_image_row_col(bg_svg, (row, col), layer=PILGL.SELECTED_AGENT_LAYER)
        if show_debug:
            if not clear_debug_text:
                dr = 0.2
                dc = 0.2
                if in_direction == 0:
                    dr = 0.8
                    dc = 0.0
                if in_direction == 1:
                    dr = 0.0
                    dc = 0.8
                if in_direction == 2:
                    dr = 0.4
                    dc = 0.8
                if in_direction == 3:
                    dr = 0.8
                    dc = 0.4

                self.text_rowcol((row + dr, col + dc,), str(agent_idx), layer=PILGL.SELECTED_AGENT_LAYER)
            else:
                self.text_rowcol((row + 0.2, col + 0.2,), str(agent_idx))
        if malfunction:
            self.draw_malfunction(agent_idx, (row, col))

    def set_cell_occupied(self, agent_idx, row, col):
        occupied_im = self.cell_occupied[agent_idx % len(self.cell_occupied)]
        self.draw_image_row_col(occupied_im, (row, col), 1)

    def draw_malfunction(self, agent_idx, rcTopLeft):
        # Roughly an "X" shape to indicate malfunction
        grcOffsets = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
        grcPoints = np.array(rcTopLeft)[None] + grcOffsets
        gxyPoints = grcPoints[:, [1, 0]]
        gxPoints, gyPoints = gxyPoints.T
        # print(agent_idx, rcTopLeft, gxyPoints, "X:", gxPoints, "Y:", gyPoints)
        # plot(self, gX, gY, color=None, linewidth=3, layer=RAIL_LAYER, opacity=255, **kwargs):
        self.plot(gxPoints, -gyPoints, color=(0, 0, 0, 255), layer=PILGL.AGENT_LAYER, linewidth=2)


def main2():
    gl = PILSVG(10, 10)
    for i in range(10):
        gl.begin_frame()
        gl.plot([3 + i, 4], [-4 - i, -5], color="r")
        gl.endFrame()
        time.sleep(1)


def main():
    gl = PILSVG(width=10, height=10)

    for i in range(1000):
        gl.process_events()
        time.sleep(0.1)
    time.sleep(1)


if __name__ == "__main__":
    main()
