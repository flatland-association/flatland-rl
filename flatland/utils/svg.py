

import svgutils
import re
import copy
from flatland.core.transitions import RailEnvTransitions


class SVG(object):
    def __init__(self, sfName):
        self.svg = svgutils.transform.fromfile(sfName)
        self.init2()

    def init2(self):
        expr = "//*[local-name() = $name]"
        self.eStyle = self.svg.root.xpath(expr, name="style")[0]
        ltMatch = re.findall(r".st([a-zA-Z0-9]+)[{]([^}]*)}", self.eStyle.text)
        self.dStyles = dict(ltMatch)

    def copy(self):
        self2 = copy.deepcopy(self)
        self2.init2()
        return self2

    def merge(self, svg2):
        svg3 = svg2.copy()
        
        svg3.renumber_styles(offset=10)
        svg3.eStyle.text = self.eStyle.text + "\n" + svg3.eStyle.text
        
        for child in self.svg.root:
            if not child.tag.endswith("style"):
                svg3.svg.root.append(child)

        return svg3

    def renumber_styles(self, offset=10):
        sNewStyles = "\n"
        for sStyle in self.dStyles.keys():
            iStyle = int(sStyle)
            sClass = "st{:}".format(iStyle)
            lEl = self.svg.root.xpath("//*[@class='{}']".format(sClass))
            for el in lEl:
                el.attrib["class"] = "st{}".format(iStyle + offset)
        
            sStyle2 = str(iStyle+offset)

            sNewStyle = "\t.st"+sStyle2+"{"+self.dStyles[sStyle]+"}\n"
            sNewStyles += sNewStyle
        
        self.eStyle.text = sNewStyles

    def set_rotate(self, angle):
        self.svg.root.attrib["transform"] = "rotate({}, 120, 120)".format(angle)

    def to_string(self):
        return self.svg.to_str().decode("utf-8")


class Zug(object):
    def __init__(self, iDir=0):
        self.svg_straight = SVG("svg/Zug_Gleis_#0091ea.svg")
        self.svg_curve1 = SVG("svg/Zug_1_Weiche_#0091ea.svg")
        self.svg_curve2 = SVG("svg/Zug_2_Weiche_#0091ea.svg")

    def getSvg(self, iAgent, iDirIn, iDirOut):
        delta_dir = (iDirOut - iDirIn) % 4
        # if delta_dir != 0:
        #    print("Bend:", iAgent, iDirIn, iDirOut)

        if delta_dir in (0, 2):
            svg = self.svg_straight.copy()
            svg.set_rotate(iDirIn * 90)
            return svg
        
        if delta_dir == 1:  # bend to right, eg N->E, E->S
            svg = self.svg_curve1.copy()
            svg.set_rotate((iDirIn - 1) * 90)
            return svg

        elif delta_dir == 3:  # bend to left, eg N->W
            svg = self.svg_curve2.copy()
            svg.set_rotate(iDirIn * 90)
            return svg


class Track(object):
    def __init__(self):
        dFiles = {
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
            "NN SS NE WS": "Weiche_vertikal_unten_rechts.svg",
            }

        self.dSvg = {}

        transitions = RailEnvTransitions()

        lDirs = list("NESW")

        svgBG = SVG("./svg/Background_#91D1DD.svg")

        for sTrans, sFile in dFiles.items():
            
            svg = SVG("./svg/"+sFile)

            lTrans16 = ["0"] * 16
            for sTran in sTrans.split(" "):
                if len(sTran) == 2:
                    iDirIn = lDirs.index(sTran[0])
                    iDirOut = lDirs.index(sTran[1])
                    iTrans = 4 * iDirIn + iDirOut
                    lTrans16[iTrans] = "1"
            sTrans16 = "".join(lTrans16)
            binTrans = int(sTrans16, 2)
            print(sTrans, sTrans16, sFile)

            if binTrans > 0:
                svg = svg.merge(svgBG)

            self.dSvg[binTrans] = svg

            for nRot in [90, 180, 270]:
                binTrans2 = transitions.rotate_transition(binTrans, nRot)
                svg2 = svg.copy()
                svg2.set_rotate(nRot)
                self.dSvg[binTrans2] = svg2


def main():
    svg1 = SVG("./svg/Gleis_vertikal.svg")
    svg2 = SVG("./svg/Zug_1_Weiche_#0091ea.svg")
    svg3 = svg2.merge(svg1)
    svg3.set_rotate(90)

    s = svg3.to_string()
    print(s)

    track = Track()

    print(len(track.dSvg))


if __name__ == "__main__":
    main()

