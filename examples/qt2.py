

import sys
from PyQt5 import QtSvg
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QGridLayout, QWidget
from PyQt5.QtCore import Qt, QByteArray

from flatland.utils import svg


# Subclass QMainWindow to customise your application's main window
class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("My Awesome App")

        layout = QGridLayout()
        layout.setSpacing(0)

        wMain = QWidget(self)

        wMain.setLayout(layout)

        label = QLabel("This is a PyQt5 window!")

        # The `Qt` namespace has a lot of attributes to customise
        # widgets. See: http://doc.qt.io/qt-5/qt.html
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label, 0, 0)

        svgWidget = QtSvg.QSvgWidget("./svg/Gleis_vertikal.svg")
        layout.addWidget(svgWidget, 1, 0)

        if True:
            track = svg.Track()

            svgWidget = None
            iRow = 0
            iCol = 2
            iArt = 0
            nCols = 3
            for binTrans in list(track.dSvg.keys())[:2]:
                sSVG = track.dSvg[binTrans].to_string()

                bySVG = bytearray(sSVG, encoding='utf-8')

                # with open(sfPath, "r") as fIn:
                #    sSVG = fIn.read()
                # bySVG = bytearray(sSVG, encoding='utf-8')

                svgWidget = QtSvg.QSvgWidget()
                oQB = QByteArray(bySVG)

                bSuccess = svgWidget.renderer().load(oQB)
                # print(x0, y0, x1, y1)
                print(iRow, iCol, bSuccess)
                print("\n\n\n", bySVG.decode("utf-8"))
                # svgWidget.setGeometry(x0, y0, x1, y1)
                layout.addWidget(svgWidget, iRow, iCol)

                iArt += 1
                iRow = int(iArt / nCols)
                iCol = iArt % nCols

        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        self.setCentralWidget(wMain)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()

