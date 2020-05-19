
import tkinter as tk

from PIL import ImageTk
# from numpy import array
# from pkg_resources import resource_string as resource_bytes

# from flatland.utils.graphics_layer import GraphicsLayer
from flatland.utils.graphics_pil import PILSVG


class TKPILGL(PILSVG):
    # tk.Tk() must be a singleton!
    # https://stackoverflow.com/questions/26097811/image-pyimage2-doesnt-exist
    window = tk.Tk()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_open = False

    def open_window(self):
        print("open_window - tk")
        assert self.window_open is False, "Window is already open!"
        self.__class__.window.title("Flatland")
        self.__class__.window.configure(background='grey')
        self.window_open = True

    def close_window(self):
        self.panel.destroy()
        # quit but not destroy!
        self.__class__.window.quit()

    def show(self, block=False):
        # print("show - ", self.__class__)
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
