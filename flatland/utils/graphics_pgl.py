
import pyglet as pgl
import time

from PIL import Image
# from numpy import array
# from pkg_resources import resource_string as resource_bytes

# from flatland.utils.graphics_layer import GraphicsLayer
from flatland.utils.graphics_pil import PILSVG


class PGLGL(PILSVG):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_open = False  # means the window has not yet been opened.
        self.close_requested = False  # user has clicked
        self.closed = False  # windows has been closed (currently, we leave the env still running)

    def open_window(self):
        print("open_window - pyglet")
        assert self.window_open is False, "Window is already open!"
        self.window = pgl.window.Window(resizable=True, vsync=False, width=1200, height=800)
        #self.__class__.window.title("Flatland")
        #self.__class__.window.configure(background='grey')
        self.window_open = True


        @self.window.event
        def on_draw():
            #print("pyglet draw event")
            self.window.clear()
            self.show(from_event=True)
            #print("pyglet draw event done")
            

        @self.window.event
        def on_resize(width, height):
            #print(f"The window was resized to {width}, {height}")
            self.show(from_event=True)
            self.window.dispatch_event("on_draw")
            #print("pyglet resize event done")

        @self.window.event
        def on_close():
            self.close_requested = True


    def close_window(self):
        self.window.close()
        self.closed=True

    def show(self, block=False, from_event=False):
        if not self.window_open:
            self.open_window()

        if self.close_requested:
            if not self.closed:
                self.close_window()
            return
            
        #tStart = time.time()
        self._processEvents()

        pil_img = self.alpha_composite_layers()
        pil_img_resized = pil_img.resize((self.window.width, self.window.height), resample=Image.NEAREST)

        # convert our PIL image to pyglet:
        bytes_image = pil_img_resized.tobytes()
        pgl_image = pgl.image.ImageData(pil_img_resized.width, pil_img_resized.height,
            #self.window.width, self.window.height, 
            'RGBA',
            bytes_image, pitch=-pil_img_resized.width * 4)

        pgl_image.blit(0,0)
        #tEnd = time.time()
        #print("show time: ", tEnd - tStart)

    def _processEvents(self):
        """ This is the replacement for a custom event loop for Pyglet.
            The lines below are typical of Pyglet examples.
            Manually resizing the window is still very clunky.
        """
        #print("process events...", end="")
        pgl.clock.tick()
        #for window in pgl.app.windows:
        if not self.closed:
            self.window.switch_to()
            self.window.dispatch_events()
            self.window.flip()
        #print(" events done")
        
        

    def idle(self, seconds=0.00001):
        tStart = time.time()
        tEnd = tStart + seconds
        while (time.time() < tEnd):
            self._processEvents()
            #self.show()
            time.sleep(min(seconds, 0.1))


def test_pyglet():
    oGL = PGLGL(400,300)
    time.sleep(2)


def test_event_loop():
    """ Shows how it should work with the standard event loop
        Resizing is fairly smooth (ie runs at least 10-20x a second)
    """


    window = pgl.window.Window(resizable=True)
    pil_img = Image.open("notebooks/simple_example_3.png")

    def show():
        pil_img_resized = pil_img.resize((window.width, window.height), resample=Image.NEAREST)
        bytes_image = pil_img_resized.tobytes()
        pgl_image = pgl.image.ImageData(pil_img_resized.width, pil_img_resized.height,
            #self.window.width, self.window.height, 
            'RGBA',
            bytes_image, pitch=-pil_img_resized.width * 4)
        pgl_image.blit(0,0)

    @window.event
    def on_draw():
        print("pyglet draw event")
        window.clear()
        show()
        print("pyglet draw event done")
            

        @window.event
        def on_resize(width, height):
            print(f"The window was resized to {width}, {height}")
            #show()
            print("pyglet resize event done")

        @window.event
        def on_close():
            #self.close_requested = True
            print("close")
    
    pgl.app.run()


if __name__=="__main__":
    #test_pyglet()
    test_event_loop()