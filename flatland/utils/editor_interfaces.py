

class AbstractView:
    """ AbstractView - a class to encompass and assemble the Jupyter Editor Model-View-Controller.
    """

    def __init__(self, model:"AbstractModel", sGL="PIL"):
        pass

    def init_canvas(self):
        pass

    def init_widgets(self):
        pass
    
class AbstractController:
    """ AbstractController - a class to encompass and assemble the Jupyter Editor Model-View-Controller.
    """

    def __init__(self, model:"AbstractModel", view:AbstractView):
        pass

    def set_model(self, model):
        pass

    def _getCoords(self, event:dict):
        pass

    def getModKeys(self, event:dict):
        pass

    def on_click(self, event):
        pass

    def set_debug(self, event):
        pass

    def set_debug_move(self, event):
        pass

    def handle_event(self, event:dict):
        pass

    def getBoundingRectYX(self):
        pass

class AbstractModel:
    """ AbstractModel - a class to encompass and assemble the Jupyter Editor Model-View-Controller.
    """

    def __init__(self, view:AbstractView):
        pass

    def set_view(self, view):
        pass

    def set_debug(self, bDebug):
        pass

    def set_debug_move(self, bDebugMove):
        pass

    def set_filename(self, sFilename):
        pass

    def clear_stroke(self):
        pass
