

from flask import Flask, request, redirect, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS, cross_origin
import threading
import os
import time
import webbrowser
import numpy as np
import typing
import socket

from flatland.envs.rail_env import RailEnv, RailEnvActions


#async_mode = None


class simple_flask_server(object):
    """ I wanted to wrap the flask server in a class but this seems to be quite hard;
        eg see: https://stackoverflow.com/questions/40460846/using-flask-inside-class
        I have made a messy sort of singleton pattern.
        It might be easier to revert to the "standard" flask global functions + decorators.
    """

    static_folder = os.path.join(os.getcwd(), "static")
    print("Flask static folder: ", static_folder)
    app = Flask(__name__, 
        static_url_path='',
        static_folder=static_folder)
        
    socketio = SocketIO(app, cors_allowed_origins='*')

    # This is the original format for the I/O.
    # It comes from the format used in the msgpack saved episode.
    # The lists here are truncated from the original - see CK's original main.py, in flatland-render.
    gridmap = [
        # list of rows (?).  Each cell is a 16-char binary string. Yes this is inefficient!
        ["0000000000000000", "0010000000000000", "0000000000000000", "0000000000000000", "0010000000000000", "0000000000000000", "0000000000000000", "0000000000000000", "0010000000000000", "0000000000000000"],
        ["0000000000000000", "1000000000100000", "0000000000000000", "0000000000000000", "0000000001001000", "0001001000000000", "0010000000000000", "0000000000000000", "1000000000100000", "0000000000000000"],  # ...
        ]
    agents_static = [
        # [initial position], initial direction, [target], 0 (?)
        [[7, 9], 2, [3, 5], 0, 
            # Speed and malfunction params
            {"position_fraction": 0, "speed": 1, "transition_action_on_cellexit": 3},  
            {"malfunction": 0, "malfunction_rate": 0,  "next_malfunction": 0, "nr_malfunctions": 0}], 
        [[8,  8],  1,  [1,  6],  0,  
            {"position_fraction": 0, "speed": 1, "transition_action_on_cellexit": 2},  
            {"malfunction": 0, "malfunction_rate": 0,  "next_malfunction": 0, "nr_malfunctions": 0}], 
        [[3,  7],  2,  [0,  1],  0,  
            {"position_fraction": 0, "speed": 1, "transition_action_on_cellexit": 2},  
            {"malfunction": 0, "malfunction_rate": 0,  "next_malfunction": 0, "nr_malfunctions": 0}]
        ]
    
    # "actions" are not really actions, but [row, col, direction] for each agent, at each time step
    # This format does not yet handle agents which are in states inactive or done_removed
    actions= [
        [[7, 9, 2], [8, 8, 1], [3, 7, 2]], [[7, 9, 2], [8, 7, 3], [2, 7, 0]],  # ...
            ]

    def __init__(self, env):
        # Some ugly stuff with cls and self here
        cls = self.__class__
        cls.instance = self   # intended as singleton

        cls.app.config['CORS_HEADERS'] = 'Content-Type'
        cls.app.config['SECRET_KEY'] = 'secret!'

        self.app = cls.app
        self.socketio = cls.socketio
        self.env = env
        self.renderer_ready = False  # to indicate env background not yet drawn
        self.port = None  # we only assign a port once we start the background server...
        self.host = None

    def run_flask_server(self, host='127.0.0.1', port=None):
        self.host = host

        if port is None:
            self.port = self._find_available_port(host)
        else:
            self.port = port
                    
        self.socketio.run(simple_flask_server.app, host=host, port=self.port)
    
    def run_flask_server_in_thread(self,  host="127.0.0.1", port=None):
        # daemon=True so that this thread exits when the main / foreground thread exits,
        # usually when the episode finishes.
        self.thread = threading.Thread(
            target=self.run_flask_server, 
            kwargs={"host": host, "port": port}, 
            daemon=True)
        self.thread.start()
        # short sleep to allow thread to start (may be unnnecessary)
        time.sleep(1)
    
    def open_browser(self):
        webbrowser.open("http://localhost:{}".format(self.port))
        # short sleep to allow browser to request the page etc (may be unnecessary)
        time.sleep(1)
    
    def _test_listen_port(self, host: str, port: int):
        oSock = socket.socket()
        try:
            oSock.bind((host, port))
        except OSError:
            return False  # The port is not available

        del oSock  # This should release the port        
        return True  # The port is available

    def _find_available_port(self, host: str, port_start: int = 8080):
        for nPort in range(port_start, port_start+100):
            if self._test_listen_port(host, nPort):
                return nPort
        print("Could not find an available port for Flask to listen on!")
        return None

    def get_endpoint_url(self):
        return "http://{}:{}".format(self.host, self.port)

    @app.route('/', methods=['GET'])
    def home():
        # redirects from "/" to "/index.html" which is then served from static.
        # print("Here - / - cwd:", os.getcwd())
        return redirect("index.html")

    @staticmethod
    @socketio.on('connect')
    def connected():
        '''
        When the JS Renderer connects,
        this method will send the env and agent information
        '''
        cls = simple_flask_server
        print('Client connected')
        
        # Do we really need this?
        cls.socketio.emit('message', {'message': 'Connected'})
        
        print('Send Env grid and agents')
        # cls.socketio.emit('grid', {'grid': cls.gridmap, 'agents_static': cls.agents_static}, broadcast=False)
        cls.instance.send_env()
        print("Env and agents sent")

    @staticmethod
    @socketio.on('disconnect')
    def disconnected():
        print('Client disconnected')

    def send_actions(self, dict_actions):
        ''' Sends the agent positions and directions, not really actions.
        '''
        llAgents = self.agents_to_list()
        self.socketio.emit('agentsAction', {'actions': llAgents})

    def send_observation(self, agent_handles, dict_obs):
        """ Send an observation.  
            TODO: format observation message.
        """
        self.socketio.emit("observation", {"agents": agent_handles, "observations": dict_obs})

    def send_env(self):
        """ Sends the env, ie the rail grid, and the agents (static) information
        """
        # convert 2d array of int into 2d array of 16char strings
        g2sGrid = np.vectorize(np.binary_repr)(self.env.rail.grid, width=16)
        llGrid = g2sGrid.tolist()
        llAgents = self.agents_to_list_dict()
        self.socketio.emit('grid', {
                'grid': llGrid, 
                'agents_static': llAgents
                },
             broadcast=False)
    
    def send_env_and_wait(self):
        for iAttempt in range(30):
            if self.is_renderer_ready():
                print("Background Render complete")
                break
            else:
                print("Waiting for browser to signal that rendering complete")
                time.sleep(1)

    @staticmethod
    @socketio.on('renderEvent')
    def handle_render_event(data):
        cls=simple_flask_server
        self = cls.instance
        print('RenderEvent!!')
        print('status: ' + data['status'])
        print('message: ' + data['message'])

        if data['status'] == 'listening':
            self.renderer_ready = True
    
    def is_renderer_ready(self):
        return self.renderer_ready

    def agents_to_list_dict(self):
        ''' Create a list of lists / dicts for serialisation
            Maps from the internal representation in EnvAgent to 
            the schema used by the Javascript renderer.
        '''
        llAgents = []
        for agent in self.env.agents:
            if agent.position is None:
                # the int()s are to convert from numpy int64 which causes problems in serialization 
                # to plain old python int
                lPos = [int(agent.initial_position[0]), int(agent.initial_position[1])]
            else:
                lPos = [int(agent.position[0]), int(agent.position[1])]

            lAgent = [
                        lPos,
                        int(agent.direction),
                        [int(agent.target[0]), int(agent.target[1])], 0, 
                        {   # dummy values:
                            "position_fraction": 0,
                            "speed": 1,
                            "transition_action_on_cellexit": 3
                        }, 
                        {
                            "malfunction": 0,
                            "malfunction_rate": 0,
                            "next_malfunction": 0,
                            "nr_malfunctions": 0
                        } 
                ]
            llAgents.append(lAgent)
        return llAgents
    
    def agents_to_list(self):
        llAgents = []
        for agent in self.env.agents:
            if agent.position is None:
                lPos = [int(agent.initial_position[0]), int(agent.initial_position[1])]
            else:
                lPos = [int(agent.position[0]), int(agent.position[1])]
            iDir = int(agent.direction)

            lAgent = [*lPos, iDir]

            llAgents.append(lAgent)
        return llAgents



def main1():

    print('Run Flask SocketIO Server')
    server = simple_flask_server()
    threading.Thread(target=server.run_flask_server).start()
    # Open Browser
    webbrowser.open('http://127.0.0.1:8080')

    print('Send Action')
    for i in server.actions:
        time.sleep(1)
        print('send action')
        server.socketio.emit('agentsAction', {'actions': i})





if __name__ == "__main__":
    main1()