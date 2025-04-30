"""
FlatlandCallbacks for InteractiveAI (https://github.com/AI4REALNET/InteractiveAI).

The callbacks create context and events during a scenario run.
If an InteractiveAI instance is up and running, the callbacks  send out HTTP POST requests to InteractiveAI contexts and events REST API endpoints.
In this notebook, we just log the contexts and events that would be sent out.

- The agent positions are sent as context, with geo-coordinates for display on a map.
- Agent malfunctions are sent as events.
"""
import datetime
import logging
import time
import warnings
from pathlib import Path
from typing import Optional, Dict, Tuple

import requests

from flatland.callbacks.callbacks import FlatlandCallbacks
from flatland.envs.rail_env import RailEnv
from flatland.integrations.interactiveai import event_api, context_api, historic_api
from flatland.integrations.interactiveai.context_api import ContextApiApi, ContextIn
from flatland.integrations.interactiveai.event_api import EventApiApi
from flatland.integrations.interactiveai.event_api import EventIn
from flatland.integrations.interactiveai.historic_api import HistoricApiApi

logger = logging.getLogger(__name__)


class FlatlandInteractiveAICallbacks(FlatlandCallbacks):
    def __init__(self,
                 coordinate_map: Dict[Tuple[int, int], Tuple[float, float]],
                 client_id: str = "opfab-client",
                 username: str = "admin",
                 password: str = "test",
                 token_url: str = "http://frontend/auth/token",
                 event_api_host="http://localhost:5001",
                 context_api_host="http://localhost:5100",
                 history_api_host="http://localhost:5200",
                 step_to_millis: int = 1000,
                 collect_only: bool = False,
                 now: datetime.datetime = None,
                 ):
        """

        Parameters
        ----------
        coordinate_map : Dict[Tuple[int, int], Tuple[float, float]]
            map (row,column) to (lat,lon)
        client_id:
            InteractiveAI client_id
        username : str
            InteractiveAI username
        password : str
            InteractiveAI password
        token_url : str
            InteractiveAI token_url
         event_api_host: str
            Defaults to http://localhost:5001",
         context_api_host : str
            Defaults to "http://localhost:5100",
         history_api_host : str
            Defaults to "http://localhost:5200",

        step_to_millis : int
            how many millis delay till next `env.step()`
        collect_only : bool
            Do not send out requests but collect the events and context dicts in the callbacks object.
        now : datetime.datetime
            Real-world time for `env._elapsed_steps == 0`.
            Defaults to `datetime.datetime.now()`.
        """
        self.token_url = token_url
        self.client_id = client_id
        self.username = username
        self.password = password
        self.access_token = None
        self.event_api_host = event_api_host
        self.context_api_host = context_api_host
        self.history_api_host = history_api_host
        self.context_api: ContextApiApi = None
        self.historic_api: HistoricApiApi = None
        self.events_api: EventApiApi = None
        self.in_malfunction = {}
        self.collect_only = collect_only
        self.step_to_millis = step_to_millis
        self.events = []
        self.contexts = []
        self.coordinate_map = coordinate_map
        self.now = now
        if self.now is None:
            self.now = datetime.datetime.now()

    def connect(self):
        access_token = self._get_access_token_password_grant(token_url=self.token_url, client_id=self.client_id, username=self.username, password=self.password)
        print(access_token)

        self.events_api = EventApiApi(event_api.ApiClient(event_api.Configuration(host=self.event_api_host)))
        self.events_api.api_client.set_default_header("Authorization", f"Bearer {access_token}")

        self.context_api = ContextApiApi(context_api.ApiClient(context_api.Configuration(host=self.context_api_host)))
        self.context_api.api_client.set_default_header("Authorization", f"Bearer {access_token}")

        self.historic_api = HistoricApiApi(historic_api.ApiClient(historic_api.Configuration(host=self.history_api_host)))
        self.historic_api.api_client.set_default_header("Authorization", f"Bearer {access_token}")
        self.access_token = access_token

    def _get_access_token_password_grant(self, token_url, client_id, username, password) -> str:
        payload = f"username={username}&password={password}&grant_type=password&clientId={client_id}"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = requests.request("POST", token_url, headers=headers, data=payload)

        response.raise_for_status()
        json_response = response.json()
        return json_response.get("access_token")

    def on_episode_step(
        self,
        *,
        env: Optional[RailEnv] = None,
        data_dir: Path = None,
        **kwargs,
    ) -> None:
        """Called on each episode step (after the action(s) has/have been logged).

        This callback is also called after the final step of an episode,
        meaning when terminated/truncated are returned as True
        from the `env.step()` call.

        The exact time of the call of this callback is after `env.step([action])` and
        also after the results of this step (observation, reward, terminated, truncated,
        infos) have been logged to the given `episode` object.
        """
        if self.access_token is None and not self.collect_only:
            self.connect()

        for agent in env.agents:
            prev = self.in_malfunction.get(agent.handle, False)
            cur = agent.malfunction_handler.in_malfunction
            self.in_malfunction[agent.handle] = cur
            if not prev and cur:
                event = {
                    "use_case": "Railway",
                    "criticality": "LOW",
                    "title": f"Malfunction Train {agent.handle}",
                    "description": f"Malfunction Train {agent.handle}",
                    "start_date": self.now,
                    "end_date": self.now + datetime.timedelta(milliseconds=agent.malfunction_handler.malfunction_down_counter * self.step_to_millis),
                    "data": {
                        "event_type": "Malfunction",
                        "delay": 300,
                        "agent_id": f"{agent.handle}",
                        "id_train": f"{agent.handle}",
                    }
                }
                self.events.append(event)
                if not self.collect_only:
                    self.events_api.api_v1_events_post(EventIn.from_dict(event))

        # TODO use non-blocking calls or queue?
        for agent in env.agents:
            if agent.position is not None and agent.position not in self.coordinate_map:
                warnings.warn(f"Missing mapping for {agent.position}")
        context = {
            "use_case": "Railway",
            "data": {
                "trains": [
                    {
                        'id_train': f'Train {agent.handle}',
                        'latitude': f"{self.coordinate_map[agent.position][0]}",
                        'longitude': f'{self.coordinate_map[agent.position][1]}',
                    } for agent in env.agents if agent.position is not None
                                                 # TODO
                                                 and agent.position in self.coordinate_map]
            }
        }
        self.contexts.append(context)
        if not self.collect_only:
            self.context_api.api_v1_contexts_post(ContextIn.from_dict(context))

        time.sleep(self.step_to_millis * 0.001)
