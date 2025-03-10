import datetime
import logging
import os
from typing import Optional

import requests

from benchmarks.benchmark_episodes import DOWNLOAD_INSTRUCTIONS
from flatland.callbacks.callbacks import FlatlandCallbacks
from flatland.envs.rail_env import RailEnv
from flatland.evaluators.trajectory_evaluator import TrajectoryEvaluator
from flatland.integrations.interactiveai.context_api import ContextApiApi, ContextIn
from flatland.integrations.interactiveai.event_api import EventApiApi
from flatland.integrations.interactiveai.event_api import EventIn
from flatland.integrations.interactiveai.historic_api import HistoricApiApi
from flatland.trajectories.trajectories import Trajectory

logger = logging.getLogger(__name__)


class FlatlandInteractiveAI(FlatlandCallbacks):
    def __init__(self, client_id="opfab-client",
                 username="railway_user",
                 password="test",
                 token_url="http://frontend/auth/token"):
        self.token_url = token_url
        self.client_id = client_id
        self.username = username
        self.password = password
        self.access_token = None
        self.context_api = None
        self.historic_api = None
        self.events_api = None

    def connect(self):
        access_token = self._get_access_token_password_grant(token_url=self.token_url, client_id=self.client_id, username=self.username, password=self.password)
        print(access_token)

        self.events_api = EventApiApi()
        self.events_api.api_client.set_default_header("Authorization", f"Bearer {access_token}")
        self.context_api = ContextApiApi()
        self.context_api.api_client.set_default_header("Authorization", f"Bearer {access_token}")
        self.historic_api = HistoricApiApi()
        self.historic_api.api_client.set_default_header("Authorization", f"Bearer {access_token}")

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
        if self.access_token is None:
            self.connect()

        print(self.events_api.api_v1_events_get())
        print(self.historic_api.api_v1_traces_get())
        print(self.context_api.api_v1_contexts_get())
        self.events_api.api_v1_events_post_with_http_info(EventIn.from_dict({
            "criticality": "LOW",
            "description": "Wonderland",
            # TODO problem with optionals in generated client code, leaving empty leads to 500
            "end_date": datetime.datetime.now() + datetime.timedelta(hours=1),
            "start_date": datetime.datetime.now(),
            "title": "Alice",
            "use_case": "Railway",
            "data": {
                "event_type": "blala",
                "delay": 300,
                "agent_id": "55",
                "id_train": "20"
            }
        }))
        self.context_api.api_v1_contexts_post(ContextIn.from_dict({
            "use_case": "Railway",
            "data": {
                # https://opendata.swiss/de/dataset/haltestelle-perronoberflache1
                "trains": [{
                    'failure': False, 'id_train': 'Olten	Mittelperron	8/9', 'latitude': '47.3534027132627', 'longitude': '7.90817796008907',
                    'nb_passengers_connection': 13, 'nb_passengers_onboard': 200, 'speed': 300},
                    {'failure': False, 'id_train': 'Olten	Mittelperron	2/3 ', 'latitude': '47.3515361449108', 'longitude': '7.90724203700411',
                     'nb_passengers_connection': 13, 'nb_passengers_onboard': 200, 'speed': 300},
                    {'failure': True, 'id_train': 'Olten Hammer	Mittelperron	2/3 ', 'latitude': '47.3481023003965', 'longitude': '7.89839041783048',
                     'nb_passengers_connection': 13, 'nb_passengers_onboard': 200, 'speed': 300}
                ]
            }
        }))

        print("***")
        print(self.events_api.api_v1_events_get())
        print(self.historic_api.api_v1_traces_get())
        print(self.context_api.api_v1_contexts_get())


if __name__ == '__main__':
    data_sub_dir = "30x30 map/10_trains"
    ep_id = "1649ef98-e3a8-4dd3-a289-bbfff12876ce"
    _dir = os.getenv("BENCHMARK_EPISODES_FOLDER")
    assert _dir is not None, (DOWNLOAD_INSTRUCTIONS, _dir)
    assert os.path.exists(_dir), (DOWNLOAD_INSTRUCTIONS, _dir)
    data_dir = os.path.join(_dir, data_sub_dir)

    TrajectoryEvaluator(Trajectory(data_dir=data_dir, ep_id=ep_id), callbacks=FlatlandInteractiveAI()).evaluate()
