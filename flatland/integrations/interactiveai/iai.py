import datetime
import logging
import pickle
import time
import warnings
from pathlib import Path
from typing import Optional

import requests

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
    def __init__(self,
                 coordinate_map,
                 client_id="opfab-client",
                 username="admin",
                 password="test",
                 token_url="http://frontend/auth/token",
                 step_to_millis=1000,
                 collect_only=False,
                 now=None
                 ):
        self.token_url = token_url
        self.client_id = client_id
        self.username = username
        self.password = password
        self.access_token = None
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

        self.events_api = EventApiApi()
        self.events_api.api_client.set_default_header("Authorization", f"Bearer {access_token}")
        self.context_api = ContextApiApi()
        self.context_api.api_client.set_default_header("Authorization", f"Bearer {access_token}")
        self.historic_api = HistoricApiApi()
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
        if env._elapsed_steps == 1 and not self.collect_only:
            # self.events_api.api_v1_delete_
            # self.historic_api.api_v1_delete_all_data_delete()
            # self.context_api.api_v1_delete_all_data_delete()
            for event in self.events_api.api_v1_events_get():
                self.events_api.api_v1_event_event_id_delete(event_id=event.id_event)
            print(self.events_api.api_v1_events_get())
            print(self.historic_api.api_v1_traces_get())
            print(self.context_api.api_v1_contexts_get())

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


if __name__ == '__main__':
    # data from https://github.com/flatland-association/flatland-scenarios/tree/scenario-olten/scenario_olten/data
    data_dir = Path("../../../../flatland-scenarios2/scenario_olten/data").resolve()
    trajectory = Trajectory(data_dir, ep_id="olten")

    with (data_dir / "position_to_latlon_olten.pkl").open("rb") as file_in:
        position_to_latlon_olten = pickle.load(file_in)
    cb = FlatlandInteractiveAI(position_to_latlon_olten, collect_only=True)

    TrajectoryEvaluator(trajectory, cb).evaluate()

    print(cb.contexts)
    print(cb.events)
