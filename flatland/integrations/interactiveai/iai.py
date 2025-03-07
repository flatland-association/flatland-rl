import datetime
import logging

import requests

from flatland.integrations.interactiveai.context_api import ContextApiApi, ContextIn
from flatland.integrations.interactiveai.event_api import EventApiApi
from flatland.integrations.interactiveai.event_api import EventIn
from flatland.integrations.interactiveai.historic_api import HistoricApiApi

logger = logging.getLogger(__name__)


def main(
    client_id="opfab-client",
    username="railway_user",
    password="test",
    token_url="http://frontend/auth/token"
):
    access_token = get_access_token_password_grant(token_url=token_url, client_id=client_id, username=username, password=password)
    print(access_token)

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    events_api = EventApiApi()
    events_api.api_client.set_default_header("Authorization", f"Bearer {access_token}")
    context_api = ContextApiApi()
    context_api.api_client.set_default_header("Authorization", f"Bearer {access_token}")
    historic_api = HistoricApiApi()
    historic_api.api_client.set_default_header("Authorization", f"Bearer {access_token}")

    print(events_api.api_v1_events_get())
    print(historic_api.api_v1_traces_get())
    print(context_api.api_v1_contexts_get())
    events_api.api_v1_events_post_with_http_info(EventIn.from_dict({
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
    context_api.api_v1_contexts_post(ContextIn.from_dict({
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
    print(events_api.api_v1_events_get())
    print(historic_api.api_v1_traces_get())
    print(context_api.api_v1_contexts_get())


def get_access_token_password_grant(token_url, client_id, username, password) -> str:
    payload = f"username={username}&password={password}&grant_type=password&clientId={client_id}"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.request("POST", token_url, headers=headers, data=payload)

    response.raise_for_status()
    json_response = response.json()
    return json_response.get("access_token")


if __name__ == '__main__':
    main()
