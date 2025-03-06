import datetime
import logging

import requests

from flatland.integrations.interactiveai.event_api import ApiClient
from flatland.integrations.interactiveai.event_api import EventApiApi
from flatland.integrations.interactiveai.event_api import EventIn

logger = logging.getLogger(__name__)


def main(
    # api_url="http://localhost:5100/api/v1/usecases",
    # api_url="http://localhost:5100/api/v1/context",
    api_url="http://localhost:5001/api/v1/events",
    client_id="opfab-client",
    username="publisher_test",
    password="test",
    token_url="http://frontend/auth/token"
):
    access_token = get_access_token_password_grant(token_url=token_url, client_id=client_id, username=username, password=password)
    print(access_token)

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    print(requests.get(api_url, headers=headers).json())
    ApiClient.get_default().set_default_header("Authorization", f"Bearer {access_token}")
    events_api = EventApiApi()

    events = events_api.api_v1_events_get()
    print(events)

    events_api.api_v1_events_post_with_http_info(EventIn.from_dict({
        "criticality": "LOW",
        "description": "Wonderland",
        # TODO problem with optionals in generated client code, leaving empty leads to 500
        "end_date": datetime.datetime.now(),
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

    events = events_api.api_v1_events_get()
    print(events)


def get_access_token_password_grant(token_url, client_id, username, password) -> str:
    payload = f"username={username}&password={password}&grant_type=password&clientId={client_id}"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.request("POST", token_url, headers=headers, data=payload)

    response.raise_for_status()
    json_response = response.json()
    return json_response.get("access_token")


if __name__ == '__main__':
    main()
