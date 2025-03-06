import logging

import requests

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
    # response = requests.post(api_url, data=json.dumps(todo), headers=headers)
    print(requests.get(api_url, headers=headers).json())


def get_access_token_password_grant(token_url, client_id, username, password) -> str:
    payload = f"username={username}&password={password}&grant_type=password&clientId={client_id}"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.request("POST", token_url, headers=headers, data=payload)

    response.raise_for_status()
    json_response = response.json()
    return json_response.get("access_token")


if __name__ == '__main__':
    main()
