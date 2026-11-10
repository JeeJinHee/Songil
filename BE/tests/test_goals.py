import uuid


def goal_url(client):
    session_id = client.post("/api/sessions").json()["sessionId"]
    return f"/api/sessions/{session_id}/goal"


def test_create_goal(client):
    url = goal_url(client)
    response = client.post(url, json={"text": "아이스 아메리카노 두 잔"})
    assert response.status_code == 200
    body = response.json()
    assert body["goal"] == {"menu": "아메리카노", "options": {"온도": "아이스"}, "quantity": 2}
    assert body["goalStatus"] == "NEED_CONFIRM"
    assert body["message"]


def test_create_goal_not_understood(client):
    url = goal_url(client)
    body = client.post(url, json={"text": "두 잔"}).json()
    assert body["goal"] is None
    assert body["goalStatus"] == "NOT_UNDERSTOOD"
    assert client.get(url).status_code == 404


def test_create_goal_empty_text(client):
    url = goal_url(client)
    assert client.post(url, json={"text": ""}).status_code == 422


def test_create_goal_overwrites_unconfirmed(client):
    url = goal_url(client)
    client.post(url, json={"text": "아메리카노"})
    client.post(url, json={"text": "카페라떼 세 잔"})
    assert client.get(url).json()["goal"]["menu"] == "카페라떼"


def test_confirm_goal(client):
    url = goal_url(client)
    client.post(url, json={"text": "아메리카노"})
    response = client.post(f"{url}/confirm", json={"action": "CONFIRM"})
    assert response.status_code == 200
    assert response.json()["goalStatus"] == "CONFIRMED"
    assert response.json()["currentStep"] == "MENU_SELECT"
    assert client.get(url).json()["goalStatus"] == "CONFIRMED"


def test_create_goal_after_confirm_conflict(client):
    url = goal_url(client)
    client.post(url, json={"text": "아메리카노"})
    client.post(f"{url}/confirm", json={"action": "CONFIRM"})
    assert client.post(url, json={"text": "카페라떼"}).status_code == 409


def test_confirm_without_goal_conflict(client):
    url = goal_url(client)
    assert client.post(f"{url}/confirm", json={"action": "CONFIRM"}).status_code == 409


def test_retry_clears_goal(client):
    url = goal_url(client)
    client.post(url, json={"text": "아메리카노"})
    response = client.post(f"{url}/confirm", json={"action": "RETRY"})
    assert response.json()["goalStatus"] == "CLEARED"
    assert response.json()["currentStep"] == "ORDER_INPUT"
    assert client.get(url).status_code == 404


def test_browse(client):
    url = goal_url(client)
    response = client.post(f"{url}/confirm", json={"action": "BROWSE"})
    assert response.json()["goalStatus"] == "BROWSING"
    body = client.get(url).json()
    assert body["goal"]["menu"] is None
    assert body["goalStatus"] == "BROWSING"


def test_confirm_invalid_action(client):
    url = goal_url(client)
    assert client.post(f"{url}/confirm", json={"action": "X"}).status_code == 422


def test_goal_session_not_found(client):
    url = f"/api/sessions/{uuid.uuid4()}/goal"
    assert client.post(url, json={"text": "아메리카노"}).status_code == 404
    assert client.get(url).status_code == 404


def test_goal_after_session_ended_conflict(client):
    url = goal_url(client)
    client.post(url.removesuffix("/goal") + "/end", json={"endReason": "USER_CANCELLED"})
    assert client.post(url, json={"text": "아메리카노"}).status_code == 409

# 온도를 붙여 말해도 온도와 메뉴를 나눔
def test_temperature_attached(client):
    for text, temp in [("아이스아메리카노", "아이스"), ("핫아메리카노", "따뜻한"), ("아이스 아메리카노", "아이스")]:
        url = goal_url(client)
        body = client.post(url, json={"text": text}).json()
        assert body["goal"]["menu"] == "아메리카노"
        assert body["goal"]["options"] == {"온도": temp}
