import uuid

def create_session(client):
    response = client.post("/api/sessions")
    assert response.status_code == 201
    return response.json()

def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}

def test_create_session(client):
    body = create_session(client)
    assert uuid.UUID(body["sessionId"])
    assert body["sessionStatus"] == "ACTIVE"
    assert body["currentStep"] == "ORDER_INPUT"
    assert "createdAt" in body

def test_get_session(client):
    created = create_session(client)
    response = client.get(f"/api/sessions/{created['sessionId']}")
    assert response.status_code == 200
    assert response.json()["sessionId"] == created["sessionId"]

def test_get_session_not_found(client):
    response = client.get(f"/api/sessions/{uuid.uuid4()}")
    assert response.status_code == 404

def test_end_session(client):
    created = create_session(client)
    response = client.post(
        f"/api/sessions/{created['sessionId']}/end", json={"endReason": "USER_CANCELLED"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sessionStatus"] == "CANCELLED"
    assert "endedAt" in body

    # 종료 후 조회하면 종료 상태가 보임
    assert client.get(f"/api/sessions/{created['sessionId']}").json()["sessionStatus"] == "CANCELLED"


def test_end_session_cart_verified_is_completed(client):
    created = create_session(client)
    response = client.post(
        f"/api/sessions/{created['sessionId']}/end", json={"endReason": "CART_VERIFIED"}
    )
    assert response.json()["sessionStatus"] == "COMPLETED"

def test_end_session_twice_conflict(client):
    created = create_session(client)
    url = f"/api/sessions/{created['sessionId']}/end"
    client.post(url, json={"endReason": "ERROR"})
    assert client.post(url, json={"endReason": "ERROR"}).status_code == 409

def test_end_session_invalid_reason(client):
    created = create_session(client)
    response = client.post(
        f"/api/sessions/{created['sessionId']}/end", json={"endReason": "WHATEVER"}
    )
    # Body 검증 실패를 422로 반환
    assert response.status_code == 422

def test_end_session_not_found(client):
    response = client.post(f"/api/sessions/{uuid.uuid4()}/end", json={"endReason": "ERROR"})
    assert response.status_code == 404

def test_stub_returns_501(client):
    created = create_session(client)
    response = client.post(f"/api/sessions/{created['sessionId']}/errors")
    assert response.status_code == 501

def test_datetimes_have_timezone(client):
    created = create_session(client)
    fetched = client.get(f"/api/sessions/{created['sessionId']}").json()
    assert fetched["createdAt"].endswith("Z")
