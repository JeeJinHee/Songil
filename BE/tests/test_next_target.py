import uuid

MENU = [
    {"buttonId": "m1", "text": "(아) 아메리카노"},
    {"buttonId": "m2", "text": "(핫) 아메리카노"},
    {"buttonId": "n1", "text": "다음 ▶"},
]
POPUP = [
    {"buttonId": "o1", "text": "매장컵", "group": "컵선택(필수)"},
    {"buttonId": "o2", "text": "일회용컵", "group": "컵선택(필수)"},
    {"buttonId": "o3", "text": "샷추가 +500", "group": "샷(선택)"},
    {"buttonId": "ok", "text": "선택완료"},
]
TABS = [
    {"buttonId": "t1", "text": "이화김밥", "type": "tab"},
    {"buttonId": "t2", "text": "COFFEE", "type": "tab"},
]

# 목표를 확정한 세션을 만들고 next-target 주소를 돌려줌
def confirmed_url(client, text="아메리카노 두 잔"):
    session_id = client.post("/api/sessions").json()["sessionId"]
    client.post(f"/api/sessions/{session_id}/goal", json={"text": text})
    client.post(f"/api/sessions/{session_id}/goal/confirm", json={"action": "CONFIRM"})
    return f"/api/sessions/{session_id}/next-target"

# 화면 요청 본문 만들기
def screen(seq, buttons, cart_rows=None, answer=None):
    return {
        "screenSeq": seq,
        "capturedAt": "2026-11-05T10:00:00Z",
        "buttons": buttons,
        "cartRows": cart_rows or [],
        "answer": answer,
    }

def test_asks_temperature_when_both_exist(client):
    body = client.post(confirmed_url(client), json=screen(1, MENU)).json()
    assert body["result"] == "NEED_CHOICE"
    assert body["choices"] == ["(아) 아메리카노", "(핫) 아메리카노"]
    assert body["currentStep"] == "MENU_SELECT"

def test_answer_picks_temperature(client):
    url = confirmed_url(client)
    client.post(url, json=screen(1, MENU))
    body = client.post(url, json=screen(2, MENU, answer="아이스로 할게")).json()
    assert body["result"] == "TARGET"
    assert body["target"]["text"] == "(아) 아메리카노"

def test_goal_temperature_skips_question(client):
    url = confirmed_url(client, "따뜻한 아메리카노")
    body = client.post(url, json=screen(1, MENU)).json()
    assert body["result"] == "TARGET"
    assert body["target"]["buttonId"] == "m2"

def test_sold_out(client):
    url = confirmed_url(client, "아이스 아메리카노")
    buttons = [{"buttonId": "m1", "text": "(아) 아메리카노", "soldOut": True}]
    body = client.post(url, json=screen(1, buttons)).json()
    assert body["result"] == "SOLD_OUT"

def test_next_page_when_menu_missing(client):
    url = confirmed_url(client, "카페라떼")
    body = client.post(url, json=screen(1, MENU)).json()
    assert body["result"] == "TARGET"
    assert body["target"]["text"] == "다음 ▶"

def test_not_found_without_next_page(client):
    url = confirmed_url(client, "카페라떼")
    body = client.post(url, json=screen(1, MENU[:2])).json()
    assert body["result"] == "NOT_FOUND"

def test_option_popup_flow(client):
    url = confirmed_url(client)
    body = client.post(url, json=screen(1, POPUP)).json()
    assert body["result"] == "NEED_CHOICE"
    assert body["choices"] == ["매장컵", "일회용컵"]

    body = client.post(url, json=screen(2, POPUP, answer="일회용컵")).json()
    assert body["target"]["text"] == "일회용컵"

    body = client.post(url, json=screen(3, POPUP)).json()
    assert body["target"]["text"] == "선택완료"

def test_plus_then_done(client):
    url = confirmed_url(client)
    plus = MENU + [{"buttonId": "p1", "text": "+", "group": "(아) 아메리카노"}]

    body = client.post(url, json=screen(1, plus, cart_rows=[{"texts": ["(아) 아메리카노", "1", "2,000"]}])).json()
    assert body["target"]["buttonId"] == "p1"
    assert body["currentStep"] == "QUANTITY_SELECT"

    body = client.post(url, json=screen(2, plus, cart_rows=[{"texts": ["(아) 아메리카노", "2", "4,000"]}])).json()
    assert body["result"] == "DONE"


def test_unknown_session(client):
    url = f"/api/sessions/{uuid.uuid4()}/next-target"
    assert client.post(url, json=screen(1, MENU)).status_code == 404

def test_goal_not_confirmed(client):
    session_id = client.post("/api/sessions").json()["sessionId"]
    url = f"/api/sessions/{session_id}/next-target"
    assert client.post(url, json=screen(1, MENU)).status_code == 409

def test_stale_screen(client):
    url = confirmed_url(client)
    client.post(url, json=screen(5, MENU))
    assert client.post(url, json=screen(3, MENU)).status_code == 409

def test_ended_session(client):
    url = confirmed_url(client)
    session_url = url.replace("/next-target", "")
    client.post(f"{session_url}/end", json={"endReason": "USER_CANCELLED"})
    assert client.post(url, json=screen(1, MENU)).status_code == 409


# 메뉴도 다음 ▶도 없으면 안 본 탭을 차례로 안내
def test_tabs_in_order(client):
    url = confirmed_url(client, "딸기스무디")
    first = client.post(url, json=screen(1, TABS)).json()
    assert first["target"]["text"] == "이화김밥"
    assert first["currentStep"] == "CATEGORY_SELECT"

    second = client.post(url, json=screen(2, TABS)).json()
    assert second["target"]["text"] == "COFFEE"          # 이화김밥은 이미 봤으니까

# 탭을 다 봤는데도 없으면 NOT_FOUND
def test_all_tabs_seen(client):
    url = confirmed_url(client, "딸기스무디")
    client.post(url, json=screen(1, TABS))
    client.post(url, json=screen(2, TABS))
    body = client.post(url, json=screen(3, TABS)).json()
    assert body["result"] == "NOT_FOUND"

# 다음을 3번 넘기면 탭으로 넘어감
def test_page_limit_then_tab(client):
    url = confirmed_url(client, "딸기스무디")
    for seq in range(1, 4):
        body = client.post(url, json=screen(seq, MENU + TABS)).json()
        assert body["currentStep"] == "PAGE_NEXT"
    body = client.post(url, json=screen(4, MENU + TABS)).json()
    assert body["currentStep"] == "CATEGORY_SELECT"
    assert body["target"]["text"] == "이화김밥"

def test_message_reads_temperature(client):
    url = confirmed_url(client, "아이스 아메리카노")
    body = client.post(url, json=screen(1, MENU)).json()
    assert body["target"]["text"] == "(아) 아메리카노"
    assert body["message"] == "아이스 아메리카노 버튼을 누르세요."

def test_source_rule_and_fallback(client):
    url = confirmed_url(client, "아이스 아메리카노")
    body = client.post(url, json=screen(1, MENU)).json()
    assert body["source"] == "rule"

    url = confirmed_url(client, "카페라떼")
    body = client.post(url, json=screen(1, TABS)).json()
    assert body["source"] == "fallback"
    assert body["reason"]

# 사전에 있는 메뉴면 그 카테고리 탭부터
def test_dictionary_tab_first(client):
    url = confirmed_url(client, "카페라떼")
    body = client.post(url, json=screen(1, TABS)).json()
    assert body["target"]["text"] == "COFFEE"
    assert body["currentStep"] == "CATEGORY_SELECT"
    assert "COFFEE" in body["reason"]

# 사전에서 1페이지인 탭이면 다음 ▶ 없이 다른 탭으로
def test_dictionary_pages_limit(client):
    url = confirmed_url(client, "카페라떼")
    client.post(url, json=screen(1, TABS))
    next_btn = [{"buttonId": "n1", "text": "다음 ▶"}]
    body = client.post(url, json=screen(2, TABS + next_btn)).json()
    assert body["currentStep"] == "CATEGORY_SELECT"
    assert body["target"]["text"] == "이화김밥"