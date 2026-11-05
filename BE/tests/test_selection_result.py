import uuid
from tests.test_next_target import MENU, POPUP, confirmed_url, screen

# 아이스 아메리카노 목표로 m1을 안내받은 상태를 만들고 selection-result 주소를 돌려줌
def guided_url(client):
    next_url = confirmed_url(client, "아이스 아메리카노 두 잔")
    client.post(next_url, json=screen(1, MENU))
    return next_url.replace("next-target", "selection-result")

# 선택 결과 요청 본문 만들기
def pressed(seq, target, changed, buttons, cart_rows=None):
    return {
        "targetButtonId": target,
        "screenSeq": seq,
        "screenChanged": changed,
        "buttons": buttons,
        "cartRows": cart_rows or [],
    }

# 메뉴를 눌렀더니 옵션 팝업이 뜸
def test_menu_success_by_popup(client):
    body = client.post(guided_url(client), json=pressed(2, "m1", True, POPUP)).json()
    assert body["result"] == "SUCCESS"
    assert body["currentStep"] == "MENU_SELECT"

# 화면은 그대로인데 장바구니에 담김 (이화상점)
def test_menu_success_by_cart(client):
    cart = [{"texts": ["(아) 아메리카노", "1"]}]
    body = client.post(guided_url(client), json=pressed(2, "m1", False, MENU, cart)).json()
    assert body["result"] == "SUCCESS"

# 화면은 바뀌었는데 팝업도 장바구니도 없음
def test_menu_wrong_selection(client):
    other = [{"buttonId": "x1", "text": "카페라떼"}]
    body = client.post(guided_url(client), json=pressed(2, "m1", True, other)).json()
    assert body["result"] == "WRONG_SELECTION"

# 아무 변화 없음
def test_menu_unconfirmed(client):
    body = client.post(guided_url(client), json=pressed(2, "m1", False, MENU)).json()
    assert body["result"] == "UNCONFIRMED"

# 결과가 DB에 저장되고, 다음 next-target도 이어서 됨
def test_then_next_target(client):
    url = guided_url(client)
    client.post(url, json=pressed(2, "m1", True, POPUP))
    body = client.post(url.replace("selection-result", "next-target"), json=screen(2, POPUP)).json()
    assert body["result"] == "NEED_CHOICE"

# 안내하지 않은 버튼 ID
def test_wrong_target_id(client):
    response = client.post(guided_url(client), json=pressed(2, "m2", True, POPUP))
    assert response.status_code == 409

# 아직 아무것도 안내 안 함
def test_no_guided_step(client):
    url = confirmed_url(client).replace("next-target", "selection-result")
    assert client.post(url, json=pressed(1, "m1", True, POPUP)).status_code == 409

# 옛날 화면
def test_stale_screen(client):
    url = guided_url(client)
    client.post(url, json=pressed(5, "m1", True, POPUP))
    assert client.post(url, json=pressed(3, "m1", True, POPUP)).status_code == 409

# 없는 세션
def test_unknown_session(client):
    url = f"/api/sessions/{uuid.uuid4()}/selection-result"
    assert client.post(url, json=pressed(1, "m1", True, POPUP)).status_code == 404