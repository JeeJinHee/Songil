from app.schemas.guide import NextTargetRequest
from app.services.gpt_prompt import build_gpt_input

def test_gpt_input_fields():
    request = NextTargetRequest(
        screen_seq=1,
        captured_at="2026-11-14T05:00:00Z",
        buttons=[{"buttonId": "4", "text": "COFFEE", "type": "tab", "bbox": [0.5, 0.0, 0.7, 0.1], "soldOut": False}],
    )
    data = build_gpt_input("아메리카노", {"온도": "아이스"}, request)
    assert data["goal"] == {"menu": "아메리카노", "temperature": "아이스"}
    assert data["buttons"] == [{"buttonId": "4", "text": "COFFEE", "type": "tab", "bbox": [0.5, 0.0, 0.7, 0.1]}]