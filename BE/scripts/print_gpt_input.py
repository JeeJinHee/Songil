from app.schemas.guide import NextTargetRequest
from app.services.gpt_prompt import build_gpt_input, to_chat_text

# 이화김밥 탭 화면 (메뉴는 참치김밥만 보이고, 다음 ▶ 있음)
SCREEN = {
    "screenSeq": 1,
    "capturedAt": "2026-11-14T05:00:00Z",
    "title": "이화김밥",
    "buttons": [
        {"buttonId": "1", "text": "이화김밥", "type": "tab", "bbox": [0.037, 0.017, 0.185, 0.061]},
        {"buttonId": "2", "text": "이화과자", "type": "tab", "bbox": [0.209, 0.017, 0.357, 0.061]},
        {"buttonId": "3", "text": "세트", "type": "tab", "bbox": [0.381, 0.017, 0.529, 0.061]},
        {"buttonId": "4", "text": "COFFEE", "type": "tab", "bbox": [0.554, 0.017, 0.702, 0.061]},
        {"buttonId": "5", "text": "NON COFFEE", "type": "tab", "bbox": [0.726, 0.017, 0.874, 0.061]},
        {"buttonId": "6", "text": "참치김밥", "type": "menu", "bbox": [0.046, 0.268, 0.241, 0.425]},
        {"buttonId": "7", "text": "다음 ▶", "type": "etc", "bbox": [0.800, 0.850, 0.950, 0.900]},
    ],
}

# (번호, 목표 메뉴, 온도, 정답)
CASES = [
    (1, "아메리카노", "아이스", "4 COFFEE"),
    (2, "녹차라떼", None, "5 NON COFFEE"),
    (3, "스팸김밥", None, "7 다음 ▶"),
]

request = NextTargetRequest(**SCREEN)
for no, menu, temp, answer in CASES:
    options = {"온도": temp} if temp else {}
    print(f"===== 예시 {no} =====")
    print(to_chat_text(build_gpt_input(menu, options, request)))
    print()