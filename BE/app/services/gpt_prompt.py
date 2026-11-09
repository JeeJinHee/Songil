import json

from app.schemas.guide import NextTargetRequest

SYSTEM_PROMPT = """너는 키오스크 안내 도우미다. 사용자가 찾는 메뉴가 지금 화면에 없다.
화면 버튼 목록(JSON)을 보고, 메뉴를 찾기 위해 다음에 누를 버튼 하나를 고른다.
1. 메뉴가 다른 카테고리에 속하면 그 카테고리 탭(type="tab")을 고른다.
2. 메뉴가 지금 카테고리에 속하는데 화면에 없으면 "다음" 버튼을 고른다.
3. 둘 다 아니면 buttonId=null.
- (아)=아이스, (핫)=따뜻한 음료
- bbox는 0~1 좌표 [x1, y1, x2, y2], y가 작을수록 위쪽 (탭은 보통 화면 위쪽에 있다)
- 반드시 목록에 있는 buttonId만 사용한다
- 답은 JSON 하나로만: {"buttonId": string 또는 null, "buttonText": string 또는 null, "reason": string}"""

# GPT에 넘길 입력: 목표 메뉴·온도 + 화면 제목 + 버튼 목록
def build_gpt_input(menu: str, options: dict, request: NextTargetRequest) -> dict:
    return {
        "goal": {"menu": menu, "temperature": options.get("온도")},
        "title": request.title,
        "buttons": [
            {"buttonId": b.button_id, "text": b.text, "type": b.type, "bbox": b.bbox}
            for b in request.buttons
        ],
    }

# ChatGPT 창에 그대로 붙여 넣을 글 (프롬프트 + 입력 JSON)
def to_chat_text(gpt_input: dict) -> str:
    return SYSTEM_PROMPT + "\n---\n" + json.dumps(gpt_input, ensure_ascii=False, indent=2)