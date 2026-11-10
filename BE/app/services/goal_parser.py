import re
from app.schemas.goal import Goal

# Claude API 연결 전까지 쓰는 임시 규칙 기반
KOREAN_NUMBERS = {"한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4, "다섯": 5}
TEMPERATURE_WORDS = {"아이스": "아이스", "차가운": "아이스", "따뜻한": "따뜻한", "핫": "따뜻한"}
QUANTITY_PATTERN = re.compile(r"(\d+|하나|한|둘|두|셋|세|넷|네|다섯)\s*(잔|개)")

def parse_goal(text: str) -> Goal | None:
    quantity = 1
    match = QUANTITY_PATTERN.search(text)
    if match:
        number = match.group(1)
        quantity = int(number) if number.isdigit() else KOREAN_NUMBERS[number]
        text = text[: match.start()] + text[match.end() :]

    options: dict[str, str] = {}
    menu_words: list[str] = []

    for word in text.split():
        for temp_word, temp in TEMPERATURE_WORDS.items():
            if word.startswith(temp_word):
                options["온도"] = temp
                word = word[len(temp_word):]
                break
        if word:
            menu_words.append(word)
        
    if not menu_words:
        return None
    return Goal(menu=" ".join(menu_words), options=options, quantity=quantity)
