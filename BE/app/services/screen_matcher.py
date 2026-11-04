import re

ICE_MARKS = ["(아이스)", "(ICE)", "(아)", "아이스", "ICE"]
HOT_MARKS = ["(따뜻한)", "(HOT)", "(핫)", "HOT"]
ICE_WORDS = ["아이스", "차가", "ICE", "시원"]
HOT_WORDS = ["따뜻", "뜨거", "HOT", "핫"]

def normalize(text: str) -> str:
    return text.replace(" ", "").upper()

def split_temperature(text: str) -> tuple[str | None, str]:
    name = normalize(text)
    for mark in ICE_MARKS:
        if name.startswith(mark):
            return "아이스", name[len(mark):]
    for mark in HOT_MARKS:
        if name.startswith(mark):
            return "따뜻한", name[len(mark):]
    return None, name

def temperature_of(value: str) -> str | None:
    value = normalize(value)
    if any(word in value for word in ICE_WORDS):
        return "아이스"
    if any(word in value for word in HOT_WORDS):
        return "따뜻한"
    return None


def group_name(group: str) -> str:
    return re.sub(r"\(.*?\)", "", group).strip()

def is_required(group: str) -> bool:
    return "필수" in group


def match_choice(answer: str, choices: list[str]) -> str | None:
    words = [w for w in answer.split() if len(w) >= 2]
    matched = []
    for choice in choices:
        # 1) 띄어쓰기·대문자 정리 → "-" 지우기
        core = normalize(choice)
        core = re.sub(r"[+-]\d[\d,]*", "", core)
        core = core.replace("-", "")

        # 2) 단어 중 하나라도 맞으면 matched에 추가
        for word in words:
            word = normalize(word)
            if word in core or core[:3] in word:
                matched.append(choice)
                break

    # 3) 딱 1개만 맞았을 때만 돌려주기
    if len(matched) == 1:
        return matched[0]
    return None