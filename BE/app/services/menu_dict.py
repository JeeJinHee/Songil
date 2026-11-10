import json
from functools import lru_cache
from pathlib import Path
from app.services.screen_matcher import normalize

MENU_FILE = Path(__file__).resolve().parent.parent / "data" / "kiosk_menu.json"

# 사전 파일은 처음 한 번만 읽기
@lru_cache
def _load() -> dict:
    with open(MENU_FILE, encoding="utf-8") as f:
        return json.load(f)

# 목표 메뉴가 있는 카테고리 이름 (사전에 없으면 None)
def category_of(menu: str) -> str | None:
    want = normalize(menu)
    for m in _load()["menus"]:
        names = [m["name"], *m.get("aliases", [])]
        if want in [normalize(n) for n in names]:
            return m["category"]
    return None

# 카테고리의 페이지 수
def pages_of(category: str) -> int | None:
    for c in _load()["categories"]:
        if normalize(c["name"]) == normalize(category):
            return c["pages"]
    return None
