import uuid

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models import GuideStep
from app.models.goal import GoalStatus
from app.models.session import ENDED_STATUSES, CurrentStep
from app.schemas.guide import (Button, NextTargetRequest, NextTargetResponse, NextTargetResult, Target, SelectionResultResponse, SelectionResultRequest)
from app.services.goal_service import _find_goal
from app.services.session_service import SessionAlreadyEnded, get_session, log_event

from app.services.screen_matcher import match_choice, normalize, split_temperature, temperature_of, group_name, is_required
from app.models.guide import SelectionResult

# 예외처리
class GoalNotConfirmed(Exception):
    pass
class StaleScreen(Exception):
    pass
class NoGuidedStep(Exception):
    pass

def _reply(order_session, request, result, message, choices=None) -> NextTargetResponse:
    return NextTargetResponse(
        screen_seq=request.screen_seq,
        result=result,
        choices=choices,
        current_step=order_session.current_step,
        message=message,
    )

def _target(db, order_session, request, button: Button, step: CurrentStep, message=None) -> NextTargetResponse:
    db.add(GuideStep(
        session_id=order_session.id,
        screen_seq=request.screen_seq,
        step=step,
        target_button_id=button.button_id,
        target_text=button.text,
        target_group=button.group,
    ))
    order_session.current_step = step
    return NextTargetResponse(
        screen_seq=request.screen_seq,
        result=NextTargetResult.TARGET,
        target=Target(button_id=button.button_id, text=button.text),
        current_step=step,
        message=message or f"{button.text} 버튼을 누르세요.",
    )

def _is_option_popup(buttons: list[Button]) -> bool:
    for b in buttons:
        if b.text == "선택완료":
            return True
        if b.group and ("필수" in b.group or "(선택)" in b.group):
            return True
    return False


# 장바구니에 목표 메뉴 중 몇 개가 담겼는지?
def _cart_quantity(goal, cart_rows) -> int:
    menu = normalize(goal.menu)
    for row in cart_rows:
        for text in row.texts:
            if split_temperature(text) [1] == menu:
                numbers = [t for t in row.texts if t.isdigit() ]
                return int(numbers[0]) if numbers else 1
    return 0
# 화면에서 목표 메뉴 버튼 찾기 (온도가 정해져 있으면 그 온도만)
def _find_menu_buttons(goal, buttons) -> list[Button]:
    menu = normalize(goal.menu)
    want = temperature_of(goal.options.get("온도", ""))
    found = []
    for b in buttons:
        if b.group: continue
        temp, name = split_temperature(b.text)
        if name != menu: continue
        if want and temp and temp != want: continue
        found.append(b)
    return found

# 메인화면 판단
def _decide_on_main(db, order_session, goal, request) -> NextTargetResponse:
    # 1. 장바구니 확인
    count = _cart_quantity(goal, request.cart_rows)
    if count >= goal.quantity:
        order_session.current_step = CurrentStep.DONE
        return _reply(order_session, request, NextTargetResult.DONE, "주문 목표를 다 담았어요.")
    if count > 0:
        for b in request.buttons:
            if b.text == "+" and b.group and split_temperature(b.group) [1] == normalize(goal.menu):
                return _target(db, order_session, request, b, CurrentStep.QUANTITY_SELECT, f"+ 버튼을 누르세요. 지금 {count}개예요.")
    
    # 2. 메뉴 찾기
    found = _find_menu_buttons(goal, request.buttons)
    if request.answer and len(found) > 1:
        want = temperature_of(request.answer)
        if want:
            goal.options = {**goal.options, "온도": want}
            found = [b for b in found if split_temperature(b.text)[0] == want]

    if len(found) == 1:
        button = found[0]                   # 첫번째 버튼
        # 품절이라면
        if button.sold_out:                 # AI가 "일시품절"이 붙은 메뉴를 True로 보내줌
            return _reply(order_session, request, NextTargetResult.SOLD_OUT, f"{button.text}는 지금 품절이에요.")
        return _target(db, order_session, request, button, CurrentStep.MENU_SELECT)
    if len(found) > 1:
        order_session.current_step = CurrentStep.MENU_SELECT
        return _reply(order_session, request, NextTargetResult.NEED_CHOICE, "아이스와 따뜻한 중 어떤 걸로 할까요?", choices=[b.text for b in found])   # 버튼 글자 목록

    # 3. 메뉴가 없을 때
    tabs_seen, pages = _browse_history(db, order_session.id)
    # 3-1. (사전 힌트: kiosk_menu.json 받은 뒤 추가)

    # 3-2. 이 탭에서 아직 덜 넘겼으면 다음 페이지
    if pages < MAX_PAGES:
        for b in request.buttons:
            if "다음" in b.text:
                return _target(db, order_session, request, b, CurrentStep.PAGE_NEXT, "이 화면에 없어서 다음 페이지로 넘기겠습니다.")

    # 3-3. 아직 안 본 탭
    for tab in _tab_buttons(request.buttons):
        if tab.text not in tabs_seen:
            return _target(db, order_session, request, tab, CurrentStep.CATEGORY_SELECT, f"{tab.text} 탭을 누르세요.")
    # 3-4. 다 봤는데 없음
    return _reply(order_session, request, NextTargetResult.NOT_FOUND, "이 화면에서 메뉴를 못 찾았어요. 다른 카테고리 탭을 눌러주세요. ")

# 메뉴를 고른 뒤에 이미 안내한 옵션 묶음들
def _guided_groups(db, session_id) -> set[str]:
    steps = db.scalars(
        select(GuideStep).where(GuideStep.session_id == session_id).order_by(GuideStep.id)
    ).all()

    groups = set()
    for s in steps:
        if s.step == CurrentStep.MENU_SELECT:
            groups = set()
        elif s.step == CurrentStep.OPTION_SELECT and s.target_group:
            groups.add(s.target_group)                     # 안내했던 묶음 이름
    return groups

# 옵션 팝업 판단
def _decide_on_popup(db, order_session, goal, request) -> NextTargetResponse:
    groups = []
    for b in request.buttons:
        if b.group and b.group not in groups:
            groups.append(b.group)
    guided = _guided_groups(db, order_session.id)

    for group in groups:
        key = group_name(group)
        if not is_required(group) and key not in goal.options:
            continue
        if group in guided: continue

        buttons = [b for b in request.buttons if b.group == group and not b.sold_out]
        texts = [b.text for b in buttons]

        # 아직 고른 값이 없으면 대답으로 고르거나, 물어보기
        if key not in goal.options:
            chosen = match_choice(request.answer, texts) if request.answer else None
            if chosen is None:
                order_session.current_step = CurrentStep.OPTION_SELECT
                return _reply(order_session, request, NextTargetResult.NEED_CHOICE, f"{key}을 골라 주세요.", choices=texts) 
            
            goal.options = {**goal.options, key:chosen}
        # 고릅 값에 맞는 버튼 안내
        picked = match_choice(goal.options[key], texts)
        for b in buttons:
            if b.text == picked:
                return _target(db, order_session, request, b, CurrentStep.OPTION_SELECT)

    # 다 골랐으면 선택완료
    for b in request.buttons:
        if b.text == "선택완료":
            return _target(db, order_session, request, b, CurrentStep.OPTION_SELECT)            

    return _reply(order_session, request, NextTargetResult.NOT_FOUND, "선택완료 버튼을 못 찾았어요.")


def decide_next_target(db: Session, session_id: uuid.UUID, request: NextTargetRequest) -> NextTargetResponse:
    # 끝난 세션인지 확인
    order_session = get_session(db, session_id)
    if order_session.status in ENDED_STATUSES:
        raise SessionAlreadyEnded
    # 주문 목표가 확정되었는지 확인
    goal = _find_goal(db, session_id)
    if goal is None or goal.status != GoalStatus.CONFIRMED:
        raise GoalNotConfirmed
    # 늦게 도착한 옛날 화면이면 부시하기
    last = order_session.last_screen_seq
    if last is not None and request.screen_seq < last:
        raise StaleScreen
    order_session.last_screen_seq = request.screen_seq

    # 옵션 팝업/메인 화면 나눠서 판단하기
    if _is_option_popup(request.buttons):
        response = _decide_on_popup(db, order_session, goal, request)
    else:
        response = _decide_on_main(db, order_session, goal, request)

    log_event(db, session_id, "NEXT_TARGET", {"screenSeq": request.screen_seq, "result": response.result.value})
    db.commit()
    return response

######

# 가장 마지막으로 안내한 버튼
def _last_step(db, session_id) -> GuideStep | None:
    return db.scalars(
        select(GuideStep).where(GuideStep.session_id == session_id).order_by(GuideStep.id.desc())
    ).first()

# 눌렀는지 판단
def _judge(goal, step, request) -> SelectionResult:
    # 메뉴 버튼: 옵션 팝업이 떴거나 장바구니에 담겼으면 성공
    if step.step == CurrentStep.MENU_SELECT:
        if _is_option_popup(request.buttons) or _cart_quantity(goal, request.cart_rows) > 0:
            return SelectionResult.SUCCESS
        if request.screen_changed:
            return SelectionResult.WRONG_SELECTION
        return SelectionResult.UNCONFIRMED
    
    # 그 외 버튼: 화면이 바뀌면 성공
    if request.screen_changed:
        return SelectionResult.SUCCESS
    return SelectionResult.UNCONFIRMED

MESSAGES = {
    SelectionResult.SUCCESS: "{text}을(를) 눌렀어요.",
    SelectionResult.WRONG_SELECTION: "다른 버튼이 눌린 것 같아요. 화면을 다시 확인할게요.",
    SelectionResult.UNCONFIRMED: "눌렸는지 확인하지 못했어요. 다음으로 넘어갈게요.",
}

def check_selection_result(db: Session, session_id: uuid.UUID, request: SelectionResultRequest) -> SelectionResultResponse:
    # 끝난 세션인지 확인
    order_session = get_session(db, session_id)
    if order_session.status in ENDED_STATUSES:
        raise SessionAlreadyEnded
    # 늦게 도착한 옛날 화면이면 무시하기
    last = order_session.last_screen_seq
    if last is not None and request.screen_seq < last:
        raise StaleScreen
    order_session.last_screen_seq = request.screen_seq

    # 안내한 버튼이 맞는지 확인
    step = _last_step(db, session_id)
    if step is None or step.target_button_id != request.target_button_id:
        raise NoGuidedStep

    goal = _find_goal(db, session_id)
    result = _judge(goal, step, request)
    step.result = result

    log_event(db, session_id, "SELECTION_RESULT", {"screenSeq": request.screen_seq, "result": result.value})
    db.commit()
    return SelectionResultResponse(
        result=result,
        current_step=order_session.current_step,
        message=MESSAGES[result].format(text=step.target_text),
    )

TAB_GROUP = "카테고리"
MAX_PAGES = 3
# 화면 버튼 중 카테고리 탭만
def _tab_buttons(buttons) -> list[Button]:
    return [b for b in buttons if b.group == TAB_GROUP]

# 지금까지 안내한 탭들과, 마지막 탭에서 다음 ▶를 몇 번 눌렀는지
def _browse_history(db, session_id) -> tuple[set[str], int]:
    steps = db.scalars(
        select(GuideStep).where(GuideStep.session_id == session_id).order_by(GuideStep.id)
    ).all()

    tabs = set()
    pages = 0
    for s in steps:
        if s.step == CurrentStep.CATEGORY_SELECT:
            tabs.add(s.target_text)
            pages = 0                       # 탭을 옮기면 다시 1페이지
        elif s.step == CurrentStep.PAGE_NEXT:
            pages += 1
    return tabs, pages

