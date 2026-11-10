import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import OrderGoal, OrderSession
from app.models.goal import GoalStatus
from app.models.session import ENDED_STATUSES, CurrentStep, utcnow
from app.schemas.goal import (
    Goal,
    GoalConfirmAction,
    GoalConfirmResponse,
    GoalCreateResponse,
    GoalResponse,
    GoalResultStatus,
)
from app.services.goal_parser import parse_goal
from app.services.session_service import SessionAlreadyEnded, get_session, log_event


class GoalNotFound(Exception):
    pass


class GoalAlreadyConfirmed(Exception):
    pass


class GoalNotReady(Exception):
    pass


def _get_active_session(db: Session, session_id: uuid.UUID) -> OrderSession:
    order_session = get_session(db, session_id)
    if order_session.status in ENDED_STATUSES:
        raise SessionAlreadyEnded
    return order_session


def _find_goal(db: Session, session_id: uuid.UUID) -> OrderGoal | None:
    return db.scalar(select(OrderGoal).where(OrderGoal.session_id == session_id))


def _confirm_message(goal: Goal) -> str:
    words = [*goal.options.values(), goal.menu, f"{goal.quantity}개"]
    return f"{' '.join(words)} 맞나요?"


def create_goal(db: Session, session_id: uuid.UUID, text: str) -> GoalCreateResponse:
    _get_active_session(db, session_id)
    order_goal = _find_goal(db, session_id)
    if order_goal is not None and order_goal.status == GoalStatus.CONFIRMED:
        raise GoalAlreadyConfirmed

    parsed = parse_goal(text)
    if parsed is None:
        log_event(db, session_id, "GOAL_NOT_UNDERSTOOD", {"text": text})
        db.commit()
        return GoalCreateResponse(
            goal=None,
            goal_status=GoalResultStatus.NOT_UNDERSTOOD,
            message="잘 알아듣지 못했어요. 메뉴를 다시 말씀해 주세요.",
        )

    if order_goal is None:
        order_goal = OrderGoal(session_id=session_id)
        db.add(order_goal)
    order_goal.menu = parsed.menu
    order_goal.options = parsed.options
    order_goal.quantity = parsed.quantity
    order_goal.status = GoalStatus.NEED_CONFIRM

    log_event(db, session_id, "GOAL_PARSED", {"text": text, "goal": parsed.model_dump()})
    db.commit()
    return GoalCreateResponse(
        goal=parsed,
        goal_status=GoalResultStatus.NEED_CONFIRM,
        message=_confirm_message(parsed),
    )


def confirm_goal(db: Session, session_id: uuid.UUID, action: GoalConfirmAction) -> GoalConfirmResponse:
    order_session = _get_active_session(db, session_id)
    order_goal = _find_goal(db, session_id)

    if action == GoalConfirmAction.CONFIRM:
        if order_goal is None or order_goal.status != GoalStatus.NEED_CONFIRM:
            raise GoalNotReady
        order_goal.status = GoalStatus.CONFIRMED
        order_goal.confirmed_at = utcnow()
        order_session.current_step = CurrentStep.MENU_SELECT
        result = GoalResultStatus.CONFIRMED
        message = "주문을 시작할게요. 키오스크 화면을 비춰 주세요."

    elif action == GoalConfirmAction.RETRY:
        if order_goal is not None:
            db.delete(order_goal)
        order_session.current_step = CurrentStep.ORDER_INPUT
        result = GoalResultStatus.CLEARED
        message = "주문하실 메뉴를 다시 말씀해 주세요."

    else:
        if order_goal is None:
            order_goal = OrderGoal(session_id=session_id)
            db.add(order_goal)
        order_goal.menu = None
        order_goal.options = {}
        order_goal.quantity = 1
        order_goal.status = GoalStatus.BROWSING
        order_session.current_step = CurrentStep.MENU_SELECT
        result = GoalResultStatus.BROWSING
        message = "화면에 있는 메뉴를 읽어 드릴게요. 키오스크 화면을 비춰 주세요."

    log_event(db, session_id, "GOAL_" + result.value)
    db.commit()
    return GoalConfirmResponse(
        goal_status=result, current_step=order_session.current_step, message=message
    )


def get_goal(db: Session, session_id: uuid.UUID) -> GoalResponse:
    get_session(db, session_id)
    order_goal = _find_goal(db, session_id)
    if order_goal is None:
        raise GoalNotFound
    return GoalResponse(
        goal=Goal.model_validate(order_goal),
        goal_status=GoalResultStatus(order_goal.status.value),
    )
