import uuid

from sqlalchemy.orm import Session

from app.models import Event, OrderSession
from app.models.session import END_REASON_TO_STATUS, ENDED_STATUSES, EndReason, utcnow


class SessionNotFound(Exception):
    pass


class SessionAlreadyEnded(Exception):
    pass


def log_event(db: Session, session_id: uuid.UUID | None, event_type: str, payload: dict | None = None) -> None:
    db.add(Event(session_id=session_id, event_type=event_type, payload=payload))


def create_session(db: Session) -> OrderSession:
    order_session = OrderSession()
    db.add(order_session)
    db.flush()
    log_event(db, order_session.id, "SESSION_CREATED")
    db.commit()
    return order_session


def get_session(db: Session, session_id: uuid.UUID) -> OrderSession:
    order_session = db.get(OrderSession, session_id)
    if order_session is None:
        raise SessionNotFound
    return order_session


def end_session(db: Session, session_id: uuid.UUID, end_reason: EndReason) -> OrderSession:
    """앱의 안내 종료일 뿐, 키오스크 주문 취소와는 별개다."""
    order_session = get_session(db, session_id)
    if order_session.status in ENDED_STATUSES:
        raise SessionAlreadyEnded

    order_session.status = END_REASON_TO_STATUS[end_reason]
    order_session.end_reason = end_reason
    order_session.ended_at = utcnow()
    log_event(db, order_session.id, "SESSION_ENDED", {"endReason": end_reason.value})
    db.commit()
    return order_session
