import enum, uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Uuid, Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

class SessionStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"

ENDED_STATUSES = {SessionStatus.COMPLETED, SessionStatus.CANCELLED, SessionStatus.FAILED}

class CurrentStep(str, enum.Enum):
    ORDER_INPUT = "ORDER_INPUT"
    CATEGORY_SELECT = "CATEGORY_SELECT"
    MENU_SELECT = "MENU_SELECT"
    OPTION_SELECT = "OPTION_SELECT"
    QUANTITY_SELECT = "QUANTITY_SELECT"
    CART_CHECK = "CART_CHECK"
    DONE = "DONE"

class EndReason(str, enum.Enum):
    CART_VERIFIED = "CART_VERIFIED"
    USER_CANCELLED = "USER_CANCELLED"
    ERROR = "ERROR"

# 종료 사유 → 종료 후 Session 상태
END_REASON_TO_STATUS = {
    EndReason.CART_VERIFIED: SessionStatus.COMPLETED,
    EndReason.USER_CANCELLED: SessionStatus.CANCELLED,
    EndReason.ERROR: SessionStatus.FAILED,
}

class OrderSession(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus, native_enum=False, length=20), default=SessionStatus.ACTIVE
    )
    current_step: Mapped[CurrentStep] = mapped_column(
        Enum(CurrentStep, native_enum=False, length=30), default=CurrentStep.ORDER_INPUT
    )
    end_reason: Mapped[EndReason | None] = mapped_column(
        Enum(EndReason, native_enum=False, length=20), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    last_screen_seq: Mapped[int | None] = mapped_column(Integer, nullable=True)