import enum, uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, Integer, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base
from app.models.session import utcnow

class GoalStatus(str, enum.Enum):
    NEED_CONFIRM = "NEED_CONFIRM"
    CONFIRMED = "CONFIRMED"
    BROWSING = "BROWSING"

class OrderGoal(Base): # 사용자가 말한 주문 목표 (Session당 1개)
    __tablename__ = "order_goals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("sessions.id"), unique=True, nullable=False
    )

    menu: Mapped[str | None] = mapped_column(String(100), nullable=True)
    options: Mapped[dict] = mapped_column(JSON, default=dict)
    quantity: Mapped[int] = mapped_column(Integer, default=1)

    status: Mapped[GoalStatus] = mapped_column(
        Enum(GoalStatus, native_enum=False, length=20), default=GoalStatus.NEED_CONFIRM
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
