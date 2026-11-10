import enum, uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Uuid, Enum
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base
from app.models.session import CurrentStep, utcnow

class SelectionResult(str, enum.Enum):
    SUCCESS = "SUCCESS"
    WRONG_SELECTION = "WRONG_SELECTION"
    UNCONFIRMED = "UNCONFIRMED"

class GuideStep(Base):
    __tablename__ = "guide_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("sessions.id"), nullable=False, index=True
    )

    screen_seq: Mapped[int] = mapped_column(Integer)
    step: Mapped[CurrentStep] = mapped_column(
        Enum(CurrentStep, native_enum=False, length=30)
    )

    target_button_id: Mapped[str] = mapped_column(String(50))
    target_text: Mapped[str] = mapped_column(String(200))
    target_group: Mapped[str | None] = mapped_column(String(100), nullable=True)

    result: Mapped[SelectionResult | None] = mapped_column(
        Enum(SelectionResult, native_enum = False, length=20), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
