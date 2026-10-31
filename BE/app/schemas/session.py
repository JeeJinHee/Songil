import uuid
from pydantic import Field

from app.models.session import CurrentStep, EndReason, SessionStatus
from app.schemas.common import CamelModel, UtcDatetime


class SessionResponse(CamelModel):
    """BE-01-01 생성, BE-01-02 조회 응답."""

    session_id: uuid.UUID = Field(validation_alias="id")
    session_status: SessionStatus = Field(validation_alias="status")
    current_step: CurrentStep
    created_at: UtcDatetime


class SessionEndRequest(CamelModel):
    end_reason: EndReason


class SessionEndResponse(CamelModel):
    session_id: uuid.UUID = Field(validation_alias="id")
    session_status: SessionStatus = Field(validation_alias="status")
    ended_at: UtcDatetime
