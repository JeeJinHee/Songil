import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.session import SessionEndRequest, SessionEndResponse, SessionResponse
from app.services import session_service
from app.services.session_service import SessionAlreadyEnded, SessionNotFound

router = APIRouter(prefix="/api/sessions", tags=["BE-01 Session 관리"])


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_session(db: Session = Depends(get_db)):
    return session_service.create_session(db)

@router.get(
    "/{sessionId}",
    response_model=SessionResponse,
    responses={404: {"description": "Session이 존재하지 않음"}},
)
def get_session(sessionId: uuid.UUID, db: Session = Depends(get_db)):
    try:
        return session_service.get_session(db, sessionId)
    except SessionNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session이 존재하지 않습니다.")

@router.post(
    "/{sessionId}/end",
    response_model=SessionEndResponse,
    responses={
        404: {"description": "Session이 존재하지 않음"},
        409: {"description": "이미 종료된 Session"},
    },
)
def end_session(sessionId: uuid.UUID, body: SessionEndRequest, db: Session = Depends(get_db)):
    try:
        return session_service.end_session(db, sessionId, body.end_reason)
    except SessionNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session이 존재하지 않습니다.")
    except SessionAlreadyEnded:
        raise HTTPException(status.HTTP_409_CONFLICT, "이미 종료된 Session입니다.")
