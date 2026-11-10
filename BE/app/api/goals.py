import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.goal import (
    GoalConfirmRequest,
    GoalConfirmResponse,
    GoalCreateRequest,
    GoalCreateResponse,
    GoalResponse,
)
from app.services import goal_service
from app.services.goal_service import GoalAlreadyConfirmed, GoalNotFound, GoalNotReady
from app.services.session_service import SessionAlreadyEnded, SessionNotFound

router = APIRouter(prefix="/api/sessions/{sessionId}/goal", tags=["BE-02 주문 목표"])

SESSION_NOT_FOUND = "Session이 존재하지 않습니다."
SESSION_ENDED = "이미 종료된 Session입니다."


@router.post(
    "",
    response_model=GoalCreateResponse,
    responses={
        404: {"description": "Session이 존재하지 않음"},
        409: {"description": "종료된 Session 또는 이미 확정된 목표"},
    },
)
def create_goal(sessionId: uuid.UUID, body: GoalCreateRequest, db: Session = Depends(get_db)):
    try:
        return goal_service.create_goal(db, sessionId, body.text)
    except SessionNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, SESSION_NOT_FOUND)
    except SessionAlreadyEnded:
        raise HTTPException(status.HTTP_409_CONFLICT, SESSION_ENDED)
    except GoalAlreadyConfirmed:
        raise HTTPException(status.HTTP_409_CONFLICT, "이미 확정된 목표가 있습니다.")


@router.post(
    "/confirm",
    response_model=GoalConfirmResponse,
    responses={
        404: {"description": "Session이 존재하지 않음"},
        409: {"description": "확정할 목표가 없음 또는 종료된 Session"},
    },
)
def confirm_goal(sessionId: uuid.UUID, body: GoalConfirmRequest, db: Session = Depends(get_db)):
    try:
        return goal_service.confirm_goal(db, sessionId, body.action)
    except SessionNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, SESSION_NOT_FOUND)
    except SessionAlreadyEnded:
        raise HTTPException(status.HTTP_409_CONFLICT, SESSION_ENDED)
    except GoalNotReady:
        raise HTTPException(status.HTTP_409_CONFLICT, "확정할 목표가 없습니다.")


@router.get(
    "",
    response_model=GoalResponse,
    responses={404: {"description": "Session 또는 목표가 존재하지 않음"}},
)
def get_goal(sessionId: uuid.UUID, db: Session = Depends(get_db)):
    try:
        return goal_service.get_goal(db, sessionId)
    except SessionNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, SESSION_NOT_FOUND)
    except GoalNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "저장된 목표가 없습니다.")
