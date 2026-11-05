import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.guide import NextTargetRequest, NextTargetResponse
from app.services import guide_service
from app.services.guide_service import GoalNotConfirmed, StaleScreen
from app.services.session_service import SessionAlreadyEnded, SessionNotFound

router = APIRouter(prefix="/api/sessions/{sessionId}", tags=["BE-03 주문 진행"])

@router.post(
    "/next-target",
    response_model=NextTargetResponse,
    responses={
        404: {"description": "Session이 존재하지 않음"},
        409: {"description": "종료된 Sesseion, 확정 전 목표 또는 옛날 화면"},
    },
)
def decide_next_target(sessionId: uuid.UUID, body: NextTargetRequest, db: Session = Depends(get_db)):
    try:
        return guide_service.decide_next_target(db, sessionId, body)
    except SessionNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session이 존재하지 않습니다.")
    except SessionAlreadyEnded:
        raise HTTPException(status.HTTP_409_CONFLICT, "이미 종료된 Session입니다.")
    except GoalNotConfirmed:
        raise HTTPException(status.HTTP_409_CONFLICT, "주문 목표가 아직 확정되지 않았습니다.")
    except StaleScreen:
        raise HTTPException(status.HTTP_409_CONFLICT, "이미 더 최신 화면을 처리했습니다.")


