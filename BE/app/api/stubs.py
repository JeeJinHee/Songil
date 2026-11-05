import uuid

from fastapi import APIRouter, HTTPException, status
router = APIRouter(prefix="/api/sessions/{sessionId}")

def not_implemented():
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "아직 구현되지 않은 API입니다.")

# BE-03 주문 진행 관리


# BE-04 오류·복구 관리
@router.post("/errors", tags=["BE-04 오류·복구"], summary="BE-04-01 기본 오류 처리")
def report_error(sessionId: uuid.UUID):
    not_implemented()

@router.post("/recovery", tags=["BE-04 오류·복구"], summary="BE-04-02 오선택 복구 판단 (2차)")
def decide_recovery(sessionId: uuid.UUID):
    not_implemented()


# BE-05 장바구니 확인
@router.post("/cart/verify", tags=["BE-05 장바구니 확인"], summary="BE-05-01 장바구니 검증")
def verify_cart(sessionId: uuid.UUID):
    not_implemented()
