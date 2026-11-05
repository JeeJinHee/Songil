"""BE-03 ~ BE-05: 경로만 먼저 열어 둔 API (아직 미구현, 501 반환).

세부 명세가 확정되면 그룹별 파일(goals.py, progress.py 등)로 옮겨 구현한다.
"""

import uuid

from fastapi import APIRouter, HTTPException, status

router = APIRouter(prefix="/api/sessions/{sessionId}")


def not_implemented():
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "아직 구현되지 않은 API입니다.")


# BE-03 주문 진행 관리 — 명세는 키오스크 조사(D-01)·데이터 계약(D-08) 후 확정
@router.post("/next-target", tags=["BE-03 주문 진행"], summary="BE-03-01 다음 목표 버튼 결정")
def decide_next_target(sessionId: uuid.UUID):
    not_implemented()


@router.post("/selection-result", tags=["BE-03 주문 진행"], summary="BE-03-02 선택 결과 확인·상태 갱신")
def check_selection_result(sessionId: uuid.UUID):
    not_implemented()


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
