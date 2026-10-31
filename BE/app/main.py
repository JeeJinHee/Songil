import logging

from fastapi import FastAPI

from app.api import sessions, stubs
from app.core.config import settings
from app.core.logging import RequestLogMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title=settings.app_name, description="손길 — 시각장애인 키오스크 손끝 안내 AI 백엔드")
app.add_middleware(RequestLogMiddleware)

app.include_router(sessions.router)
app.include_router(stubs.router)


@app.get("/health", tags=["공통"], summary="서버 상태 확인")
def health():
    return {"status": "ok"}
