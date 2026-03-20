"""
Точка входа FastAPI-приложения.

Запуск из корня проекта:
    uvicorn src.backend.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .ner.model import is_loaded, load_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# Путь к папке с фронтендом
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружаем NER-модель при старте, освобождаем ресурсы при остановке."""
    logger.info("Запуск — загрузка NER-модели (%s)…", settings.ner_model_id)
    load_model(settings.ner_model_id)
    logger.info("NER-модель готова.")
    yield
    logger.info("Приложение остановлено.")


app = FastAPI(
    title="Alfa Bank PII Proxy",
    description="Прокси-чат с детекцией, маскированием ПДн и стримингом LLM.",
    version="1.0.0",
    lifespan=lifespan,
)

# Разрешаем запросы с любого источника (для локальной разработки)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API-роуты ─────────────────────────────────────────────────────────────────
from .api.chat import router as chat_router  # noqa: E402

app.include_router(chat_router, prefix="/api")


@app.get("/api/health")
async def health() -> JSONResponse:
    """Проверка состояния сервиса и готовности модели."""
    return JSONResponse(
        {
            "status": "ok",
            "model_loaded": is_loaded(),
            "model_id": settings.ner_model_id,
            "llm_model": settings.openrouter_model,
            "llm_configured": bool(settings.openrouter_api_key),
        }
    )


# ── Раздача статики (фронтенд) ────────────────────────────────────────────────
@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
