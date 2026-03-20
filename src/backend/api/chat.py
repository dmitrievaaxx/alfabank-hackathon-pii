"""
POST /api/chat — SSE-эндпоинт стриминга ответа LLM с маскированием ПДн.

Последовательность событий (протокол совместим с фронтендом):
  1. data: {"type": "analysis", "spans": [...], "masked": "...", "entities": [...]}
  2. data: {"type": "token", "content": "..."}   ← сырой маскированный токен от LLM
  3. data: {"type": "done"}                       ← завершение стрима
  4. data: [DONE]
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..config import settings
from ..llm.openrouter import stream_chat
from ..ner.masker import mask_text
from ..ner.model import predict_spans

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    # История диалога содержит маскированные версии сообщений
    history: list[dict] = []


@router.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    if not settings.openrouter_api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY не настроен. Добавьте его в файл .env.",
        )
    if not settings.openrouter_model:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_MODEL не настроен. Добавьте его в файл .env.",
        )

    # Маскируем ПДн и получаем хранилище токен → оригинал
    masked_text, vault = mask_text(req.message)
    entities = predict_spans(req.message)

    # Формируем spans для клиентского демаскирования
    # span.entity — ключ без скобок, например "ФИО_1"
    # фронтенд сам заменяет [ФИО_1] → оригинальное значение
    spans = [
        {"entity": token[1:-1], "text": value}
        for token, value in vault.items()
    ]

    # Маскированный запрос уходит в LLM
    llm_messages = req.history + [{"role": "user", "content": masked_text}]

    async def event_stream():
        # ── Событие 1: результаты NER-анализа ────────────────────────────
        analysis_event = {
            "type": "analysis",
            "spans": spans,          # для демаскирования на клиенте
            "masked": masked_text,   # маскированный текст запроса
            "entities": entities,    # подробности найденных сущностей
        }
        yield f"data: {json.dumps(analysis_event, ensure_ascii=False)}\n\n"

        # ── События 2..N: сырые маскированные токены от LLM ──────────────
        # Токены передаются без восстановления — фронтенд показывает маски
        # в реальном времени, а затем демаскирует всё сразу по завершении
        try:
            async for raw_token in stream_chat(
                messages=llm_messages,
                api_key=settings.openrouter_api_key,
                model=settings.openrouter_model,
            ):
                token_event = {"type": "token", "content": raw_token}
                yield f"data: {json.dumps(token_event, ensure_ascii=False)}\n\n"

            # Завершение стрима
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        except Exception as exc:  # noqa: BLE001
            logger.exception("Ошибка стриминга LLM")
            err_event = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(err_event, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
