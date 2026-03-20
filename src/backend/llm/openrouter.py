"""
Асинхронный стриминговый клиент для OpenRouter API.
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Системный промпт: объясняем модели, что токены-заглушки нельзя раскрывать
SYSTEM_PROMPT = (
    "Ты полезный ассистент банка Альфа-Банк. "
    "В сообщениях пользователя некоторые персональные данные заменены "
    "токенами-заглушками вида [ФИО_1], [ТЕЛЕФОН_1] и т.д. "
    "Используй эти токены в своих ответах так же, как они были переданы — "
    "не раскрывай исходные данные и не придумывай значения. "
    "Отвечай на русском языке."
)


async def stream_chat(
    messages: list[dict],
    api_key: str,
    model: str = "openai/gpt-4o-mini",
) -> AsyncIterator[str]:
    """
    Стримит текстовые токены от LLM по мере их поступления.
    Выбрасывает httpx.HTTPStatusError при ответе не 2xx.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Alfa Bank PII Proxy",
    }

    # Добавляем системный промпт в начало истории диалога
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    payload = {
        "model": model,
        "messages": full_messages,
        "stream": True,
    }

    # Буфер для склейки неполных SSE-фреймов между чанками
    buffer = ""

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST", OPENROUTER_URL, headers=headers, json=payload
        ) as response:
            response.raise_for_status()

            async for raw_chunk in response.aiter_bytes():
                buffer += raw_chunk.decode("utf-8", errors="replace")

                # SSE-сообщения разделяются двойным переносом строки
                while "\n\n" in buffer:
                    message, buffer = buffer.split("\n\n", 1)
                    for line in message.splitlines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"]
                            content = delta.get("content") or ""
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            # Пропускаем некорректные или служебные фреймы
                            continue
