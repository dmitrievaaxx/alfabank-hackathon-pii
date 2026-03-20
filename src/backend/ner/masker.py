"""
Утилиты маскирования и восстановления персональных данных.

Схема работы:
  1. mask_text(text) → (masked_text, vault)
     "Иван Петров, тел. +7 903 123-45-67"
     → "[ФИО_1], тел. [ТЕЛЕФОН_1]",  {"[ФИО_1]": "Иван Петров", "[ТЕЛЕФОН_1]": "+7 903 123-45-67"}

  2. restore_text(llm_response, vault)
     "Добрый день, [ФИО_1]! Ваш номер [ТЕЛЕФОН_1] подтверждён."
     → "Добрый день, Иван Петров! Ваш номер +7 903 123-45-67 подтверждён."
"""
from __future__ import annotations

from collections import defaultdict

from .labels import CATEGORY_SHORT_KEYS
from .model import predict_spans


def mask_text(text: str) -> tuple[str, dict[str, str]]:
    """
    Обнаруживает ПДн в тексте, заменяет каждую сущность токеном-заглушкой.
    Возвращает (маскированный текст, хранилище токен → оригинальное значение).
    """
    entities = predict_spans(text)
    if not entities:
        return text, {}

    counters: dict[str, int] = defaultdict(int)
    vault: dict[str, str] = {}

    # Нумеруем токены слева направо для предсказуемого порядка
    sorted_fwd = sorted(entities, key=lambda e: e["start"])
    for entity in sorted_fwd:
        short = CATEGORY_SHORT_KEYS.get(entity["category"], entity["category"][:12].upper())
        counters[short] += 1
        token = f"[{short}_{counters[short]}]"
        entity["_token"] = token
        vault[token] = entity["value"]

    # Заменяем справа налево, чтобы не сдвигать символьные позиции
    result = text
    for entity in sorted(entities, key=lambda e: e["start"], reverse=True):
        result = result[: entity["start"]] + entity["_token"] + result[entity["end"]:]

    return result, vault


def restore_text(text: str, vault: dict[str, str]) -> str:
    """Восстанавливает оригинальные значения ПДн, заменяя токены-заглушки обратно."""
    for token, value in vault.items():
        text = text.replace(token, value)
    return text


def get_entities(text: str) -> list[dict]:
    """Возвращает список найденных сущностей для отображения в интерфейсе."""
    return predict_spans(text)
