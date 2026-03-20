"""
Конфигурация приложения — читается из переменных окружения и файла .env.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Загружаем .env из корня проекта (на два уровня выше backend/)
_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_env_path)


class Settings:
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "")
    ner_model_id: str = os.getenv("NER_MODEL_ID", "dancessa/ner-alpha-bank")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))


settings = Settings()
