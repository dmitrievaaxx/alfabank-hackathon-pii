# ── Этап 1: установка зависимостей через uv ──────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

# Копируем только манифест зависимостей — слой кешируется до изменения pyproject.toml
COPY pyproject.toml ./

# Устанавливаем зависимости в изолированное виртуальное окружение
# --no-dev — пропускаем dev-зависимости
# --index-strategy unsafe-best-match — нужно для корректного выбора CPU-pytorch
RUN uv sync --no-dev --index-strategy unsafe-best-match

# ── Этап 2: финальный образ ───────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Копируем готовое виртуальное окружение из этапа сборки
COPY --from=builder /app/.venv /app/.venv

# Копируем исходный код
COPY src/ src/

# Добавляем venv в PATH
ENV PATH="/app/.venv/bin:$PATH"

# Кеш HuggingFace — модель (~700 МБ) хранится в volume, не в образе
ENV HF_HOME=/cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
