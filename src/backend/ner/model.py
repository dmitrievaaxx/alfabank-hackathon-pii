"""
Синглтон NER-модели — загружается один раз при старте, переиспользуется во всех запросах.
"""
from __future__ import annotations

import logging
from threading import Lock

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .labels import id2label, label2id, labels, norm2cat

logger = logging.getLogger(__name__)

# Блокировка для потокобезопасной однократной загрузки модели
_lock = Lock()
_tokenizer = None
_model = None
_device: str = "cpu"
_loaded = False

NER_MODEL_ID = "dancessa/ner-alpha-bank"
MAX_LENGTH = 128  # максимальная длина входной последовательности в токенах


def load_model(model_id: str = NER_MODEL_ID) -> None:
    """Загружает токенизатор и модель классификации токенов. Повторный вызов игнорируется."""
    global _tokenizer, _model, _device, _loaded
    with _lock:
        if _loaded:
            return
        logger.info("Загрузка NER-модели: %s", model_id)
        _device = "cuda" if torch.cuda.is_available() else "cpu"

        # Токенизатор берём от базовой модели (WordPiece, словарь 100k токенов)
        _tokenizer = AutoTokenizer.from_pretrained(
            "DeepPavlov/rubert-base-cased-conversational"
        )
        # Дообученная модель для классификации токенов с 61 BIO-меткой
        _model = AutoModelForTokenClassification.from_pretrained(
            model_id,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        _model.eval()
        _model.to(_device)
        _loaded = True
        logger.info("NER-модель готова на устройстве: %s", _device)


def is_loaded() -> bool:
    """Возвращает True, если модель уже загружена и готова к инференсу."""
    return _loaded


def predict_spans(text: str) -> list[dict]:
    """
    Возвращает список найденных сущностей в формате:
    [{"start": int, "end": int, "category": str, "value": str}, ...]
    """
    if not _loaded:
        raise RuntimeError("NER-модель не загружена. Вызовите load_model() при старте приложения.")

    encoding = _tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    # Сохраняем маппинг позиций токенов до передачи в модель
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        outputs = _model(**{k: v.to(_device) for k, v in encoding.items()})

    # Берём метку с максимальным логитом для каждого токена
    pred_ids = outputs.logits[0].argmax(-1).tolist()

    spans: list[dict] = []
    current: dict | None = None

    for pred_id, (tok_start, tok_end) in zip(pred_ids, offset_mapping):
        # Специальные токены [CLS], [SEP], [PAD] имеют offset (0, 0) — пропускаем
        if tok_start == 0 and tok_end == 0:
            if current:
                spans.append(current)
                current = None
            continue

        label = id2label[pred_id]

        if label == "O":
            # Токен вне сущности — закрываем текущий спан если есть
            if current:
                spans.append(current)
                current = None
        elif label.startswith("B-"):
            # Начало новой сущности
            if current:
                spans.append(current)
            current = {"start": tok_start, "end": tok_end, "label": label[2:]}
        elif label.startswith("I-"):
            # Продолжение сущности — расширяем правую границу
            cat = label[2:]
            if current and current["label"] == cat:
                current["end"] = tok_end
            else:
                if current:
                    spans.append(current)
                current = {"start": tok_start, "end": tok_end, "label": cat}

    if current:
        spans.append(current)

    # Преобразуем нормализованные метки обратно в читаемые названия категорий
    return [
        {
            "start": s["start"],
            "end": s["end"],
            "category": norm2cat.get(s["label"], s["label"]),
            "value": text[s["start"]: s["end"]],
        }
        for s in spans
    ]
