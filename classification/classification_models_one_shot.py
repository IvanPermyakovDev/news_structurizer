from pathlib import Path
import json
from functools import lru_cache
from typing import Dict, List, Sequence
from tqdm import tqdm
from transformers import pipeline
import pandas as pd

# Фиксированные метки (строго из ТЗ)
TOPIC_ALLOWED = {
    "политика",
    "спорт",
    "экономика",
    "технологии и наука",
    "культура и искусство",
    "экология и климат",
    "mixed",
}
SCALE_ALLOWED = {"local", "global"}

# Кандидаты для zero-shot по topic (без "mixed"; mixed выводим логикой)
TOPIC_CANDIDATES = [
    "политика",
    "спорт",
    "экономика",
    "технологии и наука",
    "культура и искусство",
    "экология и климат",
]

# Порог для решения "mixed": если отрыв топ-1 от топ-2 меньше — считаем сюжет смешанным
MIXED_MARGIN = 0.06   # можно варьировать 0.05–0.10
TOPN_FOR_MIXED = 3    # рассматриваем ничью в топ-3

TOPIC_TEMPLATE = "Этот текст относится к категории: {}."
SCALE_TEMPLATE = "Масштаб описываемых событий: {}."
SCALE_LABELS = [
    ("локальная", "local"),
    ("глобальная", "global"),
]

# Подборка многоязычных zero-shot моделей, которые можно быстро сравнить
ZERO_SHOT_MODELS = {
    # ключ -> конфигурация: huggingface id + (опционально) переопределённые шаблоны
    "mdeberta-v3-base": {
        "model_id": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "description": "mDeBERTa-v3 base (MoritzLaurer)",
    },
    "xlm-roberta-large": {
        "model_id": "joeddav/xlm-roberta-large-xnli",
        "description": "XLM-R large NLI (joeddav)",
    },
    "distilbert-mnli-multilingual": {
        "model_id": "typeform/distilbert-base-multilingual-cased-mnli",
        "description": "DistilBERT multilingual MNLI (Typeform)",
    },
    "deberta-v3-large": {
        "model_id": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling",
        "description": "DeBERTa-v3 large multilingual (MoritzLaurer)",
    },
}

DEFAULT_MODEL_KEY = "mdeberta-v3-base"
DEFAULT_COMPARISON_MODELS: List[str] = [
    "mdeberta-v3-base",
    "xlm-roberta-large",
    "deberta-v3-large",
    "distilbert-mnli-multilingual",
]


@lru_cache(maxsize=None)
def _get_zero_shot_pipeline(model_id: str):
    """HF pipeline с кешированием по имени модели."""
    return pipeline(task="zero-shot-classification", model=model_id, device=None)


def _get_model_config(model_key: str):
    try:
        spec = ZERO_SHOT_MODELS[model_key]
    except KeyError as exc:
        available = ", ".join(ZERO_SHOT_MODELS)
        raise KeyError(f"Неизвестный ключ модели '{model_key}'. Доступно: {available}") from exc
    clf = _get_zero_shot_pipeline(spec["model_id"])
    topic_template = spec.get("topic_template", TOPIC_TEMPLATE)
    scale_template = spec.get("scale_template", SCALE_TEMPLATE)
    return {
        "key": model_key,
        "model_id": spec["model_id"],
        "description": spec.get("description", spec["model_id"]),
        "classifier": clf,
        "topic_template": topic_template,
        "scale_template": scale_template,
    }

def _predict_topic(text: str, classifier, hypothesis_template: str) -> Dict[str, float]:
    res = classifier(
        text,
        TOPIC_CANDIDATES,
        multi_label=False,
        hypothesis_template=hypothesis_template,
    )
    return dict(zip(res["labels"], map(float, res["scores"])))

def _choose_topic(scores: Dict[str, float]) -> str:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = ordered[0]
    if len(ordered) >= 2:
        second_score = ordered[1][1]
        if (top_score - second_score) < MIXED_MARGIN:
            return "mixed"
    return top_label

def _predict_scale(text: str, classifier, hypothesis_template: str) -> str:
    scale_candidates = [cand for cand, _ in SCALE_LABELS]
    res = classifier(
        text,
        scale_candidates,
        multi_label=False,
        hypothesis_template=hypothesis_template,
    )
    label_ru = res["labels"][0]
    mapping = dict(SCALE_LABELS)
    if label_ru not in mapping:
        raise ValueError(f"Неожиданная метка масштаба: {label_ru!r}")
    return mapping[label_ru]

def classify_record(
    record: Dict,
    model_key: str = DEFAULT_MODEL_KEY,
    model_config: Dict | None = None,
) -> Dict:
    assert isinstance(record.get("text", None), str), "Поле 'text' должно быть строкой"
    text = record["text"]  # подаём как есть
    cfg = model_config or _get_model_config(model_key)
    classifier = cfg["classifier"]
    topic_scores = _predict_topic(text, classifier, cfg["topic_template"])
    topic = _choose_topic(topic_scores)
    scale = _predict_scale(text, classifier, cfg["scale_template"])
    return {
        "topic": topic,
        "scale": scale,
        "topic_scores": topic_scores,
        "model_key": model_key,
        "model_id": cfg["model_id"],
    }

def evaluate_json(
    json_path: str,
    show_examples: int = 5,
    limit: int | None = None,
    model_key: str = DEFAULT_MODEL_KEY,
):
    model_cfg = _get_model_config(model_key)
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert isinstance(data, list), "Корневой JSON должен быть списком объектов"
    if limit is not None:
        assert isinstance(limit, int) and limit > 0, "'limit' должен быть положительным целым числом"
        data = data[:limit]

    n = len(data)
    assert n > 0, "Датасет пуст после применения лимита"

    topic_correct = 0
    scale_correct = 0
    both_correct = 0

    mismatches_rows = []

    print(
        f"\n=== Модель: {model_cfg['description']}\n    HF id: {model_cfg['model_id']} ==="
    )

    for i, rec in enumerate(tqdm(data, desc="Evaluating", unit="item")):
        gt_topic = rec.get("topic")
        gt_scale = rec.get("scale")
        assert gt_topic in TOPIC_ALLOWED, f"[idx={i}] Неверная эталонная topic-метка: {gt_topic}"
        assert gt_scale in SCALE_ALLOWED, f"[idx={i}] Неверная эталонная scale-метка: {gt_scale}"

        pred = classify_record(rec, model_key=model_key, model_config=model_cfg)
        p_topic, p_scale = pred["topic"], pred["scale"]

        t_ok = (p_topic == gt_topic)
        s_ok = (p_scale == gt_scale)
        if t_ok: topic_correct += 1
        if s_ok: scale_correct += 1
        if t_ok and s_ok: both_correct += 1
        else:
            if len(mismatches_rows) < show_examples:
                mismatches_rows.append({
                    "idx": i,
                    "title": rec.get("title", None),
                    "gt_topic": gt_topic, "pred_topic": p_topic,
                    "gt_scale": gt_scale, "pred_scale": p_scale,
                })

    # численные метрики
    topic_acc = topic_correct / n
    scale_acc = scale_correct / n
    exact_match = both_correct / n

    # красивая табличка: счётчики + проценты + бар
    metrics_df = pd.DataFrame(
        [
            {"Метрика": "Topic accuracy",     "Accuracy": topic_acc,  "Верно": topic_correct, "Всего": n},
            {"Метрика": "Scale accuracy",     "Accuracy": scale_acc,  "Верно": scale_correct, "Всего": n},
            {"Метрика": "Exact match (both)", "Accuracy": exact_match,"Верно": both_correct,  "Всего": n},
        ]
    )
    metrics_df["Model"] = model_cfg["description"]
    metrics_df["Model id"] = model_cfg["model_id"]
    metrics_df = metrics_df[["Model", "Model id", "Метрика", "Accuracy", "Верно", "Всего"]]
    metrics_df["Accuracy %"] = (metrics_df["Accuracy"] * 100).round(2)

    display(
        metrics_df.style
        .hide(axis="index")
        .format({"Accuracy": "{:.4f}", "Accuracy %": "{:.2f}"})
        .bar(subset=["Accuracy"], color=None, vmin=0, vmax=1)  # горизонтальный бар по accuracy
    )

    # краткое резюме в тексте (наглядно копируется в отчёты)
    print(
        f"\nИтоги ({model_cfg['description']}) на {n} примерах"
        f"\n  • Topic accuracy:     {topic_correct}/{n}  ({topic_acc:.2%})"
        f"\n  • Scale accuracy:     {scale_correct}/{n}  ({scale_acc:.2%})"
        f"\n  • Exact match (both): {both_correct}/{n}  ({exact_match:.2%})"
    )

    if mismatches_rows:
        print(f"\nПримеры расхождений (первые {len(mismatches_rows)}):")
        mismatches_df = pd.DataFrame(
            mismatches_rows, columns=["idx","title","gt_topic","pred_topic","gt_scale","pred_scale"]
        )
        display(mismatches_df)

    return metrics_df


def evaluate_models(
    json_path: str,
    model_keys: Sequence[str] | None = None,
    show_examples: int = 5,
    limit: int | None = None,
    display_summary: bool = True,
):
    """Прогоняет сравнение нескольких zero-shot моделей подряд.

    Возвращает объединённый датафрейм со строками по (модель, метрика).
    """

    if model_keys is None:
        model_keys = DEFAULT_COMPARISON_MODELS

    summary_frames: List[pd.DataFrame] = []

    for key in model_keys:
        metrics_df = evaluate_json(
            json_path=json_path,
            show_examples=show_examples,
            limit=limit,
            model_key=key,
        )
        enriched = metrics_df.copy()
        enriched["Model key"] = key
        summary_frames.append(enriched)

    if not summary_frames:
        return pd.DataFrame()

    summary_df = pd.concat(summary_frames, ignore_index=True)

    if display_summary:
        pivot = (
            summary_df
            .pivot_table(
                index=["Model key", "Model", "Model id"],
                columns="Метрика",
                values="Accuracy",
                aggfunc="first",
            )
            .sort_index()
        )
        display(
            pivot.style
            .format("{:.4f}")
            .set_caption("Сравнение точности по моделям")
        )

    return summary_df


# пример запуска (сравнение всех преднастроенных моделей):
_ = evaluate_models("./dataset.json", show_examples=5)
