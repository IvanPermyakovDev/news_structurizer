"""
News Structurizer Pipeline
Streamlit-приложение для обработки потока новостей.

Запуск: streamlit run pipeline.py
"""

import sys
import os
from pathlib import Path

# Добавляем пути для импорта локальных модулей
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "topicsegmenter"))

import re
import inspect
from typing import List, Dict, Any

import torch
import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline as hf_pipeline,
)


# ============ Утилиты ============

def normalize_text(text: str) -> str:
    """Нормализация текста (совпадает с topicsegmenter/src/utils.py)."""
    text = text.lower()
    text = re.sub(r'[^a-zа-яё\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============ Модели ============

class NewsSegmenter:
    """Сегментатор новостей - разбивает поток текста на отдельные новости."""

    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self._accepts_token_type_ids = self._check_token_type_ids()

    def _check_token_type_ids(self) -> bool:
        try:
            return "token_type_ids" in inspect.signature(self.model.forward).parameters
        except (TypeError, ValueError):
            return False

    def _predict_batch(self, pairs: List[tuple]) -> List[float]:
        if not pairs:
            return []

        batch_size = 32
        all_probs = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            lefts = [normalize_text(p[0]) for p in batch]
            rights = [normalize_text(p[1]) for p in batch]

            inputs = self.tokenizer(
                lefts, rights,
                add_special_tokens=True,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            if not self._accepts_token_type_ids and "token_type_ids" in inputs:
                inputs.pop("token_type_ids")

            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                all_probs.extend(probs[:, 1].cpu().tolist())

        return all_probs

    def segment(self, text: str) -> List[str]:
        """Разбивает текст на отдельные новости."""
        words = text.split()

        MIN_LEN = 10
        CONFIRM_THR = 0.8

        scan_indices = list(range(MIN_LEN, len(words) - MIN_LEN))
        if not scan_indices:
            return [text]

        candidates = []
        for i in scan_indices:
            ctx_left = " ".join(words[max(0, i-50):i])
            ctx_right = " ".join(words[i:min(len(words), i+50)])
            candidates.append((ctx_left, ctx_right))

        probs = self._predict_batch(candidates)

        split_indices = [0]
        i = 0
        while i < len(probs):
            prob = probs[i]
            idx = scan_indices[i]

            is_peak = True
            if i > 0 and probs[i-1] >= prob:
                is_peak = False
            if i < len(probs) - 1 and probs[i+1] > prob:
                is_peak = False

            if is_peak and prob > CONFIRM_THR:
                if idx - split_indices[-1] >= MIN_LEN:
                    split_indices.append(idx)
                    while i < len(scan_indices) and scan_indices[i] < idx + MIN_LEN:
                        i += 1
                    continue
            i += 1

        split_indices.append(len(words))

        segments = []
        for k in range(len(split_indices) - 1):
            segments.append(" ".join(words[split_indices[k]:split_indices[k+1]]))

        return segments


class NewsClassifier:
    """Классификатор новостей - определяет topic и scale."""

    def __init__(self, topic_model_path: str, scale_model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.topic_tokenizer = AutoTokenizer.from_pretrained(topic_model_path)
        self.topic_model = AutoModelForSequenceClassification.from_pretrained(topic_model_path)
        self.topic_model.to(self.device)
        self.topic_model.eval()

        self.scale_tokenizer = AutoTokenizer.from_pretrained(scale_model_path)
        self.scale_model = AutoModelForSequenceClassification.from_pretrained(scale_model_path)
        self.scale_model.to(self.device)
        self.scale_model.eval()

    def _predict(self, text: str, tokenizer, model, max_len: int = 256) -> Dict[str, Any]:
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = int(torch.argmax(probs).item())

        label = model.config.id2label[pred_idx]
        return {"label": label, "confidence": float(probs[pred_idx])}

    def classify(self, text: str) -> Dict[str, Any]:
        """Классифицирует текст по topic и scale."""
        topic_result = self._predict(text, self.topic_tokenizer, self.topic_model)
        scale_result = self._predict(text, self.scale_tokenizer, self.scale_model)

        return {
            "topic": topic_result["label"],
            "topic_confidence": topic_result["confidence"],
            "scale": scale_result["label"],
            "scale_confidence": scale_result["confidence"],
        }


class NewsAttributeGenerator:
    """Генератор атрибутов - извлекает title, key_events, location, key_names."""

    def __init__(self, model_path: str):
        device = 0 if torch.cuda.is_available() else -1
        self.generator = hf_pipeline(
            "text2text-generation",
            model=model_path,
            tokenizer=model_path,
            device=device
        )

        self.tasks = {
            "title": "заголовок: ",
            "key_events": "событие: ",
            "location": "локация: ",
            "key_names": "имена: "
        }

    def generate(self, text: str) -> Dict[str, str]:
        """Генерирует атрибуты для текста."""
        results = {}

        for key, prefix in self.tasks.items():
            input_text = prefix + text
            output = self.generator(
                input_text,
                max_length=200,
                num_beams=4,
                early_stopping=True
            )[0]['generated_text']
            results[key] = output

        return results


# ============ Кэширование моделей ============

@st.cache_resource
def load_segmenter():
    model_path = str(BASE_DIR / "topicsegmenter" / "checkpoints" / "best_model")
    return NewsSegmenter(model_path)


@st.cache_resource
def load_classifier():
    topic_path = str(BASE_DIR / "classification" / "models_out_sbert_large_nlu_ru" / "topic" / "best")
    scale_path = str(BASE_DIR / "classification" / "models_out_sbert_large_nlu_ru" / "scale" / "best")
    return NewsClassifier(topic_path, scale_path)


@st.cache_resource
def load_generator():
    model_path = str(BASE_DIR / "rut5_extractor" / "final_model")
    return NewsAttributeGenerator(model_path)


# ============ Основной пайплайн ============

def process_news_stream(text: str) -> List[Dict[str, Any]]:
    """Обрабатывает поток новостей и возвращает структурированные данные."""
    segmenter = load_segmenter()
    classifier = load_classifier()
    generator = load_generator()

    # 1. Сегментация
    segments = segmenter.segment(text)

    results = []
    for segment in segments:
        # 2. Классификация
        classification = classifier.classify(segment)

        # 3. Генерация атрибутов
        attributes = generator.generate(segment)

        results.append({
            "text": segment,
            "topic": classification["topic"],
            "topic_confidence": classification["topic_confidence"],
            "scale": classification["scale"],
            "scale_confidence": classification["scale_confidence"],
            "title": attributes["title"],
            "key_events": attributes["key_events"],
            "location": attributes["location"],
            "key_names": attributes["key_names"],
        })

    return results


# ============ Streamlit UI ============

def main():
    st.set_page_config(
        page_title="News Structurizer",
        page_icon="📰",
        layout="wide"
    )

    st.title("📰 News Structurizer")
    st.markdown("Извлечение структурированной информации из потока новостей")

    # Sidebar с информацией
    with st.sidebar:
        st.header("О приложении")
        st.markdown("""
        **Пайплайн обработки:**
        1. **Сегментация** - разбиение текста на отдельные новости
        2. **Классификация** - определение темы и масштаба
        3. **Генерация** - извлечение заголовка, событий, локации, имён
        """)

        st.header("Модели")
        st.markdown("""
        - **Segmenter**: RuBERT (DeepPavlov)
        - **Classifier**: SBERT Large NLU RU
        - **Generator**: ruT5-base
        """)

    # Основной интерфейс
    text_input = st.text_area(
        "Введите текст новостей:",
        height=200,
        placeholder="Вставьте сюда сплошной текст с несколькими новостями..."
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        process_btn = st.button("🚀 Обработать", type="primary")

    if process_btn and text_input.strip():
        with st.spinner("Загрузка моделей..."):
            # Предзагрузка моделей
            load_segmenter()
            load_classifier()
            load_generator()

        with st.spinner("Обработка текста..."):
            results = process_news_stream(text_input)

        st.success(f"Найдено новостей: {len(results)}")

        # Отображение результатов
        for i, news in enumerate(results, 1):
            with st.expander(f"📰 Новость {i}: {news['title'][:80]}...", expanded=(i == 1)):
                # Метаданные
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Тема", news["topic"], f"{news['topic_confidence']:.1%}")
                with col2:
                    st.metric("Масштаб", news["scale"], f"{news['scale_confidence']:.1%}")

                st.divider()

                # Атрибуты
                st.markdown(f"**Заголовок:** {news['title']}")
                st.markdown(f"**Ключевые события:** {news['key_events']}")
                st.markdown(f"**Локация:** {news['location']}")
                st.markdown(f"**Ключевые имена:** {news['key_names']}")

                st.divider()

                # Исходный текст
                with st.container():
                    st.markdown("**Исходный текст:**")
                    st.text(news["text"][:500] + ("..." if len(news["text"]) > 500 else ""))

    elif process_btn:
        st.warning("Пожалуйста, введите текст для обработки")


if __name__ == "__main__":
    main()
