# News Structurizer: Отчет о результатах обучения

Этот проект использует гибридный подход для извлечения структурированной информации из новостных текстов:
1.  **BERT (ai-forever/sbert_large_nlu_ru)**: Классификация атрибутов `topic` и `scale`.
2.  **T5 (ai-forever/ruT5-base)**: Генерация атрибутов `title`, `key_events`, `location`, `key_names`.

---

## 1. Классификация (BERT)
Модель обучена предсказывать тему новости (`topic`) и масштаб события (`scale`).

### Метрики обучения (Validation)

| Задача | Accuracy | Macro F1 | Loss (Best) |
| :--- | :---: | :---: | :---: |
| **Topic / Scale** | **0.93** | **0.92** | **0.24** |


> **Анализ:** Модель быстро сходится и достигает плато на ~150-200 шагах. Высокий F1 (>0.9) говорит о том, что классы сбалансированы и модель уверенно различает темы.

---

## 2. Генерация (ruT5)
Модель обучена в режиме Multi-Task Learning для генерации заголовков, суммаризации событий и извлечения сущностей.

### Итоговые метрики (Validation)

| Метрика | Значение |
| :--- | :---: |
| **BERTScore F1** | **0.8451** |
| **Loss** | **0.1238** |

### Детализация по задачам (BERTScore)

| Атрибут | Score |
| :--- | :---: |
| **Location** | **0.91** |
| **Key Names** | **0.87** |
| **Title** | **0.83** |
| **Key Events** | **0.76** |

## 3. Сегментация текста (Topic Segmenter)

Модель для определения границ между тематическими блоками в тексте. Используется для разделения потока новостей на отдельные сообщения.

### Метрики (Validation)

| Метрика | Значение |
| :--- | :---: |
| **F1** | **0.9085** |
| **Precision** | **0.9308** |
| **Recall** | **0.8874** |

---

## 4. Итоговая архитектура пайплайна

Для обработки потока новостей используется следующий алгоритм:

```python
def process_news_stream(text):
    # 1. Сегментация (Topic Segmenter)
    # Разбиваем поток текста на отдельные новости
    news_segments = segmenter.segment(text)

    results = []
    for segment in news_segments:
        # 2. Классификация (BERT)
        topic = bert_model.predict(segment)  # "технологии и наука"
        scale = bert_model.predict(segment)  # "global"

        # 3. Генерация (T5)
        title = t5_model.generate("заголовок: " + segment)
        events = t5_model.generate("событие: " + segment)
        location = t5_model.generate("локация: " + segment)
        names = t5_model.generate("имена: " + segment)

        results.append({
            "text": segment,
            "topic": topic,
            "scale": scale,
            "title": title,
            "key_events": events,
            "location": location,
            "key_names": names
        })

    return results