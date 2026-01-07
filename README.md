# Модели, метрики, датасет | ASR

## Датасет

Для тестирования и оценки используется датасет **Kazakh Speech Corpus 2 (KSC2)**, а именно сегмент **radio**.

- **Датасет**: [issai/Kazakh_Speech_Corpus_2](https://huggingface.co/datasets/issai/Kazakh_Speech_Corpus_2)
- **Статья**: [Mussakhojayeva et al., INTERSPEECH 2022](https://www.isca-archive.org/interspeech_2022/mussakhojayeva22_interspeech.pdf)

### Статистика датасета (сегмент Radio)

| Сплит | Количество образцов |
|-------|---------------------|
| test_radio | 702 |
| dev_radio | 696 |
| train_radio | 34 941 |

Все оценки в данном проекте выполнены на сегменте **test_radio** (702 образца).

## Результаты оценки моделей

Были оценены три модели автоматического распознавания речи (ASR) на сегменте test_radio. В таблице ниже представлено сравнение Word Error Rate (WER), Character Error Rate (CER), Match Error Rate (MER) и скорости обработки:

| Модель | WER (%) | CER (%) | MER (%) | Среднее время на образец (с) | Общее время (с) |
|--------|---------|---------|---------|------------------------------|-----------------|
| **abilmansplus/whisper-turbo-ksc2** | **14.93** | **5.93** | **14.72** | **0.229** | **160.59** |
| facebook/seamless-m4t-v2-large | 54.06 | 26.08 | 52.81 | 0.840 | 589.81 |
| openai/whisper-large-v3-turbo | 105.97 | 78.03 | 80.26 | 0.394 | 276.53 |

**Конфигурация тестирования:**
- Количество образцов: 702 (сегмент test_radio)
- Язык: казахский (kaz/kk)
- Аудио формат: FLAC, 16 кГц

### Ключевые выводы

1. **Лучшая производительность**: `abilmansplus/whisper-turbo-ksc2` значительно превосходит другие модели с наименьшими показателями ошибок по всем метрикам (WER: 14.93%, CER: 5.93%, MER: 14.72%). Эта модель специально дообучена для казахской речи.

2. **Скорость обработки**: `abilmansplus/whisper-turbo-ksc2` также является самой быстрой, обрабатывая образцы в среднем за 0.229 секунды.

3. **Универсальные модели**:
   - `facebook/seamless-m4t-v2-large` показывает умеренную производительность, но значительно медленнее (0.840с на образец).
   - `openai/whisper-large-v3-turbo` плохо справляется с казахской речью, с WER превышающим 100%, что указывает на то, что модель генерирует больше ошибок, чем длина эталонного текста.

## Скрипты для оценки

### `evaluate_whisper.py`

Скрипт для оценки ASR-моделей на основе Whisper на датасете test_radio.

**Возможности:**
- Поддерживает любую модель Whisper из HuggingFace (например, `abilmansplus/whisper-turbo-ksc2`, `openai/whisper-large-v3-turbo`)
- Использует `transformers` pipeline для эффективной пакетной обработки
- Настраиваемый размер пакета для оптимизации производительности
- Измеряет метрики WER, CER и MER
- Отслеживает время обработки и генерирует подробные отчеты об оценке

**Использование:**
```bash
# Модель по умолчанию (abilmansplus/whisper-turbo-ksc2)
python evaluate_whisper.py

# Пользовательская модель с размером пакета 16
python evaluate_whisper.py --model openai/whisper-large-v3-turbo --batch-size 16

# Оценка на подмножестве данных
python evaluate_whisper.py --max-samples 100

# Пользовательская директория с данными
python evaluate_whisper.py --data-dir /path/to/data
```

**Аргументы:**
- `--model`: ID модели HuggingFace (по умолчанию: `abilmansplus/whisper-turbo-ksc2`)
- `--data-dir`: Путь к директории с аудиофайлами (по умолчанию: `test_radio/radio`)
- `--batch-size`: Размер пакета для обработки (по умолчанию: 8)
- `--max-samples`: Максимальное количество образцов для оценки
- `--device`: Устройство для использования (`cuda:0`, `cpu`)
- `--output`: Путь к выходному JSON файлу

### `evaluate_seamless.py`

Скрипт для оценки модели SeamlessM4T v2 на датасете test_radio.

**Возможности:**
- Оценивает `facebook/seamless-m4t-v2-large` (многоязычная модель речь-в-текст)
- Использует `SeamlessM4Tv2ForSpeechToText` для задач ASR
- Поддерживает несколько исходных языков через параметр `--src-lang`
- Реализует штраф за повторения для предотвращения проблем с повторением токенов
- Измеряет метрики WER, CER и MER
- Отслеживает время обработки и генерирует подробные отчеты об оценке

**Использование:**
```bash
# Модель по умолчанию (facebook/seamless-m4t-v2-large)
python evaluate_seamless.py

# Пользовательский исходный язык
python evaluate_seamless.py --src-lang kaz

# Оценка на подмножестве данных
python evaluate_seamless.py --max-samples 100

# Пользовательская директория с данными
python evaluate_seamless.py --data-dir /path/to/data
```

**Аргументы:**
- `--model`: ID модели HuggingFace (по умолчанию: `facebook/seamless-m4t-v2-large`)
- `--data-dir`: Путь к директории с аудиофайлами (по умолчанию: `test_radio/radio`)
- `--src-lang`: Код исходного языка (по умолчанию: `kaz` для казахского)
- `--max-samples`: Максимальное количество образцов для оценки
- `--device`: Устройство для использования (`cuda:0`, `cpu`)
- `--output`: Путь к выходному JSON файлу

**Примечание**: SeamlessM4T обрабатывает образцы последовательно (пакетная обработка не поддерживается для генерации), что приводит к более медленному выводу по сравнению с моделями Whisper.

## Объяснение метрик

- **WER (Word Error Rate)**: Измеряет процент неправильно распознанных слов. Чем ниже, тем лучше.
- **CER (Character Error Rate)**: Измеряет процент неправильно распознанных символов. Чем ниже, тем лучше.
- **MER (Match Error Rate)**: Измеряет процент неправильных совпадений слов. Чем ниже, тем лучше.

Все метрики рассчитываются с использованием библиотеки [Hugging Face evaluate](https://huggingface.co/docs/evaluate/) и [jiwer](https://github.com/jitsi/jiwer).