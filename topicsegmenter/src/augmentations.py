import random
import re
from .utils import normalize_text

class ASRAugmentor:
    """
    Класс для аугментации текстов на Казахском языке, имитирующий ошибки ASR.
    Включает специфические лингвистические ошибки (агглютинация, фонетика)
    и экстремальные искажения для повышения устойчивости модели.
    """
    def __init__(self):
        # 1. Фонетическая путаница (Матрица ошибок для Казахского языка)
        # Эти буквы часто путаются моделью распознавания
        self.confusable_chars = {
            'қ': 'к', 'к': 'қ', 
            'ғ': 'г', 'г': 'ғ',
            'ң': 'н', 'н': 'ң',
            'ө': 'о', 'о': 'ө', 
            'ұ': 'ү', 'ү': 'ұ', 
            'ы': 'і', 'і': 'ы',
            'һ': 'х', 'х': 'һ',
            'ә': 'а', 'а': 'ә'
        }
        
        # 2. Слова-ловушки (Topic Traps)
        # Слова, которые часто триггерят смену темы. Мы будем вставлять их внутрь
        # обычных предложений, чтобы отучить модель реагировать только на ключевые слова.
        self.trap_words = [
            "спорт", "футбол", "ауа райы", "доллар", "теңге", 
            "мәдениет", "концерт", "полиция", "президент", "төтенше жағдай"
        ]

        # 3. Якорные фразы (Anchors)
        # Типичные фразы ведущих новостей для смены темы (для Label 1)
        self.anchors = [
            "ал енді келесі тақырып", 
            "спорт жаңалықтарына тоқталсақ", 
            "ауа райы болжамына көшсек", 
            "бұл туралы толығырақ", 
            "сонымен қатар", 
            "еліміздегі басқа да оқиғалар",
            "қысқаша жаңалықтар легі",
            "шетелдік басылымдар не дейді",
            "ендігі кезекте",
            "бағдарламамызды жалғастырамыз"
        ]

        # 4. Слова-паразиты (Fillers)
        self.fillers = ["іммм", "жаңағы", "негізі", "былайша айтқанда", "ал", "ііі", "енді", "сөйтіп"]

        # 5. Частицы и короткие слова, которые часто приклеиваются к соседям
        self.particles = {"мен", "бен", "пен", "да", "де", "та", "те", "ма", "ме", "ба", "бе", "па", "пе"}

        # 6. Омофоны / Паронимы (Слова с похожим звучанием)
        self.word_swaps = {
            'қала': 'қара', 'қара': 'қала',    # город / смотри
            'келді': 'берді', 'берді': 'келді', # пришел / дал
            'елу': 'елі', 'елі': 'елу',         # 50 / страна
            'бар': 'нар', 'нар': 'бар'
        }

    # === БЛОК 1: КАЗАХСКАЯ ЛИНГВИСТИКА И ASR ===

    def _phonetic_noise(self, text: str, p: float = 0.15) -> str:
        """Замена специфических казахских букв (ң->н, қ->к)."""
        chars = list(text)
        for i, char in enumerate(chars):
            if char in self.confusable_chars and random.random() < p:
                chars[i] = self.confusable_chars[char]
        return "".join(chars)

    def _split_suffixes(self, text: str, p: float = 0.3) -> str:
        """
        Ошибка агглютинации: Отделение окончаний от корня.
        Пример: 'парламенттің' -> 'парламент тің'
        """
        words = text.split()
        new_words = []
        for word in words:
            if len(word) > 5 and random.random() < p:
                # Отделяем последние 2-4 буквы
                split_idx = len(word) - random.randint(2, 4)
                if split_idx > 2: 
                    new_words.append(word[:split_idx])
                    new_words.append(word[split_idx:])
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        return " ".join(new_words)

    def _drop_endings(self, text: str, p: float = 0.2) -> str:
        """
        Грамматическая ошибка: Выпадение окончаний.
        Пример: 'жұмыстарын' -> 'жұмыстары'
        """
        words = text.split()
        new_words = []
        for word in words:
            if len(word) > 5 and random.random() < p:
                cut_len = random.randint(1, 3)
                new_words.append(word[:-cut_len])
            else:
                new_words.append(word)
        return " ".join(new_words)

    def _glue_words_agglutinative(self, text: str, p: float = 0.25) -> str:
        """Склейка слов (особенно частиц мен/пен/да/де)."""
        words = text.split()
        if len(words) < 2: return text
        new_words = []
        skip_next = False
        
        for i in range(len(words) - 1):
            if skip_next:
                skip_next = False
                continue
            curr_word = words[i]
            next_word = words[i+1]
            
            is_particle = next_word in self.particles
            random_glue = (len(curr_word) <= 4 and random.random() < p)

            if is_particle or random_glue:
                new_words.append(curr_word + next_word)
                skip_next = True
            else:
                new_words.append(curr_word)
        
        if not skip_next: new_words.append(words[-1])
        return " ".join(new_words)

    # === БЛОК 2: ЭКСТРЕМАЛЬНЫЕ АУГМЕНТАЦИИ (HARD NEGATIVES) ===

    def _insert_gibberish(self, text: str, p: float = 0.1) -> str:
        """
        Вставка 'мусора' (галлюцинации ASR).
        Симулирует шум, музыку или неразборчивую речь.
        Пример: 'заң қабылданды ыыы ғғғ жжж бірақ'
        """
        words = text.split()
        if len(words) < 5: return text
        new_words = []
        # Набор символов, часто встречающихся в ошибках
        chars_set = "аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя"
        
        for word in words:
            new_words.append(word)
            if random.random() < p:
                # Генерируем бессмысленный токен
                gibberish = "".join(random.choices(chars_set, k=random.randint(3, 6)))
                if random.random() < 0.3: # Иногда повторяем его
                    gibberish = f"{gibberish} {gibberish}"
                new_words.append(gibberish)
        return " ".join(new_words)

    def _mega_glue(self, text: str, p: float = 0.1) -> str:
        """
        Экстремальная склейка (Snake Case).
        Убирает пробелы между цепочкой слов. Заставляет модель читать буквы, а не слова.
        Пример: 'бүгінпарламенттебюджет'
        """
        if random.random() > p: return text
        words = text.split()
        if len(words) < 6: return text.replace(" ", "")
        
        start_idx = random.randint(0, len(words) - 5)
        glue_len = random.randint(3, 6)
        
        segment = "".join(words[start_idx : start_idx + glue_len])
        new_words = words[:start_idx] + [segment] + words[start_idx + glue_len:]
        return " ".join(new_words)

    def _insert_topic_trap(self, text: str, p: float = 0.1) -> str:
        """
        Ловушка контекста: Вставляет слово из другой тематики.
        Используется для Label 0, чтобы модель не реагировала на слово "спорт" как на начало новости.
        """
        if random.random() > p:
            return text
        words = text.split()
        if len(words) < 4:
            return text

        max_inserts = 2 if len(words) < 30 else 3
        inserts = random.randint(1, max_inserts)
        for _ in range(inserts):
            insert_at = random.randint(1, len(words) - 1)
            words.insert(insert_at, random.choice(self.trap_words))
        return " ".join(words)

    def _insert_false_anchor(self, text: str, p: float = 0.2) -> str:
        """
        Вставляет "якорную" фразу ВНУТРИ сегмента (hard negative для Label 0),
        чтобы модель не резала текст по фразам ведущего, если они зашумлены/ошибочны.
        """
        if random.random() > p:
            return text
        words = text.split()
        if len(words) < 8:
            return text
        anchor = random.choice(self.anchors)
        anchor_words = anchor.split()
        insert_at = random.randint(2, len(words) - 2)
        words[insert_at:insert_at] = anchor_words
        return " ".join(words)

    def _insert_fillers(self, text: str, p: float = 0.15) -> str:
        """Вставляет слова-паразиты между словами (симулирует разговорную/ASR речь)."""
        words = text.split()
        if len(words) < 4:
            return text
        new_words = []
        for w in words:
            new_words.append(w)
            if random.random() < p:
                new_words.append(random.choice(self.fillers))
        return " ".join(new_words)

    def _drop_fillers(self, text: str, p: float = 0.2) -> str:
        """Удаляет часть слов-паразитов (симулирует неустойчивый ASR/редактуру)."""
        words = text.split()
        if len(words) < 4:
            return text
        return " ".join([w for w in words if not (w in self.fillers and random.random() < p)])

    def _aggressive_loop(self, text: str, p: float = 0.1) -> str:
        """Заедание буфера: Повтор слова 3-5 раз."""
        words = text.split()
        new_words = []
        for word in words:
            if random.random() < p:
                count = random.randint(3, 5)
                new_words.extend([word] * count)
            else:
                new_words.append(word)
        return " ".join(new_words)

    # === БЛОК 3: ПАРАЗИТЫ И СТРУКТУРА ===

    def _stutter_syllable(self, text: str, p: float = 0.1) -> str:
        """Повтор первого слога (заикание): 'ме мемлекеттік'."""
        words = text.split()
        new_words = []
        for word in words:
            if len(word) > 3 and random.random() < p:
                new_words.append(word[:2])
                new_words.append(word)
            else:
                new_words.append(word)
        return " ".join(new_words)

    def _insert_anchor(self, text: str, p: float = 0.3) -> str:
        """Вставка фразы ведущего (для Label 1)."""
        if random.random() > p: return text
        anchor = random.choice(self.anchors)
        return f"{anchor} {text}"

    def _sentence_shuffle(self, text: str, p: float = 0.1) -> str:
        """Перестановка кусков текста (потеря логики)."""
        if random.random() > p: return text
        words = text.split()
        if len(words) < 15: return text
        
        chunk_size = random.randint(7, 15)
        chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
        if len(chunks) > 1:
            random.shuffle(chunks)
        return " ".join([w for chunk in chunks for w in chunk])

    def apply(self, text: str, is_start_of_segment: bool = False) -> str:
        """
        Главный метод применения аугментаций.
        """
        text = normalize_text(text)
        
        # 1. Структурные изменения
        if is_start_of_segment:
             text = self._insert_anchor(text, p=0.4)
        
        text = self._sentence_shuffle(text, p=0.1)

        # 2. Экстремальные искажения (с небольшой вероятностью)
        if random.random() < 0.15: text = self._insert_gibberish(text, p=0.15)
        if random.random() < 0.15: text = self._mega_glue(text, p=0.5) # p здесь - шанс применения к строке
        if random.random() < 0.15: text = self._aggressive_loop(text, p=0.1)

        # 3. Речевые дефекты
        if random.random() < 0.3: text = self._stutter_syllable(text, p=0.15)
        if random.random() < 0.3: text = self._insert_fillers(text, p=0.15)
        if random.random() < 0.15: text = self._drop_fillers(text, p=0.25)
        
        # 4. Казахская морфология (Самое частое)
        if random.random() < 0.6: text = self._split_suffixes(text, p=0.3)
        if random.random() < 0.5: text = self._drop_endings(text, p=0.2)
        if random.random() < 0.5: text = self._glue_words_agglutinative(text, p=0.3)

        # 5. Фонетика
        if random.random() < 0.5: text = self._phonetic_noise(text, p=0.15)
        
        return text
