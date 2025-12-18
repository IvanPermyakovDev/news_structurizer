import random
from .utils import normalize_text

class ASRAugmentor:
    def __init__(self):
        # Типичные ошибки распознавания
        self.vowels = {'о': 'а', 'а': 'о', 'е': 'и', 'и': 'е', 'я': 'и'}
        self.consonants = {'тся': 'ца', 'ться': 'ца', 'в': 'ф', 'г': 'к', 'д': 'т'}
        self.anchors = [
            "а теперь к другим новостям", "продолжаем выпуск", "в то же время", 
            "кстати о погоде", "вернемся к главной теме", "следующий сюжет",
            "переходим к международной панораме", "коротко о других событиях",
            "и напоследок", "еще одна новость"
        ]

    def _phonetic_noise(self, text: str) -> str:
        """Меняет гласные и оглушает согласные."""
        chars = list(text)
        for i, char in enumerate(chars):
            if char in self.vowels and random.random() < 0.15:
                chars[i] = self.vowels[char]
        
        res = "".join(chars)
        # Обработка сочетаний букв
        for k, v in self.consonants.items():
            if k in res and random.random() < 0.2:
                res = res.replace(k, v, 1)
        return res

    def _glue_words(self, text: str) -> str:
        """Склеивает предлоги и короткие слова."""
        words = text.split()
        if len(words) < 2: return text
        
        new_words = []
        skip_next = False
        
        for i in range(len(words) - 1):
            if skip_next:
                skip_next = False
                continue
                
            # Склеиваем короткое слово со следующим
            if len(words[i]) <= 3 and random.random() < 0.25:
                new_words.append(words[i] + words[i+1])
                skip_next = True
            else:
                new_words.append(words[i])
        
        if not skip_next: 
            new_words.append(words[-1])
            
        return " ".join(new_words)

    def _word_deletion(self, text: str, p: float = 0.1) -> str:
        """Случайное удаление слов."""
        words = text.split()
        if len(words) < 2: return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        if not new_words: return random.choice(words)
        return " ".join(new_words)

    def _word_insertion(self, text: str, p: float = 0.1) -> str:
        """Вставка случайных слов (дубликатов)."""
        words = text.split()
        if not words: return text
        
        new_words = []
        for word in words:
            new_words.append(word)
            if random.random() < p:
                new_words.append(random.choice(words))
        return " ".join(new_words)

    def _word_swap(self, text: str, p: float = 0.1) -> str:
        """Перестановка соседних слов."""
        words = text.split()
        if len(words) < 2: return text
        
        for i in range(len(words) - 1):
            if random.random() < p:
                words[i], words[i+1] = words[i+1], words[i]
        return " ".join(words)

    def _char_noise(self, text: str, p: float = 0.05) -> str:
        """Случайное удаление или вставка символов."""
        chars = list(text)
        new_chars = []
        for char in chars:
            r = random.random()
            if r < p:
                continue # Deletion
            new_chars.append(char)
            if r > 1 - p:
                new_chars.append(char) # Insertion (duplication)
        return "".join(new_chars)

    def _cutout(self, text: str, p: float = 0.1) -> str:
        """Вырезание куска текста."""
        if random.random() > p: return text
        if len(text) < 10: return text
        
        start = random.randint(0, len(text) - 5)
        length = random.randint(3, min(20, len(text) - start))
        return text[:start] + text[start+length:]

    def _random_trim(self, text: str, p: float = 0.1) -> str:
        """Случайное обрезание слов с краев."""
        words = text.split()
        if len(words) < 4: return text
        
        if random.random() < p:
            # Trim start
            n_trim = random.randint(1, 3)
            words = words[n_trim:]
            
        if random.random() < p:
            # Trim end
            if len(words) > 2:
                n_trim = random.randint(1, 3)
                words = words[:-n_trim]
                
        return " ".join(words)

    def _stuttering(self, text: str, p: float = 0.1) -> str:
        """Дублирует слова (имитация заикания/повторов)."""
        words = text.split()
        if not words: return text
        
        new_words = []
        for word in words:
            new_words.append(word)
            if random.random() < p:
                new_words.append(word) # Дублируем: "погода" -> "погода погода"
                
        return " ".join(new_words)

    def _filler_injection(self, text: str, p: float = 0.1) -> str:
        """Вставляет слова-паразиты."""
        fillers = ["эээ", "ммм", "ну", "как бы", "типа", "значит", "вот", "короче"]
        words = text.split()
        if not words: return text
        
        new_words = []
        for word in words:
            new_words.append(word)
            if random.random() < p:
                new_words.append(random.choice(fillers))
                
        return " ".join(new_words)

    def _insert_anchor(self, text: str, p: float = 0.3) -> str:
        """Вставляет фразу ведущего в начало (для правой части контекста)."""
        if random.random() > p: return text
        anchor = random.choice(self.anchors)
        # В ASR нет пунктуации, поэтому просто через пробел
        return f"{anchor} {text}"

    def _sentence_shuffle(self, text: str, p: float = 0.2) -> str:
        """Меняет местами соседние предложения (симуляция потери логики без смены темы)."""
        if random.random() > p: return text
        # Эвристика: разбиваем по длинным паузам или просто по количеству слов (т.к. точек нет)
        words = text.split()
        if len(words) < 20: return text
        
        # Пытаемся разбить на условные "предложения" по 10-15 слов
        chunk_size = random.randint(10, 20)
        chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
        
        if len(chunks) > 1:
            random.shuffle(chunks)
            
        flat_words = [w for chunk in chunks for w in chunk]
        return " ".join(flat_words)

    def apply(self, text: str, is_start_of_segment: bool = False) -> str:
        """Применяет пайплайн искажений."""
        # Базовая предобработка
        text = normalize_text(text)
        
        # 0. Anchor Phrases (Новое: Фразы-клише ведущих)
        if is_start_of_segment:
             text = self._insert_anchor(text, p=0.4)

        # 0.1 Sentence Shuffle (Новое: Перестановка "предложений")
        text = self._sentence_shuffle(text, p=0.2)

        # 0.2 Disfluencies (Заикание и Паразиты)
        if random.random() < 0.3: text = self._stuttering(text, p=0.1)
        if random.random() < 0.3: text = self._filler_injection(text, p=0.1)

        # 1. Word-level augmentations
        if random.random() < 0.5: text = self._word_deletion(text, p=0.1)
        if random.random() < 0.3: text = self._word_insertion(text, p=0.1)
        if random.random() < 0.3: text = self._word_swap(text, p=0.1)
        
        # 2. Existing ASR-like augmentations
        if random.random() < 0.5: text = self._glue_words(text)
        if random.random() < 0.5: text = self._phonetic_noise(text)
        
        # 3. Char-level & Structural
        if random.random() < 0.3: text = self._char_noise(text, p=0.05)
        text = self._cutout(text, p=0.1)
        text = self._random_trim(text, p=0.2)
        
        return text
