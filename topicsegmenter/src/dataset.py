import json
import torch
import random
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from .config import Config
from .augmentations import ASRAugmentor
from .utils import normalize_text

class SegmentationDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, config: Config, is_train: bool = False, data: Optional[List[Dict]] = None):
        self.data = data if data is not None else self._load_jsonl(data_path)
        self.tokenizer = tokenizer
        self.cfg = config
        self.is_train = is_train
        self.augmentor = ASRAugmentor() if is_train else None

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict]:
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
                missing = [k for k in ("context_left", "context_right", "label") if k not in obj]
                if missing:
                    raise ValueError(f"Missing keys {missing} at {path}:{line_num}")
                data.append(obj)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # --- Synthetic Topic Mixing ---
        # С вероятностью mix_prob берем левый контекст от текущего примера,
        # а правый - от случайного другого. Это создает гарантированный разрыв (Label 1).
        is_mixed = False
        if self.is_train and random.random() < self.cfg.mix_prob:
            rand_idx = random.randint(0, len(self.data) - 1)
            if len(self.data) > 1:
                while rand_idx == idx:
                    rand_idx = random.randint(0, len(self.data) - 1)
            other_item = self.data[rand_idx]
            
            left_words = item['context_left'].split()
            # Берем начало другого текста как продолжение
            right_words = other_item['context_left'].split() 
            # (можно брать и context_right, но context_left обычно начало новости/фразы)
            
            orig_label = 1 # Мы искусственно склеили -> разрыв есть
            is_mixed = True
        else:
            left_words = item['context_left'].split()
            right_words = item['context_right'].split()
            orig_label = item['label']

        # 1. Реконструкция полного потока слов
        full_text = left_words + right_words
        true_split_idx = len(left_words)

        split_point = true_split_idx
        label = 0

        # 2. Логика Jittering (Сдвиг границ)
        if self.is_train:
            if orig_label == 1:
                # Если это настоящий разрыв:
                # С вероятностью jitter_prob оставляем как есть (Label 1)
                # Иначе сдвигаем границу (Label 0 -> Hard Negative)
                # NB: for synthetic topic mixing we must keep label=1, otherwise we create noisy label=0 pairs
                # that still cross a real topic boundary.
                if is_mixed or random.random() <= self.cfg.jitter_prob:
                    label = 1
                    split_point = true_split_idx
                else:
                    label = 0
                    # Генерируем сдвиг: от 1 до max_jitter_shift, знак случайно
                    shift_dist = random.randint(1, self.cfg.max_jitter_shift)
                    shift = shift_dist if random.random() < 0.5 else -shift_dist
                    split_point = true_split_idx + shift
            else:
                # Если изначально разрыва не было, режем в случайном месте (но не с краю)
                label = 0
                safe_margin = 2
                if len(full_text) > safe_margin * 2:
                    split_point = random.randint(safe_margin, len(full_text) - safe_margin)
                else:
                    split_point = len(full_text) // 2
        else:
            # На валидации берем данные "как есть"
            label = orig_label
            split_point = true_split_idx

        # Защита границ массива
        split_point = max(1, min(len(full_text) - 1, split_point))

        # 3. Формирование контекстов
        ctx_left = " ".join(full_text[:split_point])
        ctx_right = " ".join(full_text[split_point:])

        # 4. Аугментация текста (только Train)
        if self.augmentor and random.random() < self.cfg.aug_prob:
            is_start = (label == 1)
            ctx_left = self.augmentor.apply(ctx_left)
            ctx_right = self.augmentor.apply(ctx_right, is_start_of_segment=is_start)

        # 4.1 Нормализация должна совпадать с inference
        ctx_left = normalize_text(ctx_left)
        ctx_right = normalize_text(ctx_right)

        # 4.2 EmbeddingGemma prompt (recommended by model card; Sentence-Transformers uses prompts)
        if (
            self.cfg.architecture == "bi_encoder"
            and getattr(self.cfg, "use_embedding_prompt", False)
            and "embeddinggemma" in str(self.cfg.model_name).lower()
        ):
            task = getattr(self.cfg, "embedding_task", "sentence similarity")
            prefix = f"task: {task} | query: "
            ctx_left = prefix + ctx_left
            ctx_right = prefix + ctx_right

        # 5. Токенизация
        if self.cfg.architecture == "bi_encoder":
            # Для Bi-Encoder токенизируем отдельно
            enc_left = self.tokenizer(
                ctx_left,
                add_special_tokens=True,
                max_length=self.cfg.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            enc_right = self.tokenizer(
                ctx_right,
                add_special_tokens=True,
                max_length=self.cfg.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids_left': enc_left['input_ids'].flatten(),
                'attention_mask_left': enc_left['attention_mask'].flatten(),
                'input_ids_right': enc_right['input_ids'].flatten(),
                'attention_mask_right': enc_right['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            # Для Cross-Encoder как было
            encoding = self.tokenizer(
                ctx_left,
                ctx_right,
                add_special_tokens=True,
                max_length=self.cfg.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].flatten()
            attention_mask = encoding['attention_mask'].flatten()
            token_type_ids = encoding.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.flatten()
            else:
                token_type_ids = torch.zeros_like(input_ids)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'labels': torch.tensor(label, dtype=torch.long)
            }

class SyntheticDataset(Dataset):
    """
    Датасет, генерирующий синтетические примеры из полных текстов новостей.
    Берет случайные новости и либо склеивает их (разрыв), либо режет одну (нет разрыва).
    """
    def __init__(self, json_path: str, tokenizer, config: Config, is_train: bool = True, epoch_multiplier: int = 10, split_ratio: float = 0.9):
        from collections import defaultdict
        self.all_texts = self._load_full_data(json_path)
        
        # Делим тексты на train/val
        random.seed(config.seed)
        random.shuffle(self.all_texts)
        split_idx = int(len(self.all_texts) * split_ratio)
        
        if is_train:
            self.data_source = self.all_texts[:split_idx]
        else:
            self.data_source = self.all_texts[split_idx:]
            
        # Группируем по топикам для Hard Positives
        self.topic_map = defaultdict(list)
        for item in self.data_source:
            t = item.get('topic', 'unknown')
            self.topic_map[t].append(item['text'])
            
        # Плоский список текстов для простых операций
        self.texts = [x['text'] for x in self.data_source]
            
        self.tokenizer = tokenizer
        self.cfg = config
        self.is_train = is_train
        self.epoch_multiplier = epoch_multiplier
        self.augmentor = ASRAugmentor() if is_train else None

    @staticmethod
    def _load_full_data(path: str) -> List[Dict]:
        data_list = []
        with open(path, 'r', encoding='utf-8') as f:
            try:
                full_json = json.load(f)
                for item in full_json:
                    if 'text' in item and item['text'].strip():
                        data_list.append(item)
            except json.JSONDecodeError as exc:
                print(f"Error loading {path}: {exc}")
        return data_list

    def __len__(self):
        return len(self.texts) * self.epoch_multiplier

    def __getitem__(self, idx):
        # 1. Сначала решаем, будет ли это "потенциальный" разрыв
        is_split_case = random.random() < 0.5
        
        if is_split_case:
            # С вероятностью 50% делаем "Сложный пример" (Same Topic Split)
            if random.random() < 0.5:
                # Пытаемся найти два текста одной темы
                rnd_item = random.choice(self.data_source)
                topic = rnd_item.get('topic', 'unknown')
                candidates = self.topic_map.get(topic, [])
                
                if len(candidates) >= 2:
                    text_a = rnd_item['text']
                    text_b = random.choice(candidates)
                    # Убедимся, что это не один и тот же текст
                    retries = 0
                    while text_b == text_a and retries < 5:
                        text_b = random.choice(candidates)
                        retries += 1
                else:
                    # Fallback на Random
                    text_a = rnd_item['text']
                    text_b = random.choice(self.texts)
            else:
                # Обычный Random Split (разные темы)
                text_a = random.choice(self.texts)
                text_b = random.choice(self.texts)
                while text_b == text_a and len(self.texts) > 1:
                    text_b = random.choice(self.texts)
            
            words_a = text_a.split()
            words_b = text_b.split()

            # Если тексты слишком короткие, fallback на non-split ветку
            if len(words_a) < 20 or len(words_b) < 20:
                is_split_case = False
            
        if is_split_case:
            # Точка склейки
            cut_a_low = max(1, len(words_a) // 4)
            cut_a_high = max(cut_a_low, len(words_a) - 5)
            cut_a = random.randint(cut_a_low, cut_a_high)

            cut_b_low = 5
            cut_b_high = max(cut_b_low, len(words_b) // 4)
            cut_b = random.randint(cut_b_low, cut_b_high)
            
            full_words = words_a[max(0, cut_a - 50):cut_a] + words_b[cut_b:min(len(words_b), cut_b + 50)]
            true_split_idx = len(words_a[max(0, cut_a - 50):cut_a])
            
            # Do not apply jittering here: both sides come from different texts, so shifting the split still
            # crosses a real topic boundary and creates noisy label=0 examples.
            label = 1
            split_point = true_split_idx
        else:
            label = 0
            # Берем один текст и режем его в случайном месте
            text = random.choice(self.texts)
            words = text.split()
            
            if len(words) < 2:
                full_words = words + words
                split_point = 1
            elif len(words) < 20:
                full_words = words
                split_point = len(words) // 2
            else:
                split_point = random.randint(10, len(words) - 10)
                # Ограничиваем контекст вокруг точки
                start = max(0, split_point - 50)
                end = min(len(words), split_point + 50)
                full_words = words[start:end]
                split_point = split_point - start

        # Защита границ
        split_point = max(1, min(len(full_words) - 1, split_point))
        
        ctx_left = " ".join(full_words[:split_point])
        ctx_right = " ".join(full_words[split_point:])

        # Аугментация (только для train)
        if self.is_train and self.augmentor and random.random() < self.cfg.aug_prob:
            is_start = (label == 1)
            ctx_left = self.augmentor.apply(ctx_left)
            ctx_right = self.augmentor.apply(ctx_right, is_start_of_segment=is_start)

        # Нормализация должна совпадать с inference
        ctx_left = normalize_text(ctx_left)
        ctx_right = normalize_text(ctx_right)

        # EmbeddingGemma prompt (recommended by model card; Sentence-Transformers uses prompts)
        if (
            self.cfg.architecture == "bi_encoder"
            and getattr(self.cfg, "use_embedding_prompt", False)
            and "embeddinggemma" in str(self.cfg.model_name).lower()
        ):
            task = getattr(self.cfg, "embedding_task", "sentence similarity")
            prefix = f"task: {task} | query: "
            ctx_left = prefix + ctx_left
            ctx_right = prefix + ctx_right

        # Токенизация
        if self.cfg.architecture == "bi_encoder":
            enc_left = self.tokenizer(
                ctx_left,
                add_special_tokens=True,
                max_length=self.cfg.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            enc_right = self.tokenizer(
                ctx_right,
                add_special_tokens=True,
                max_length=self.cfg.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids_left': enc_left['input_ids'].flatten(),
                'attention_mask_left': enc_left['attention_mask'].flatten(),
                'input_ids_right': enc_right['input_ids'].flatten(),
                'attention_mask_right': enc_right['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            encoding = self.tokenizer(
                ctx_left,
                ctx_right,
                add_special_tokens=True,
                max_length=self.cfg.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].flatten()
            attention_mask = encoding['attention_mask'].flatten()
            token_type_ids = encoding.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.flatten()
            else:
                token_type_ids = torch.zeros_like(input_ids)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'labels': torch.tensor(label, dtype=torch.long)
            }
