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
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    if all(k in obj for k in ("context_left", "context_right", "label")):
                        data.append(obj)
                except json.JSONDecodeError:
                    pass
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Topic Mixing (Label 1)
        is_mixed = False
        if self.is_train and random.random() < getattr(self.cfg, 'mix_prob', 0.0):
            other_item = self.data[random.randint(0, len(self.data) - 1)]
            left_words = item['context_left'].split()
            right_words = other_item['context_left'].split()
            orig_label = 1
            is_mixed = True
        
        if not is_mixed:
            left_words = item['context_left'].split()
            right_words = item['context_right'].split()
            orig_label = item['label']

        full_text = left_words + right_words
        true_split_idx = len(left_words)
        split_point = true_split_idx
        label = 0

        # 2. Jittering (Создаем Hard Negatives)
        if self.is_train:
            if orig_label == 1:
                # Вероятность оставить Label 1
                if is_mixed or random.random() <= getattr(self.cfg, 'jitter_prob', 0.5):
                    label = 1
                    split_point = true_split_idx
                else:
                    # Сдвиг -> Label 0 (Смена темы рядом, но не здесь)
                    # Используем экстремальный сдвиг для сложных случаев
                    if random.random() < getattr(self.cfg, 'extreme_jitter_prob', 0.2):
                        shift = random.choice([-1, 1]) * random.randint(20, 50)
                    else:
                        shift = random.choice([-1, 1]) * random.randint(1, getattr(self.cfg, 'max_jitter_shift', 10))
                    
                    label = 0
                    split_point = true_split_idx + shift
            else:
                label = 0
                split_point = random.randint(2, len(full_text) - 2) if len(full_text) > 4 else len(full_text)//2
        else:
            label = orig_label
            split_point = true_split_idx

        split_point = max(1, min(len(full_text) - 1, split_point))
        ctx_left = " ".join(full_text[:split_point])
        ctx_right = " ".join(full_text[split_point:])

        # 3. АУГМЕНТАЦИЯ (С ФОКУСОМ НА ЛОВУШКИ)
        if self.is_train and self.augmentor:
            
            # Если Label 0 -> ОБЯЗАТЕЛЬНО пытаемся вставить ловушки (мысалы, бірақ)
            # Чтобы модель видела их и не резала текст.
            if label == 0 and random.random() < getattr(self.cfg, 'false_anchor_prob', 0.5):
                ctx_left = self.augmentor._insert_topic_trap(ctx_left, p=0.9)
                ctx_right = self.augmentor._insert_topic_trap(ctx_right, p=0.9)
                ctx_left = self.augmentor._insert_false_anchor(ctx_left, p=0.7)
                ctx_right = self.augmentor._insert_false_anchor(ctx_right, p=0.7)
                ctx_right = self.augmentor._insert_gibberish(ctx_right, p=0.3)

            # Если Label 1 -> добавляем экстремальный шум, чтобы граница была видна даже в грязи
            elif label == 1 and random.random() < getattr(self.cfg, 'extreme_noise_prob', 0.2):
                ctx_left = self.augmentor._insert_gibberish(ctx_left, p=0.4)
                ctx_right = self.augmentor._mega_glue(ctx_right, p=0.5)
            
            # Обычная аугментация
            elif random.random() < getattr(self.cfg, 'aug_prob', 0.5):
                is_start = (label == 1)
                ctx_left = self.augmentor.apply(ctx_left, is_start_of_segment=False)
                ctx_right = self.augmentor.apply(ctx_right, is_start_of_segment=is_start)

        ctx_left = normalize_text(ctx_left)
        ctx_right = normalize_text(ctx_right)

        # Токенизация (Cross Encoder)
        if self.cfg.architecture == "cross_encoder":
            encoding = self.tokenizer(
                ctx_left, ctx_right, max_length=self.cfg.max_len, 
                padding='max_length', truncation=True, return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            # Bi-encoder stub
            enc_left = self.tokenizer(ctx_left, max_length=self.cfg.max_len, padding='max_length', truncation=True, return_tensors='pt')
            enc_right = self.tokenizer(ctx_right, max_length=self.cfg.max_len, padding='max_length', truncation=True, return_tensors='pt')
            return {
                'input_ids_left': enc_left['input_ids'].flatten(),
                'attention_mask_left': enc_left['attention_mask'].flatten(),
                'input_ids_right': enc_right['input_ids'].flatten(),
                'attention_mask_right': enc_right['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

class SyntheticDataset(Dataset):
    """
    Генерирует примеры на лету. 
    Важно: теперь генерирует 70% примеров Label 0!
    """
    def __init__(self, json_path: str, tokenizer, config: Config, is_train: bool = True, epoch_multiplier: int = 10, split_ratio: float = 0.9):
        from collections import defaultdict
        self.all_texts = self._load_full_data(json_path)
        random.seed(config.seed)
        random.shuffle(self.all_texts)
        split_idx = int(len(self.all_texts) * split_ratio)
        
        self.data_source = self.all_texts[:split_idx] if is_train else self.all_texts[split_idx:]
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
                    if 'text' in item: data_list.append(item)
            except: pass
        return data_list

    def __len__(self):
        return len(self.texts) * self.epoch_multiplier

    def __getitem__(self, idx):
        # БАЛАНСИРОВКА: synthetic_split_prob = 0.3
        # Значит 70% случаев будет Label 0 (продолжение темы)
        is_split_case = random.random() < getattr(self.cfg, 'synthetic_split_prob', 0.3)
        
        if is_split_case:
            # Label 1: Берем два разных текста
            text_a = random.choice(self.texts)
            text_b = random.choice(self.texts)
            words_a, words_b = text_a.split(), text_b.split()
            
            if len(words_a) < 20 or len(words_b) < 20:
                is_split_case = False # Fallback to Label 0
            else:
                cut_a = random.randint(len(words_a)//4, len(words_a)-5)
                cut_b = random.randint(5, len(words_b)//4)
                full_words = words_a[max(0, cut_a-50):cut_a] + words_b[cut_b:min(len(words_b), cut_b+50)]
                split_point = len(words_a[max(0, cut_a-50):cut_a])
                label = 1

        if not is_split_case:
            # Label 0: Берем один текст
            text = random.choice(self.texts)
            words = text.split()
            if len(words) < 20: 
                full_words = words; split_point = len(words)//2
            else:
                split_point = random.randint(10, len(words)-10)
                start, end = max(0, split_point-50), min(len(words), split_point+50)
                full_words = words[start:end]
                split_point = split_point - start
            label = 0

        split_point = max(1, min(len(full_words)-1, split_point))
        ctx_left = " ".join(full_words[:split_point])
        ctx_right = " ".join(full_words[split_point:])

        # Аугментация
        if self.is_train and self.augmentor:
            if label == 0 and random.random() < getattr(self.cfg, 'false_anchor_prob', 0.5):
                ctx_left = self.augmentor._insert_topic_trap(ctx_left, p=0.9)
                ctx_right = self.augmentor._insert_topic_trap(ctx_right, p=0.9)
                ctx_left = self.augmentor._insert_false_anchor(ctx_left, p=0.7)
                ctx_right = self.augmentor._insert_false_anchor(ctx_right, p=0.7)
            elif random.random() < getattr(self.cfg, 'aug_prob', 0.5):
                is_start = (label == 1)
                ctx_left = self.augmentor.apply(ctx_left)
                ctx_right = self.augmentor.apply(ctx_right, is_start_of_segment=is_start)

        ctx_left = normalize_text(ctx_left)
        ctx_right = normalize_text(ctx_right)
        
        # Tokenization (Cross Encoder copy)
        encoding = self.tokenizer(ctx_left, ctx_right, max_length=self.cfg.max_len, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
