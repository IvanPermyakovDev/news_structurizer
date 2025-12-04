import json
import random
from typing import Any, Dict, List, Sequence, Tuple
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class NewsSegmentationDataset(Dataset):
    """
    Dataset for news segmentation task.
    Concatenates multiple news texts and creates labels for boundary detection.
    Label 1 = start of new news, Label 0 = continuation

    Trims start and end of each news to remove greetings/farewells,
    forcing model to learn semantic boundaries instead of lexical markers.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        min_news: int = 2,
        max_news: int = 4,
        samples_per_epoch: int = 5000,
        seed: int = 42,
        trim_start_pct: float = 0.15,
        trim_end_pct: float = 0.15,
        trim_random_range: float = 0.1,
    ):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.news_items = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_news = min_news
        self.max_news = max_news
        self.samples_per_epoch = samples_per_epoch
        self.rng = random.Random(seed)

        self.trim_start_pct = trim_start_pct
        self.trim_end_pct = trim_end_pct
        self.trim_random_range = trim_random_range

        self.separator = " "

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        num_news = self.rng.randint(self.min_news, self.max_news)
        selected = self.rng.sample(self.news_items, min(num_news, len(self.news_items)))

        texts = [self._trim_text(item['text']) for item in selected]
        combined_text, char_boundaries = self._combine_texts(texts)

        encoding = self.tokenizer(
            combined_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        token_labels = self._create_token_labels(
            encoding['offset_mapping'][0],
            char_boundaries,
            encoding['attention_mask'][0]
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': token_labels,
            'num_boundaries': torch.tensor(len(char_boundaries) - 1)
        }

    def _trim_text(self, text: str) -> str:
        """
        Trim start and end of text to remove greetings/farewells.
        Adds randomness to prevent model from learning fixed positions.
        """
        length = len(text)
        if length < 100:
            return text

        start_pct = self.trim_start_pct + self.rng.uniform(-self.trim_random_range, self.trim_random_range)
        end_pct = self.trim_end_pct + self.rng.uniform(-self.trim_random_range, self.trim_random_range)

        start_pct = max(0.05, min(0.3, start_pct))
        end_pct = max(0.05, min(0.3, end_pct))

        start_pos = int(length * start_pct)
        end_pos = int(length * (1 - end_pct))

        if end_pos <= start_pos:
            return text

        start_space = text.find(' ', start_pos)
        if start_space != -1 and start_space < start_pos + 50:
            start_pos = start_space + 1

        end_space = text.rfind(' ', 0, end_pos)
        if end_space != -1 and end_space > end_pos - 50:
            end_pos = end_space

        return text[start_pos:end_pos].strip()

    def _combine_texts(self, texts: List[str]) -> Tuple[str, List[int]]:
        """Combine texts and track boundary positions (character level)."""
        combined = ""
        boundaries = [0]

        for i, text in enumerate(texts):
            if i > 0:
                combined += self.separator
            combined += text
            boundaries.append(len(combined))

        return combined, boundaries[:-1]

    def _create_token_labels(
        self,
        offset_mapping: torch.Tensor,
        char_boundaries: List[int],
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Create token-level labels. 1 = boundary (start of news), 0 = continuation, -100 = ignore."""
        labels = torch.full((len(offset_mapping),), -100, dtype=torch.long)

        for i, (start, end) in enumerate(offset_mapping.tolist()):
            if attention_mask[i] == 0 or (start == 0 and end == 0):
                continue

            is_boundary = any(start <= b < end or (b == start and b in char_boundaries) for b in char_boundaries)
            if not is_boundary:
                is_boundary = start in char_boundaries

            labels[i] = 1 if is_boundary else 0

        return labels


class NewsSegmentationCollator:
    """Collator for batching."""

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features]),
            'num_boundaries': torch.stack([f['num_boundaries'] for f in features])
        }
        return batch


def get_datasets(
    train_path: str,
    val_path: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    train_samples: int = 5000,
    val_samples: int = 1000
) -> Tuple[NewsSegmentationDataset, NewsSegmentationDataset]:
    train_dataset = NewsSegmentationDataset(
        train_path, tokenizer, max_length,
        samples_per_epoch=train_samples, seed=42
    )
    val_dataset = NewsSegmentationDataset(
        val_path, tokenizer, max_length,
        samples_per_epoch=val_samples, seed=123
    )
    return train_dataset, val_dataset


class BoundaryDataset(Dataset):
    """
    Holds prebuilt sequences of chunks with boundary labels.
    Each example is a dict with keys: {"chunks": List[str], "labels": List[int]}.
    Labels are 1 where a new article starts after the chunk.
    """

    def __init__(self, examples: Sequence[Dict[str, List[str]]]):
        self.examples = list(examples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        example = self.examples[idx]
        return {
            "chunks": example["chunks"],
            "labels": torch.tensor(example["labels"], dtype=torch.float),
        }


def build_collate_fn(
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
):
    """
    Pads variable-length chunk sequences into a batch tensor:
    - input_ids:  (B, T, L)
    - attention:  (B, T, L)
    - chunk_mask: (B, T) True for real chunks
    - labels:     (B, T) float boundary targets
    """

    def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        chunk_counts = [len(item["chunks"]) for item in batch]
        max_chunks = max(chunk_counts)

        flat_chunks: List[str] = []
        for item in batch:
            flat_chunks.extend(item["chunks"])

        tokenized = tokenizer(
            flat_chunks,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )

        input_ids = torch.zeros(batch_size, max_chunks, max_seq_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_chunks, max_seq_len, dtype=torch.long)
        chunk_mask = torch.zeros(batch_size, max_chunks, dtype=torch.bool)
        labels = torch.zeros(batch_size, max_chunks, dtype=torch.float)

        cursor = 0
        for batch_idx, item in enumerate(batch):
            local_count = len(item["chunks"])
            chunk_mask[batch_idx, :local_count] = True
            labels[batch_idx, :local_count] = item["labels"]

            for chunk_idx in range(local_count):
                input_ids[batch_idx, chunk_idx] = tokenized["input_ids"][cursor]
                attention_mask[batch_idx, chunk_idx] = tokenized["attention_mask"][cursor]
                cursor += 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "chunk_mask": chunk_mask,
            "labels": labels,
        }

    return collate
