from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    num_beams: int = 4
    no_repeat_ngram_size: int = 2
    repetition_penalty: float = 1.2
    early_stopping: bool = True


class AttributeExtractor:
    TASKS: dict[str, str] = {
        "title": "тақырып: ",
        "key_events": "оқиға: ",
        "location": "орын: ",
        "key_names": "есімдер: ",
    }

    def __init__(self, model_path: str, device: str | None = None) -> None:
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = resolved_device

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def extract(self, text: str, config: GenerationConfig | None = None) -> dict[str, str]:
        cfg = config or GenerationConfig()
        results: dict[str, str] = {}

        for key, prefix in self.TASKS.items():
            input_text = f"{prefix}{text}"
            inputs: Any = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=cfg.max_new_tokens,
                    num_beams=cfg.num_beams,
                    no_repeat_ngram_size=cfg.no_repeat_ngram_size,
                    repetition_penalty=cfg.repetition_penalty,
                    early_stopping=cfg.early_stopping,
                )

            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results[key] = decoded.strip()

        return results

