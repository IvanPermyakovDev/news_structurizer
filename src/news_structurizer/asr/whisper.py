from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline


@dataclass(frozen=True)
class WhisperConfig:
    model: str
    language: str = "kk"
    task: str = "transcribe"
    chunk_length_s: int | None = None
    batch_size: int = 8


class WhisperTranscriber:
    def __init__(self, config: WhisperConfig, device: str | None = None) -> None:
        self.config = config
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._pipe: Any | None = None

    def _load(self) -> Any:
        if self._pipe is not None:
            return self._pipe

        is_cuda = str(self.device).startswith("cuda")
        dtype = torch.float16 if is_cuda else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.config.model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(self.config.model)

        kwargs: dict[str, Any] = {}
        if self.config.chunk_length_s is not None:
            kwargs["chunk_length_s"] = self.config.chunk_length_s

        self._pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=self.device,
            **kwargs,
        )
        return self._pipe

    def transcribe(self, audio_path: str) -> dict[str, Any]:
        pipe = self._load()

        result: Any = pipe(
            audio_path,
            generate_kwargs={"language": self.config.language, "task": self.config.task},
            return_timestamps=False,
            batch_size=self.config.batch_size,
        )

        if isinstance(result, dict) and "text" in result:
            return {"text": result["text"]}
        return {"text": str(result)}

