#!/usr/bin/env python3

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pathlib import Path
from typing import Union, Optional
import json


class AudioProcessor:
    def __init__(
        self,
        model_id: str = "abilmansplus/whisper-turbo-ksc2",
        device: Optional[str] = None
    ):
        self.model_id = model_id
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pipe = self._load_model()

    def _load_model(self):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(self.model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16,
            device=self.device,
        )

        return pipe

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: str = "kk",
        task: str = "transcribe",
        return_timestamps: bool = False
    ) -> dict:
        audio_path = str(audio_path)

        result = self.pipe(
            audio_path,
            generate_kwargs={"language": language, "task": task},
            return_timestamps=return_timestamps
        )

        return result

    def process_and_save(
        self,
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        language: str = "kk",
        save_format: str = "txt"
    ) -> str:
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        result = self.transcribe(audio_path, language=language)
        transcription = result["text"]

        if output_path is None:
            output_path = audio_path.with_suffix(f'.{save_format}')
        else:
            output_path = Path(output_path)

        if save_format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcription)
        elif save_format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {save_format}. Use 'txt' or 'json'")

        return str(output_path)

    def process_batch(
        self,
        audio_paths: list[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        language: str = "kk",
        save_format: str = "txt",
        batch_size: int = 8
    ) -> list[str]:
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = []

        for i in range(0, len(audio_paths), batch_size):
            batch = audio_paths[i:i + batch_size]
            batch_str = [str(p) for p in batch]

            results = self.pipe(
                batch_str,
                generate_kwargs={"language": language, "task": "transcribe"},
                return_timestamps=False,
                batch_size=batch_size
            )

            if batch_size == 1:
                results = [results]

            for audio_path, result in zip(batch, results):
                audio_path = Path(audio_path)
                transcription = result["text"]

                if output_dir:
                    output_path = output_dir / audio_path.with_suffix(f'.{save_format}').name
                else:
                    output_path = audio_path.with_suffix(f'.{save_format}')

                if save_format == "txt":
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(transcription)
                elif save_format == "json":
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)

                output_paths.append(str(output_path))

        return output_paths


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process audio files with Whisper Turbo KSC2")

    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to audio file or directory with audio files"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path or directory (default: same as input with .txt extension)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="abilmansplus/whisper-turbo-ksc2",
        help="HuggingFace model ID (default: abilmansplus/whisper-turbo-ksc2)"
    )

    parser.add_argument(
        "--language",
        type=str,
        default="kk",
        help="Language code (default: kk)"
    )

    parser.add_argument(
        "--format",
        type=str,
        default="txt",
        choices=["txt", "json"],
        help="Output format (default: txt)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing multiple files (default: 8)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda:0', 'cpu'). If not specified, uses CUDA if available"
    )

    args = parser.parse_args()

    processor = AudioProcessor(model_id=args.model, device=args.device)

    audio_path = Path(args.audio_path)

    if audio_path.is_file():
        output_path = processor.process_and_save(
            audio_path,
            output_path=args.output,
            language=args.language,
            save_format=args.format
        )
        print(f"Transcription saved to: {output_path}")

    elif audio_path.is_dir():
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']:
            audio_files.extend(audio_path.glob(ext))

        if not audio_files:
            print(f"No audio files found in {audio_path}")
            return

        print(f"Found {len(audio_files)} audio files")

        output_paths = processor.process_batch(
            audio_files,
            output_dir=args.output,
            language=args.language,
            save_format=args.format,
            batch_size=args.batch_size
        )

        print(f"Processed {len(output_paths)} files")
        print(f"Transcriptions saved to: {args.output if args.output else audio_path}")

    else:
        print(f"Error: {audio_path} is not a valid file or directory")


if __name__ == "__main__":
    main()

