from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Whisper ASR model into a local folder."
    )
    parser.add_argument("--model", default="abilmansplus/whisper-turbo-ksc2", help="HF model id")
    parser.add_argument("--out", default="models/asr", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use float32 for maximum compatibility when saving locally.
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    processor = AutoProcessor.from_pretrained(args.model)

    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)

    print(f"Saved model to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
