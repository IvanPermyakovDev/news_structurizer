from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> None:
    _ensure_src_on_path()

    from news_structurizer.asr import WhisperConfig, WhisperTranscriber

    parser = argparse.ArgumentParser(description="Run ASR (audio -> text) via Whisper.")
    parser.add_argument("--audio", required=True, help="Path to audio file.")
    parser.add_argument("--out", help="Write JSON to a file (optional).")
    parser.add_argument("--device", default=None, help="cpu / cuda:0 ... (optional)")
    parser.add_argument(
        "--asr-model",
        default=os.environ.get("NS_ASR_MODEL") or "abilmansplus/whisper-turbo-ksc2",
        help="Local model dir or HF model id (env: NS_ASR_MODEL).",
    )
    parser.add_argument(
        "--asr-language",
        default=os.environ.get("NS_ASR_LANGUAGE") or "kk",
        help="Language code for generation (default: kk).",
    )
    args = parser.parse_args()

    transcriber = WhisperTranscriber(
        WhisperConfig(model=args.asr_model, language=args.asr_language),
        device=args.device,
    )
    result = transcriber.transcribe(args.audio)
    payload = json.dumps(result, ensure_ascii=False, indent=2)

    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
