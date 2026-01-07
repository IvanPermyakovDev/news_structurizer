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

    from news_structurizer import Pipeline, PipelineConfig

    repo_root = Path(__file__).resolve().parents[1]
    default_segmenter = (
        repo_root / "research" / "ml" / "topicsegmenter" / "checkpoints" / "best_model"
    )
    default_topic = (
        repo_root / "research" / "ml" / "classification" / "models_out_kz" / "topic" / "best"
    )
    default_scale = (
        repo_root / "research" / "ml" / "classification" / "models_out_kz" / "scale" / "best"
    )
    default_extractor = (
        repo_root
        / "research"
        / "ml"
        / "extractor"
        / "t5gemma_270m_kz_news_attributes_frozen"
        / "checkpoint-2000"
    )

    parser = argparse.ArgumentParser(description="Run full pipeline on audio (ASR -> structuring).")
    parser.add_argument("--audio", required=True, help="Path to audio file.")
    parser.add_argument("--out", help="Write report JSON to a file (optional).")
    parser.add_argument("--device", default=None, help="cpu / cuda:0 ... (optional)")

    parser.add_argument("--segmenter-model", default=str(default_segmenter))
    parser.add_argument("--topic-model", default=str(default_topic))
    parser.add_argument("--scale-model", default=str(default_scale))
    parser.add_argument("--extractor-model", default=str(default_extractor))
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

    cfg = PipelineConfig(
        segmenter_model_path=args.segmenter_model,
        topic_model_path=args.topic_model,
        scale_model_path=args.scale_model,
        extractor_model_path=args.extractor_model,
        asr_model=args.asr_model,
        asr_language=args.asr_language,
        device=args.device,
    )

    report = Pipeline(cfg).process_audio(args.audio)
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)

    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
