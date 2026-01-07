from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .schemas import PipelineConfig


def _env_default(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="news-structurizer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_asr = sub.add_parser("asr-audio", help="Run only ASR (audio -> text) using Whisper.")
    p_asr.add_argument("--audio", required=True, help="Path to audio file.")
    p_asr.add_argument("--out", help="Write JSON to a file. If omitted, prints to stdout.")
    p_asr.add_argument(
        "--asr-model",
        default=_env_default("NS_ASR_MODEL") or "abilmansplus/whisper-turbo-ksc2",
    )
    p_asr.add_argument("--asr-language", default=_env_default("NS_ASR_LANGUAGE") or "kk")
    p_asr.add_argument("--device", default=_env_default("NS_DEVICE"))

    p_seg = sub.add_parser("segment-text", help="Run only segmentation on input text.")
    p_seg.add_argument("--text", help="Input text. If omitted, reads from stdin.")
    p_seg.add_argument("--file", help="Read input text from a file.")
    p_seg.add_argument("--out", help="Write JSON to a file. If omitted, prints to stdout.")
    p_seg.add_argument("--segmenter-model", default=_env_default("NS_SEGMENTER_MODEL"))
    p_seg.add_argument("--device", default=_env_default("NS_DEVICE"))

    p_cls = sub.add_parser(
        "classify-text",
        help="Run only topic/scale classification on input text.",
    )
    p_cls.add_argument("--text", help="Input text. If omitted, reads from stdin.")
    p_cls.add_argument("--file", help="Read input text from a file.")
    p_cls.add_argument("--out", help="Write JSON to a file. If omitted, prints to stdout.")
    p_cls.add_argument("--topic-model", default=_env_default("NS_TOPIC_MODEL"))
    p_cls.add_argument("--scale-model", default=_env_default("NS_SCALE_MODEL"))
    p_cls.add_argument("--device", default=_env_default("NS_DEVICE"))

    p_ext = sub.add_parser("extract-text", help="Run only attribute extraction on input text.")
    p_ext.add_argument("--text", help="Input text. If omitted, reads from stdin.")
    p_ext.add_argument("--file", help="Read input text from a file.")
    p_ext.add_argument("--out", help="Write JSON to a file. If omitted, prints to stdout.")
    p_ext.add_argument("--extractor-model", default=_env_default("NS_EXTRACTOR_MODEL"))
    p_ext.add_argument("--device", default=_env_default("NS_DEVICE"))

    p_text = sub.add_parser("process-text", help="Run pipeline on input text.")
    p_text.add_argument("--text", help="Input text. If omitted, reads from stdin.")
    p_text.add_argument("--file", help="Read input text from a file.")
    p_text.add_argument("--out", help="Write report JSON to a file. If omitted, prints to stdout.")

    p_text.add_argument("--segmenter-model", default=_env_default("NS_SEGMENTER_MODEL"))
    p_text.add_argument("--topic-model", default=_env_default("NS_TOPIC_MODEL"))
    p_text.add_argument("--scale-model", default=_env_default("NS_SCALE_MODEL"))
    p_text.add_argument("--extractor-model", default=_env_default("NS_EXTRACTOR_MODEL"))
    p_text.add_argument("--device", default=_env_default("NS_DEVICE"))

    p_audio = sub.add_parser(
        "process-audio",
        help="Run full pipeline on audio (ASR -> structuring).",
    )
    p_audio.add_argument("--audio", required=True, help="Path to audio file.")
    p_audio.add_argument("--out", help="Write report JSON to a file. If omitted, prints to stdout.")
    p_audio.add_argument("--segmenter-model", default=_env_default("NS_SEGMENTER_MODEL"))
    p_audio.add_argument("--topic-model", default=_env_default("NS_TOPIC_MODEL"))
    p_audio.add_argument("--scale-model", default=_env_default("NS_SCALE_MODEL"))
    p_audio.add_argument("--extractor-model", default=_env_default("NS_EXTRACTOR_MODEL"))
    p_audio.add_argument(
        "--asr-model",
        default=_env_default("NS_ASR_MODEL") or "abilmansplus/whisper-turbo-ksc2",
    )
    p_audio.add_argument("--asr-language", default=_env_default("NS_ASR_LANGUAGE") or "kk")
    p_audio.add_argument("--device", default=_env_default("NS_DEVICE"))

    return parser


def _load_text(args: argparse.Namespace) -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8").strip()
    if args.text:
        return args.text.strip()
    return input().strip()


def _require(value: str | None, flag: str) -> str:
    if not value:
        raise SystemExit(f"Missing required model path. Provide `{flag}` or set env var.")
    return value


def _is_missing_ml_dep(exc: ModuleNotFoundError) -> bool:
    return exc.name in {"torch", "transformers"}


def _ml_missing_message() -> str:
    return (
        "Missing ML dependencies. Install extras and retry:\n"
        "  pip install -e '.[ml]'\n"
        "or for dev:\n"
        "  pip install -e '.[dev,ml]'\n"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "asr-audio":
        try:
            from .asr.whisper import WhisperConfig, WhisperTranscriber
        except ModuleNotFoundError as exc:
            if _is_missing_ml_dep(exc):
                raise SystemExit(_ml_missing_message()) from exc
            raise

        asr_cfg = WhisperConfig(
            model=args.asr_model,
            language=args.asr_language,
        )
        transcriber = WhisperTranscriber(asr_cfg, device=args.device)
        result = transcriber.transcribe(args.audio)
        payload = json.dumps(result, ensure_ascii=False, indent=2)

        if args.out:
            Path(args.out).write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return

    if args.command == "segment-text":
        try:
            from .segmentation.topicsegmenter import NewsSegmenter
        except ModuleNotFoundError as exc:
            if _is_missing_ml_dep(exc):
                raise SystemExit(_ml_missing_message()) from exc
            raise

        text = _load_text(args)
        if not text:
            raise SystemExit("Empty text input.")

        model_path = _require(args.segmenter_model, "--segmenter-model")
        segments = NewsSegmenter(model_path=model_path, device=args.device).segment(text)
        payload = json.dumps({"segments": segments}, ensure_ascii=False, indent=2)

        if args.out:
            Path(args.out).write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return

    if args.command == "classify-text":
        try:
            from .classification.topic_scale import NewsClassifier
        except ModuleNotFoundError as exc:
            if _is_missing_ml_dep(exc):
                raise SystemExit(_ml_missing_message()) from exc
            raise

        text = _load_text(args)
        if not text:
            raise SystemExit("Empty text input.")

        classifier = NewsClassifier(
            topic_model_path=_require(args.topic_model, "--topic-model"),
            scale_model_path=_require(args.scale_model, "--scale-model"),
            device=args.device,
        )
        result = classifier.classify(text)
        payload = json.dumps(result, ensure_ascii=False, indent=2)

        if args.out:
            Path(args.out).write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return

    if args.command == "extract-text":
        try:
            from .extraction.attributes import AttributeExtractor
        except ModuleNotFoundError as exc:
            if _is_missing_ml_dep(exc):
                raise SystemExit(_ml_missing_message()) from exc
            raise

        text = _load_text(args)
        if not text:
            raise SystemExit("Empty text input.")

        extractor = AttributeExtractor(
            model_path=_require(args.extractor_model, "--extractor-model"),
            device=args.device,
        )
        result = extractor.extract(text)
        payload = json.dumps(result, ensure_ascii=False, indent=2)

        if args.out:
            Path(args.out).write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return

    if args.command == "process-text":
        text = _load_text(args)
        if not text:
            raise SystemExit("Empty text input.")

        config = PipelineConfig(
            segmenter_model_path=_require(args.segmenter_model, "--segmenter-model"),
            topic_model_path=_require(args.topic_model, "--topic-model"),
            scale_model_path=_require(args.scale_model, "--scale-model"),
            extractor_model_path=_require(args.extractor_model, "--extractor-model"),
            device=args.device,
        )

        try:
            from .pipeline import Pipeline
        except ModuleNotFoundError as exc:
            if _is_missing_ml_dep(exc):
                raise SystemExit(_ml_missing_message()) from exc
            raise

        report = Pipeline(config).process_text(text)
        payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)

        if args.out:
            Path(args.out).write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return

    if args.command == "process-audio":
        config = PipelineConfig(
            segmenter_model_path=_require(args.segmenter_model, "--segmenter-model"),
            topic_model_path=_require(args.topic_model, "--topic-model"),
            scale_model_path=_require(args.scale_model, "--scale-model"),
            extractor_model_path=_require(args.extractor_model, "--extractor-model"),
            asr_model=args.asr_model,
            asr_language=args.asr_language,
            device=args.device,
        )

        try:
            from .pipeline import Pipeline
        except ModuleNotFoundError as exc:
            if _is_missing_ml_dep(exc):
                raise SystemExit(_ml_missing_message()) from exc
            raise

        report = Pipeline(config).process_audio(args.audio)
        payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)

        if args.out:
            Path(args.out).write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return


if __name__ == "__main__":
    main()
