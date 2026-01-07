from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .pipeline import Pipeline
from .schemas import PipelineConfig


def _env_default(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="news-structurizer")
    sub = parser.add_subparsers(dest="command", required=True)

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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "segment-text":
        from .segmentation import NewsSegmenter

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
        from .classification import NewsClassifier

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
        from .extraction import AttributeExtractor

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

        report = Pipeline(config).process_text(text)
        payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)

        if args.out:
            Path(args.out).write_text(payload, encoding="utf-8")
        else:
            print(payload)


if __name__ == "__main__":
    main()
