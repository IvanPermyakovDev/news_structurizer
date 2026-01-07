from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _load_text(args: argparse.Namespace) -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8").strip()
    if args.text:
        return args.text.strip()
    if sys.stdin.isatty():
        return DEFAULT_TEXT
    stdin_text = sys.stdin.read().strip()
    return stdin_text or DEFAULT_TEXT


DEFAULT_TEXT = (
    "бүгін астанада ауа райы суық болады түнде температура минус он бес градусқа дейін "
    "түседі ал енді спорт жаңалықтарына көшейік алматыдағы футбол ойынында қайрат командасы "
    "актобені үш бір есебімен ұтты шешуші голды екінші таймда соқты"
)


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

    parser = argparse.ArgumentParser(
        description="Run full text pipeline (segment -> classify -> extract)."
    )
    parser.add_argument("--text", help="Input text (if omitted, reads stdin).")
    parser.add_argument("--file", help="Path to text file.")
    parser.add_argument("--out", help="Write report JSON to a file (optional).")
    parser.add_argument("--segmenter-model", default=str(default_segmenter))
    parser.add_argument("--topic-model", default=str(default_topic))
    parser.add_argument("--scale-model", default=str(default_scale))
    parser.add_argument("--extractor-model", default=str(default_extractor))
    parser.add_argument("--device", default=None, help="cpu / cuda / cuda:0 ... (optional)")
    args = parser.parse_args()

    text = _load_text(args)
    if not text:
        raise SystemExit("Empty text input.")

    cfg = PipelineConfig(
        segmenter_model_path=args.segmenter_model,
        topic_model_path=args.topic_model,
        scale_model_path=args.scale_model,
        extractor_model_path=args.extractor_model,
        device=args.device,
    )

    report = Pipeline(cfg).process_text(text)
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)

    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
