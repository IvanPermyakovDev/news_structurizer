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
    "бүгін астанада ауа райы суық болады ал енді спорт жаңалықтарына көшейік "
    "алматыдағы футбол ойынында қайрат командасы актобені үш бір есебімен ұтты"
)


def main() -> None:
    _ensure_src_on_path()

    from news_structurizer import NewsClassifier

    repo_root = Path(__file__).resolve().parents[1]
    default_topic = (
        repo_root / "research" / "ml" / "classification" / "models_out_kz" / "topic" / "best"
    )
    default_scale = (
        repo_root / "research" / "ml" / "classification" / "models_out_kz" / "scale" / "best"
    )

    parser = argparse.ArgumentParser(description="Run only classification (text -> topic/scale).")
    parser.add_argument("--text", help="Input text (if omitted, reads stdin).")
    parser.add_argument("--file", help="Path to text file.")
    parser.add_argument(
        "--topic-model",
        default=str(default_topic),
        help="Path to topic model dir.",
    )
    parser.add_argument(
        "--scale-model",
        default=str(default_scale),
        help="Path to scale model dir.",
    )
    parser.add_argument("--device", default=None, help="cpu / cuda / cuda:0 ... (optional)")
    args = parser.parse_args()

    text = _load_text(args)
    if not text:
        raise SystemExit("Empty text input.")

    classifier = NewsClassifier(
        topic_model_path=args.topic_model,
        scale_model_path=args.scale_model,
        device=args.device,
    )
    result = classifier.classify(text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
