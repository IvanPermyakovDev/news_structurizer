from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> None:
    _ensure_src_on_path()

    try:
        from news_structurizer import (  # noqa: F401
            AttributeExtractor,
            NewsClassifier,
            NewsSegmenter,
            Pipeline,
            WhisperTranscriber,
        )
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "transformers"}:
            raise SystemExit(
                "Missing ML dependencies. Install extras and retry:\n"
                "  pip install -e '.[ml]'\n"
                "or for dev:\n"
                "  pip install -e '.[dev,ml]'\n"
            ) from exc
        raise

    print("ML imports: OK")


if __name__ == "__main__":
    main()
