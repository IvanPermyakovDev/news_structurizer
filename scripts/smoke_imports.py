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

    import news_structurizer  # noqa: F401
    from news_structurizer import AttributeExtractor, NewsClassifier, NewsSegmenter, Pipeline  # noqa: F401

    print("Imports: OK")


if __name__ == "__main__":
    main()

