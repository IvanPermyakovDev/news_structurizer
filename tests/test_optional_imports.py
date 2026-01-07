from __future__ import annotations


def test_package_imports_without_ml_deps() -> None:
    import news_structurizer  # noqa: F401

    # Public API should be importable even when optional ML deps are not installed.
    from news_structurizer import Pipeline  # noqa: F401
    from news_structurizer import api as _api  # noqa: F401

