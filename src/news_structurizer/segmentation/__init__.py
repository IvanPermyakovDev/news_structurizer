from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .topicsegmenter import NewsSegmenter

__all__ = ["NewsSegmenter"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.topicsegmenter")
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
