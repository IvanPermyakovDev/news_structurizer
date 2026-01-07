from __future__ import annotations

import sys

import pytest

from news_structurizer.cli import build_parser, main


def test_build_parser_supports_help() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0


def test_main_help_exits_zero() -> None:
    argv = sys.argv[:]
    try:
        sys.argv = ["news-structurizer", "--help"]
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0
    finally:
        sys.argv = argv
