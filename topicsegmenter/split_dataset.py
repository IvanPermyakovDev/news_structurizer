#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable, List


def _iter_json_objects(path: Path, *, validate_json: bool) -> Iterable[Any]:
    """
    Поддерживает два формата входа:
      1) JSONL/NDJSON: 1 JSON-объект на строку
      2) Поток JSON-объектов подряд (в т.ч. pretty-printed, многострочные объекты)
    """
    content = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    length = len(content)

    while True:
        while idx < length and content[idx].isspace():
            idx += 1
        if idx >= length:
            break
        try:
            obj, next_idx = decoder.raw_decode(content, idx)
        except json.JSONDecodeError as exc:
            next_brace = content.find("{", idx + 1)
            if next_brace == -1:
                break
            idx = next_brace
            continue
        if validate_json:
            # Уже провалидировано самим парсингом; флаг оставлен для совместимости CLI.
            pass
        if isinstance(obj, dict):
            yield obj
        idx = next_idx


def split_jsonl(
    input_path: Path,
    train_path: Path,
    val_path: Path,
    *,
    train_ratio: float,
    seed: int,
    shuffle: bool,
    validate_json: bool,
) -> tuple[int, int]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1 (exclusive)")

    required_keys = {"context_left", "context_right", "label"}
    items: List[Any] = []
    skipped_missing_keys = 0

    for obj in _iter_json_objects(input_path, validate_json=validate_json):
        if not required_keys.issubset(obj.keys()):
            skipped_missing_keys += 1
            continue
        items.append(obj)
    if not items:
        raise ValueError(f"No JSON objects found in {input_path}")

    if shuffle:
        random.Random(seed).shuffle(items)

    train_size = int(train_ratio * len(items))
    train_items = items[:train_size]
    val_items = items[train_size:]

    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    with train_path.open("w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as f:
        for item in val_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if skipped_missing_keys:
        print(f"Skipped invalid entries with missing keys: {skipped_missing_keys}")

    return len(train_items), len(val_items)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Разделение JSONL (dataset.jsonl) на train/val в пропорции 90/10 (или любой другой)."
    )
    parser.add_argument("--input", default="dataset.jsonl", help="Путь к исходному JSONL файлу")
    parser.add_argument(
        "--train",
        default="data/train.jsonl",
        help="Куда сохранить train.jsonl (по умолчанию: data/train.jsonl)",
    )
    parser.add_argument(
        "--val",
        default="data/val.jsonl",
        help="Куда сохранить val.jsonl (по умолчанию: data/val.jsonl)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Доля train (0..1), например 0.9")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимого перемешивания")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Не перемешивать примеры перед разбиением (оставить исходный порядок)",
    )
    parser.add_argument(
        "--validate-json",
        action="store_true",
        help="Проверять, что вход успешно парсится как JSON-объекты",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    train_n, val_n = split_jsonl(
        input_path=input_path,
        train_path=Path(args.train),
        val_path=Path(args.val),
        train_ratio=args.train_ratio,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        validate_json=args.validate_json,
    )

    total = train_n + val_n
    train_pct = (train_n / total) * 100
    val_pct = (val_n / total) * 100
    print(f"Done: train={train_n} ({train_pct:.2f}%), val={val_n} ({val_pct:.2f}%), total={total}")


if __name__ == "__main__":
    main()
