from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model, max_len: int) -> dict:
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(torch.argmax(probs).item())
    label = model.config.id2label[pred_idx]
    return {
        "label": label,
        "confidence": float(probs[pred_idx]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict scale label for a single text using a fine-tuned model.",
    )
    parser.add_argument(
        "--model_dir",
        default="models_out_sbert_large_nlu_ru/scale/best",
        help="Directory with the fine-tuned scale model (default: models_out/scale/best).",
    )
    parser.add_argument(
        "--text",
        help="Text to classify. If omitted, will read from stdin.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=256,
        help="Maximum sequence length to use during tokenization.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON (default is human-readable).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    text = args.text.strip() if args.text else input("Введите текст: ").strip()
    if not text:
        raise SystemExit("Пустой текст невозможно классифицировать.")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"Каталог модели не найден: {model_dir}")

    tokenizer, model = load_model(model_dir)
    result = predict(text, tokenizer, model, args.max_len)

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"Scale: {result['label']} (confidence={result['confidence']:.4f})")


if __name__ == "__main__":
    main()

