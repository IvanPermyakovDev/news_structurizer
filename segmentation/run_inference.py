"""CLI wrapper for inference with the boundary segmenter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .segment_text import load_segmenter, segment_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run boundary segmentation on a text or file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained checkpoint (.pt)")
    parser.add_argument("--text", type=str, default=None, help="Input text")
    parser.add_argument("--input-file", type=Path, default=None, help="Path to a file with raw text")
    parser.add_argument("--k-known", type=int, default=None, help="Optional known number of news items")
    parser.add_argument("--percentile", type=float, default=0.9, help="Quantile for adaptive threshold when k is unknown")
    parser.add_argument("--min-gap", type=int, default=2, help="Minimum gap between predicted boundaries (in chunks)")
    parser.add_argument("--json-out", type=Path, default=None, help="Path to dump full JSON result")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.text is None and args.input_file is None:
        raise SystemExit("Provide --text or --input-file")
    if args.input_file:
        args.text = Path(args.input_file).read_text(encoding="utf-8")

    model, tokenizer, chunk_cfg, model_cfg, soft_break = load_segmenter(args.checkpoint, device=args.device)
    result = segment_text(
        args.text,
        model,
        tokenizer,
        model_cfg=model_cfg,
        chunk_cfg=chunk_cfg,
        device=args.device,
        k_known=args.k_known,
        percentile=args.percentile,
        min_gap_chunks=args.min_gap,
        soft_break_detector=soft_break,
    )

    print(f"\nFound {len(result['segments'])} segments:\n")
    for idx, seg in enumerate(result["segments"], 1):
        print(f"--- Segment {idx} ---")
        print(seg)
        print()

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Wrote JSON output to {args.json_out}")


if __name__ == "__main__":
    main()
