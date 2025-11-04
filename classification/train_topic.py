from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from training_common import (
    MODEL_NAME,
    TOPIC_LABELS,
    build_tokenizer,
    load_and_split_dataset,
    prepare_tokenized_dataset,
    set_seed,
    tensorboard_server,
    train_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune ruBERT on topic classification.")
    parser.add_argument("--data", required=True, help="Path to dataset JSON file.")
    parser.add_argument(
        "--out_dir",
        default="models_out/topic",
        help="Directory to store the fine-tuned model and tokenizer (default: models_out/topic).",
    )
    parser.add_argument(
        "--tb_dir",
        default=None,
        help="TensorBoard log directory (default: <out_dir>/tb).",
    )
    parser.add_argument(
        "--tb_port",
        type=int,
        default=6006,
        help="Port for the TensorBoard server.",
    )
    parser.add_argument(
        "--no_tb_server",
        action="store_true",
        help="Do not auto-launch TensorBoard even if it is installed.",
    )
    parser.add_argument(
        "--model_name",
        default=MODEL_NAME,
        help=f"Pretrained model identifier (default: {MODEL_NAME}).",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Per-device batch size.")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="Disable mixed precision training even if CUDA with FP16 is available.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.10,
        help="Fraction of the dataset reserved for the test split.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    tensorboard_installed = importlib.util.find_spec("tensorboard") is not None
    if not tensorboard_installed:
        print("[TensorBoard] Module not installed. Install `tensorboard` to enable logging.")

    set_seed(args.seed)

    dataset = load_and_split_dataset(
        json_path=args.data,
        label_column="topic",
        allowed_labels=TOPIC_LABELS,
        seed=args.seed,
        test_size=args.test_size,
    )

    tokenizer = build_tokenizer(args.model_name)
    tokenized_ds, label2id, id2label = prepare_tokenized_dataset(
        dataset,
        tokenizer=tokenizer,
        label_column="topic",
        labels=TOPIC_LABELS,
        max_length=args.max_len,
    )

    output_dir = Path(args.out_dir)
    logging_dir = Path(args.tb_dir) if args.tb_dir else output_dir / "tb"

    with tensorboard_server(
        log_dir=logging_dir,
        port=args.tb_port,
        enabled=tensorboard_installed and not args.no_tb_server,
    ):
        metrics, best_dir, last_dir = train_model(
            dataset=tokenized_ds,
            model_name=args.model_name,
            tokenizer=tokenizer,
            id2label=id2label,
            label2id=label2id,
            output_dir=output_dir,
            logging_dir=logging_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            seed=args.seed,
            use_fp16=not args.no_fp16,
            log_to_tensorboard=tensorboard_installed,
        )

    print("\n=== Topic classification: validation metrics ===")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"\nSaved best checkpoint to: {best_dir}")
    print(f"Saved last checkpoint to: {last_dir}")


if __name__ == "__main__":
    main()
