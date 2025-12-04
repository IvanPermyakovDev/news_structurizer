"""Training entrypoint for the hierarchical boundary tagger."""
from __future__ import annotations

import argparse
import atexit
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from .config import (
    DEFAULT_CHUNKING,
    DEFAULT_MODEL,
    DEFAULT_SOFT_BREAK,
    DEFAULT_TRAINING,
    ChunkingConfig,
    ModelConfig,
    TrainingConfig,
)
from .data_prep import build_synthetic_sequences, load_texts, split_dataset
from .dataset import BoundaryDataset, build_collate_fn
from .metrics import MetricAccumulator
from .model import BoundarySegmenter
from .soft_break import SoftBreakConfig, SoftBreakDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the boundary segmentation model")
    parser.add_argument("--dataset", type=Path, default=Path("dataset.json"), help="Path to dataset.json")
    parser.add_argument("--output", type=Path, default=Path("segmentation/artifacts"), help="Directory to store checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=DEFAULT_TRAINING.num_synthetic_sequences)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULT_TRAINING.lr)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MODEL.max_seq_len)
    parser.add_argument("--min-chunk", type=int, default=DEFAULT_CHUNKING.min_words)
    parser.add_argument("--max-chunk", type=int, default=DEFAULT_CHUNKING.max_words)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING.seed)
    parser.add_argument("--val-split", type=float, default=DEFAULT_TRAINING.val_split)
    parser.add_argument("--test-split", type=float, default=DEFAULT_TRAINING.test_split)
    parser.add_argument("--pretrained", type=str, default=DEFAULT_MODEL.pretrained_model_name)
    parser.add_argument("--chunk-batch", type=int, default=DEFAULT_MODEL.chunk_batch_size, help="Number of chunks encoded per Transformer call")
    parser.add_argument("--max-chunks-per-seq", type=int, default=DEFAULT_TRAINING.max_chunks_per_sequence, help="Upper bound on chunks per synthetic sequence")
    parser.add_argument("--pooling", type=str, default=DEFAULT_MODEL.pooling, choices=["cls", "mean"], help="Pooling for chunk embeddings")
    parser.add_argument("--no-positional", action="store_true", help="Disable learned positional embeddings over chunks")
    parser.add_argument("--positional-max", type=int, default=DEFAULT_MODEL.positional_max_positions)
    parser.add_argument("--classifier-dropout", type=float, default=DEFAULT_MODEL.classifier_dropout)
    parser.add_argument("--use-crf", action="store_true", help="Use CRF loss over boundary tags")
    parser.add_argument("--use-contrastive", action="store_true", help="Add contrastive triplet loss near boundaries")
    parser.add_argument("--contrastive-weight", type=float, default=DEFAULT_MODEL.contrastive_weight)
    parser.add_argument("--triplet-margin", type=float, default=DEFAULT_MODEL.triplet_margin)
    parser.add_argument("--min-articles", type=int, default=DEFAULT_TRAINING.min_articles, help="Minimum number of news pieces per synthetic sequence")
    parser.add_argument("--max-articles", type=int, default=DEFAULT_TRAINING.max_articles, help="Maximum number of news pieces per synthetic sequence")
    parser.add_argument("--log-dir", type=Path, default=None, help="TensorBoard log directory (defaults to OUTPUT/tb)")
    parser.add_argument("--tensorboard-port", type=int, default=6006, help="Port for the TensorBoard server")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable automatic TensorBoard launch")
    parser.add_argument("--soft-break-model", type=Path, default=None, help="Path to save/load the soft break detector state")
    parser.add_argument("--soft-break-window", type=int, default=DEFAULT_SOFT_BREAK.window_size)
    parser.add_argument("--soft-break-epochs", type=int, default=DEFAULT_SOFT_BREAK.epochs)
    parser.add_argument("--soft-break-lr", type=float, default=DEFAULT_SOFT_BREAK.lr)
    parser.add_argument("--soft-break-threshold", type=float, default=DEFAULT_CHUNKING.break_threshold)
    parser.add_argument("--soft-break-max-positive", type=int, default=DEFAULT_SOFT_BREAK.max_positive)
    parser.add_argument("--soft-break-max-negative", type=int, default=DEFAULT_SOFT_BREAK.max_negative)
    parser.add_argument("--soft-break-seed", type=int, default=DEFAULT_SOFT_BREAK.seed)
    parser.add_argument("--disable-soft-break", action="store_true", help="Fallback to heuristic chunking without adaptive soft breaks")
    parser.add_argument("--force-soft-break-train", action="store_true", help="Retrain the soft break detector even if a checkpoint exists")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight(examples: Sequence[Dict[str, List[int]]]) -> float:
    positives = 0
    total = 0
    for example in examples:
        positives += sum(example["labels"])
        total += len(example["labels"])
    negatives = max(total - positives, 1)
    positives = max(positives, 1)
    return negatives / positives


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def build_checkpoint_state(
    model: BoundarySegmenter,
    model_cfg: ModelConfig,
    chunk_cfg: ChunkingConfig,
    epoch: int,
    soft_break_detector: Optional[SoftBreakDetector],
) -> Dict[str, object]:
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "tokenizer_name": model_cfg.pretrained_model_name,
        "chunk_config": chunk_cfg.__dict__,
        "model_config": model_cfg.__dict__,
    }
    if soft_break_detector is not None:
        state["soft_break_state"] = soft_break_detector.state_dict()
    return state


def launch_tensorboard(log_dir: Path, port: int) -> Optional[subprocess.Popen]:
    try:
        process = subprocess.Popen(
            [
                "tensorboard",
                f"--logdir={log_dir}",
                f"--port={port}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("TensorBoard executable not found. Install it (`pip install tensorboard`) to enable live metrics.")
        return None
    print(f"TensorBoard is running at http://localhost:{port} (logdir={log_dir})")

    def _cleanup() -> None:
        if process.poll() is None:
            process.terminate()

    atexit.register(_cleanup)
    return process


def run_epoch(
    model: BoundarySegmenter,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    pos_weight: float,
    current_epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_steps = 0
    progress = tqdm(dataloader, desc=f"Train {current_epoch}", leave=False, ncols=100)
    for batch in progress:
        batch = move_to_device(batch, device)
        outputs = model(**batch, pos_weight=pos_weight)
        loss = outputs.loss
        if loss is None:
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_TRAINING.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        total_steps += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")
    progress.close()
    return total_loss / max(total_steps, 1)


def evaluate(
    model: BoundarySegmenter,
    dataloader: DataLoader,
    device: torch.device,
    pos_weight: float,
    phase: str,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    accumulator = MetricAccumulator()
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        progress = tqdm(dataloader, desc=phase, leave=False, ncols=100)
        for batch in progress:
            batch = move_to_device(batch, device)
            outputs = model(**batch, pos_weight=pos_weight)
            if outputs.loss is not None:
                total_loss += outputs.loss.item()
                total_steps += 1
            accumulator.update(outputs.logits.cpu(), batch["labels"].cpu(), batch["chunk_mask"].cpu())
        progress.close()
    metrics = accumulator.compute()
    return total_loss / max(total_steps, 1), metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir or (output_dir / "tb")
    log_dir.mkdir(parents=True, exist_ok=True)

    soft_break_cfg = SoftBreakConfig(
        window_size=args.soft_break_window,
        epochs=args.soft_break_epochs,
        lr=args.soft_break_lr,
        threshold=args.soft_break_threshold,
        max_positive=args.soft_break_max_positive,
        max_negative=args.soft_break_max_negative,
        seed=args.soft_break_seed,
    )

    chunk_cfg = ChunkingConfig(min_words=args.min_chunk, max_words=args.max_chunk)
    model_cfg = ModelConfig(
        pretrained_model_name=args.pretrained,
        max_seq_len=args.max_seq_len,
        context_layers=DEFAULT_MODEL.context_layers,
        context_heads=DEFAULT_MODEL.context_heads,
        context_dropout=DEFAULT_MODEL.context_dropout,
        chunk_batch_size=args.chunk_batch,
        pooling=args.pooling,
        use_positional=not args.no_positional,
        positional_max_positions=args.positional_max,
        classifier_dropout=args.classifier_dropout,
        use_crf=args.use_crf,
        use_contrastive=args.use_contrastive,
        contrastive_weight=args.contrastive_weight,
        triplet_margin=args.triplet_margin,
    )
    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=DEFAULT_TRAINING.weight_decay,
        warmup_steps=DEFAULT_TRAINING.warmup_steps,
        grad_clip=DEFAULT_TRAINING.grad_clip,
        num_synthetic_sequences=args.samples,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        max_chunks_per_sequence=args.max_chunks_per_seq,
        min_articles=args.min_articles,
        max_articles=args.max_articles,
    )

    writer = SummaryWriter(log_dir=str(log_dir))
    if not args.no_tensorboard:
        launch_tensorboard(log_dir, args.tensorboard_port)

    texts = load_texts(args.dataset)
    soft_break_detector: Optional[SoftBreakDetector] = None
    soft_break_path = args.soft_break_model or (output_dir / "soft_break.json")
    if not args.disable_soft_break:
        if soft_break_path.exists() and not args.force_soft_break_train:
            soft_break_detector = SoftBreakDetector.load(soft_break_path)
            print(f"Loaded soft break detector from {soft_break_path}")
        else:
            print("Training soft break detector...")
            soft_break_detector = SoftBreakDetector(soft_break_cfg)
            soft_break_detector.train(texts, DEFAULT_CHUNKING.soft_breaks)
            soft_break_path.parent.mkdir(parents=True, exist_ok=True)
            soft_break_detector.save(soft_break_path)
            print(f"Soft break detector saved to {soft_break_path}")
    examples = build_synthetic_sequences(
        texts,
        num_samples=train_cfg.num_synthetic_sequences,
        chunk_cfg=chunk_cfg,
        seed=train_cfg.seed,
        max_chunks_per_sequence=train_cfg.max_chunks_per_sequence,
        min_articles=train_cfg.min_articles,
        max_articles=train_cfg.max_articles,
        soft_break_detector=soft_break_detector,
    )
    train_examples, val_examples, test_examples = split_dataset(
        examples,
        val_split=train_cfg.val_split,
        test_split=train_cfg.test_split,
        seed=train_cfg.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.pretrained_model_name)
    collate_fn = build_collate_fn(tokenizer, model_cfg.max_seq_len)

    train_loader = DataLoader(
        BoundaryDataset(train_examples),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        BoundaryDataset(val_examples),
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        BoundaryDataset(test_examples),
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = BoundarySegmenter(model_cfg)
    device = torch.device(args.device)
    model.to(device)

    pos_weight = compute_pos_weight(train_examples)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    total_steps = len(train_loader) * train_cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, device, pos_weight, epoch)
        val_loss, val_metrics = evaluate(model, val_loader, device, pos_weight, "Val")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        for metric_name, metric_value in val_metrics.items():
            writer.add_scalar(f"Val/{metric_name}", metric_value, epoch)
        writer.flush()
        f1 = val_metrics.get("f1", 0.0)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={f1:.3f} window_diff={val_metrics.get('window_diff', 0.0):.3f} pk={val_metrics.get('pk', 0.0):.3f}"
        )
        checkpoint_path = output_dir / "segmenter_last.pt"
        last_state = build_checkpoint_state(model, model_cfg, chunk_cfg, epoch, soft_break_detector)
        torch.save(last_state, checkpoint_path)
        print(f"Saved last checkpoint (epoch {epoch}) to {checkpoint_path}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = last_state

    best_path = output_dir / "segmenter_best.pt"
    if best_state is None:
        best_state = build_checkpoint_state(model, model_cfg, chunk_cfg, train_cfg.epochs, soft_break_detector)
    torch.save(best_state, best_path)
    print(f"Best checkpoint saved to {best_path}")

    test_loss, test_metrics = evaluate(model, test_loader, device, pos_weight, "Test")
    writer.add_scalar("Loss/test", test_loss, train_cfg.epochs)
    for metric_name, metric_value in test_metrics.items():
        writer.add_scalar(f"Test/{metric_name}", metric_value, train_cfg.epochs)
    writer.flush()
    writer.close()

    report = {
        "test_loss": test_loss,
        "metrics": test_metrics,
        "best_val_f1": best_val_f1,
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("Saved checkpoint to", checkpoint_path)
    print("Test metrics:", json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
