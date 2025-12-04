import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from tqdm import tqdm

from dataset import NewsSegmentationDataset, NewsSegmentationCollator
from model import create_model
from metrics import compute_boundary_metrics, get_classification_report


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/num_batches:.4f}'})

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int
) -> dict:
    model.eval()
    total_loss = 0
    num_batches = 0

    all_predictions = []
    all_labels = []
    all_attention = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss']

        total_loss += loss.item()
        num_batches += 1

        predictions = torch.argmax(outputs['logits'], dim=-1)
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())
        all_attention.append(attention_mask.cpu())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_attention = torch.cat(all_attention, dim=0)

    metrics = compute_boundary_metrics(all_predictions, all_labels, all_attention)
    metrics['loss'] = total_loss / num_batches

    return metrics, all_predictions, all_labels


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    base_dir = Path(__file__).parent.parent
    train_path = base_dir / "dataset_train.json"
    val_path = base_dir / "dataset_val.json"

    if not train_path.exists() or not val_path.exists():
        print("Splitting dataset...")
        from split_dataset import split_dataset
        split_dataset(
            str(base_dir / "dataset.json"),
            str(train_path),
            str(val_path)
        )

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Creating datasets...")
    train_dataset = NewsSegmentationDataset(
        str(train_path),
        tokenizer,
        max_length=args.max_length,
        min_news=args.min_news,
        max_news=args.max_news,
        samples_per_epoch=args.train_samples,
        seed=args.seed
    )

    val_dataset = NewsSegmentationDataset(
        str(val_path),
        tokenizer,
        max_length=args.max_length,
        min_news=args.min_news,
        max_news=args.max_news,
        samples_per_epoch=args.val_samples,
        seed=args.seed + 100
    )

    collator = NewsSegmentationCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers
    )

    print(f"Creating model: {args.model_type}")
    model = create_model(
        model_type=args.model_type,
        model_name=args.model_name,
        dropout=args.dropout
    )
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0
    history = []

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics, val_preds, val_labels = evaluate(model, val_loader, device, epoch)
        scheduler.step()

        epoch_stats = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_accuracy': val_metrics['accuracy'],
            'val_pk': val_metrics['pk'],
            'val_window_diff': val_metrics['window_diff'],
            'lr': scheduler.get_last_lr()[0]
        }
        history.append(epoch_stats)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        print(f"Train Loss:     {train_loss:.4f}")
        print(f"Val Loss:       {val_metrics['loss']:.4f}")
        print(f"{'─'*60}")
        print(f"Boundary Detection Metrics:")
        print(f"  Precision:    {val_metrics['precision']:.4f}")
        print(f"  Recall:       {val_metrics['recall']:.4f}")
        print(f"  F1:           {val_metrics['f1']:.4f}")
        print(f"  Accuracy:     {val_metrics['accuracy']:.4f}")
        print(f"{'─'*60}")
        print(f"Segmentation Metrics (lower is better):")
        print(f"  Pk:           {val_metrics['pk']:.4f}")
        print(f"  WindowDiff:   {val_metrics['window_diff']:.4f}")
        print(f"{'─'*60}")
        print(f"Learning Rate:  {scheduler.get_last_lr()[0]:.6f}")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'args': vars(args)
            }, save_dir / 'best_model.pt')
            print(f"  >> New best model saved! F1: {best_f1:.4f}")

        if epoch == args.epochs:
            print(f"\n{'='*60}")
            print("Classification Report (final epoch):")
            print("="*60)
            print(get_classification_report(val_preds, val_labels))

    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)
    }, save_dir / 'last_model.pt')

    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Models saved to: {save_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Train news segmentation model")

    parser.add_argument('--model_name', type=str, default='ai-forever/ruBert-base',
                        help='Pretrained model name')
    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['transformer', 'bilstm'],
                        help='Model architecture type')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--min_news', type=int, default=2,
                        help='Minimum news items to concatenate')
    parser.add_argument('--max_news', type=int, default=4,
                        help='Maximum news items to concatenate')
    parser.add_argument('--train_samples', type=int, default=5000,
                        help='Number of training samples per epoch')
    parser.add_argument('--val_samples', type=int, default=1000,
                        help='Number of validation samples')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save models')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
