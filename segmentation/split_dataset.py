import json
import random
from pathlib import Path

def split_dataset(input_path: str, train_path: str, val_path: str, train_ratio: float = 0.8, seed: int = 42):
    random.seed(seed)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.shuffle(data)

    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"Total: {len(data)}")
    print(f"Train: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    print(f"Val: {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    split_dataset(
        input_path=str(base_dir / "dataset.json"),
        train_path=str(base_dir / "dataset_train.json"),
        val_path=str(base_dir / "dataset_val.json"),
        train_ratio=0.8
    )
