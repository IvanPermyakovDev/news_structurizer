import json
import random

random.seed(42)

with open("dataset_kz.json", "r", encoding="utf-8") as f:
    data = json.load(f)

random.shuffle(data)

split_idx = int(len(data) * 0.9)
train_data = data[:split_idx]
val_data = data[split_idx:]

with open("dataset_kz_train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open("dataset_kz_val.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=4)

print(f"Total: {len(data)}")
print(f"Train: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
print(f"Val: {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")
