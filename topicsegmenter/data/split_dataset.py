import json
import random
from pathlib import Path

random.seed(42)

data_dir = Path(__file__).parent
input_file = data_dir / "dataset_segmentor_kz_all_cleaned.jsonl"

# Читаем весь файл и парсим как массив JSON объектов
with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# Парсим многострочные JSON объекты
decoder = json.JSONDecoder()
data = []
idx = 0
content = content.strip()
required_keys = {"context_left", "context_right", "label"}
while idx < len(content):
    # Пропускаем пробельные символы
    while idx < len(content) and content[idx] in ' \t\n\r':
        idx += 1
    if idx >= len(content):
        break
    obj, length = decoder.raw_decode(content[idx:])
    # Фильтруем записи с неполными данными
    if required_keys <= obj.keys():
        data.append(obj)
    idx += length

# Перемешиваем
random.shuffle(data)

# Разбиваем 90/10
split_idx = int(len(data) * 0.9)
train_data = data[:split_idx]
val_data = data[split_idx:]

# Сохраняем как однострочный JSONL
with open(data_dir / "train.jsonl", "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(data_dir / "val.jsonl", "w", encoding="utf-8") as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Total: {len(data)}")
print(f"Train: {len(train_data)}")
print(f"Val: {len(val_data)}")
