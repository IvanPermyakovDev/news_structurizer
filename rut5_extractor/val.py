import argparse
import json
import torch
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate

# Настройки
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_batch(model, tokenizer, texts, task_prefix):
    input_texts = [f"{task_prefix}{text}" for text in texts]
    
    inputs = tokenizer(
        input_texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=1024
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def evaluate_model(model_path, data_path):
    print(f"[INFO] Загрузка модели из: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"[ERROR] Ошибка загрузки модели: {e}")
        return

    print(f"[INFO] Загрузка данных из: {data_path}")
    data = load_data(data_path)
    
    print("[INFO] Инициализация метрик (BERTScore и ROUGE)...")
    bertscore = evaluate.load("bertscore")
    rouge = evaluate.load("rouge")

    tasks = {
        "title": "заголовок: ",
        "key_events": "событие: ",
        "location": "локация: ",
        "key_names": "имена: "
    }

    all_predictions = []
    all_references = []
    
    # Словарь для хранения результатов по каждой задаче отдельно
    task_data = {k: {'preds': [], 'refs': []} for k in tasks.keys()}

    print(f"[INFO] Начало генерации на устройстве: {DEVICE}")
    
    for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Обработка батчей"):
        batch_items = data[i : i + BATCH_SIZE]
        batch_texts = [item.get('text', '') for item in batch_items]
        
        for json_key, prefix in tasks.items():
            current_refs = []
            valid_indices = []
            
            for idx, item in enumerate(batch_items):
                if json_key in item and item[json_key]:
                    current_refs.append(item[json_key])
                    valid_indices.append(idx)
                elif json_key == 'location':
                     current_refs.append("Неизвестно")
                     valid_indices.append(idx)

            if not valid_indices:
                continue

            texts_to_process = [batch_texts[j] for j in valid_indices]
            
            preds = generate_batch(model, tokenizer, texts_to_process, prefix)
            
            all_predictions.extend(preds)
            all_references.extend(current_refs)
            
            task_data[json_key]['preds'].extend(preds)
            task_data[json_key]['refs'].extend(current_refs)

    print("\n[INFO] Вычисление метрик...")

    # 1. BERTScore (Семантическая близость)
    # lang="ru" использует соответствующую модель для русского языка
    try:
        bs_results = bertscore.compute(predictions=all_predictions, references=all_references, lang="ru")
        bs_f1_mean = np.mean(bs_results['f1'])
    except Exception as e:
        print(f"[WARNING] Ошибка расчета BERTScore: {e}")
        bs_f1_mean = 0.0

    # 2. ROUGE (Лексическое совпадение)
    rouge_results = rouge.compute(predictions=all_predictions, references=all_references)

    print("-" * 60)
    print("ОБЩИЕ РЕЗУЛЬТАТЫ (Усредненные по всем задачам)")
    print("-" * 60)
    print(f"BERTScore F1: {bs_f1_mean:.4f} (Максимум 1.0)")
    print(f"ROUGE-1:      {rouge_results['rouge1']:.4f}")
    print(f"ROUGE-2:      {rouge_results['rouge2']:.4f}")
    print(f"ROUGE-L:      {rouge_results['rougeL']:.4f}")
    print("-" * 60)
    print("\nДЕТАЛИЗАЦИЯ ПО АТРИБУТАМ (BERTScore F1):")
    
    for task, content in task_data.items():
        if not content['preds']: 
            continue
        
        res = bertscore.compute(predictions=content['preds'], references=content['refs'], lang="ru")
        score = np.mean(res['f1'])
        
        print(f"Task: {task:<15} Score: {score:.4f}")
        print(f"  Reference:  {content['refs'][0]}")
        print(f"  Prediction: {content['preds'][0]}")
        print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Валидация T5")
    parser.add_argument("--model_dir", type=str, default="./final_model", help="Путь к модели")
    parser.add_argument("--data_path", type=str, default="dataset_val.json", help="Путь к датасету")
    
    args = parser.parse_args()
    
    evaluate_model(args.model_dir, args.data_path)