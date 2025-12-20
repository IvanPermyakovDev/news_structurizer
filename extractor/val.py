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
            early_stopping=True,
            # ВАЖНО: Убирает зацикливания ("Мадина Мадина Мадина")
            no_repeat_ngram_size=2, 
            repetition_penalty=1.2
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
    try:
        data = load_data(data_path)
    except FileNotFoundError:
        print("[ERROR] Файл данных не найден. Проверьте путь.")
        return
    
    print("[INFO] Инициализация метрик (BERTScore и ROUGE)...")
    bertscore = evaluate.load("bertscore")
    rouge = evaluate.load("rouge")

    tasks = {
        "title": "тақырып: ",
        "key_events": "оқиға: ",
        "location": "орын: ",
        "key_names": "есімдер: "
    }

    all_predictions = []
    all_references = []
    
    task_data = {k: {'preds': [], 'refs': []} for k in tasks.keys()}

    print(f"[INFO] Начало генерации на устройстве: {DEVICE}")
    
    # Итерируемся с TQDM
    for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Обработка батчей"):
        batch_items = data[i : i + BATCH_SIZE]
        batch_texts = [item.get('text', '') for item in batch_items]
        
        for json_key, prefix in tasks.items():
            current_refs = []
            valid_indices = []
            
            for idx, item in enumerate(batch_items):
                # Логика извлечения таргета
                target = None
                if json_key in item:
                    target = item[json_key]
                
                # Если локация пустая, ставим "Белгісіз", если другие поля — пропускаем? 
                # (Тут ваша логика из обучения: title, key_events обязательны, location может быть пустой)
                if json_key == 'location' and not target:
                    target = "Белгісіз"
                
                if target:
                    current_refs.append(str(target))
                    valid_indices.append(idx)

            if not valid_indices:
                continue

            texts_to_process = [batch_texts[j] for j in valid_indices]
            
            # Генерация
            preds = generate_batch(model, tokenizer, texts_to_process, prefix)
            
            all_predictions.extend(preds)
            all_references.extend(current_refs)
            
            task_data[json_key]['preds'].extend(preds)
            task_data[json_key]['refs'].extend(current_refs)

    print("\n[INFO] Вычисление метрик...")

    # 1. BERTScore
    # Используем стандартный mBERT, если ваша модель не сработает
    # bert-base-multilingual-cased — это стандарт де-факто для оценки >100 языков
    BS_MODEL = "bert-base-multilingual-cased" 
    
    try:
        print(f"[INFO] Считаем BERTScore используя {BS_MODEL}...")
        bs_results = bertscore.compute(
            predictions=all_predictions,
            references=all_references,
            model_type=BS_MODEL,
            lang="kz" # Можно указать язык, но для mBERT это не критично
        )
        bs_f1_mean = np.mean(bs_results['f1'])
    except Exception as e:
        print(f"[WARNING] Ошибка расчета BERTScore: {e}")
        bs_f1_mean = 0.0

    # 2. ROUGE (С ИСПРАВЛЕНИЕМ TOKENIZER)
    rouge_results = rouge.compute(
        predictions=all_predictions, 
        references=all_references,
        tokenizer=lambda x: x.split() # <--- ОБЯЗАТЕЛЬНО ДЛЯ КАЗАХСКОГО
    )

    print("-" * 60)
    print("ОБЩИЕ РЕЗУЛЬТАТЫ (Среднее по всем задачам)")
    print("-" * 60)
    print(f"BERTScore F1: {bs_f1_mean:.4f}")
    print(f"ROUGE-1:      {rouge_results['rouge1'] * 100:.2f}")
    print(f"ROUGE-2:      {rouge_results['rouge2'] * 100:.2f}")
    print(f"ROUGE-L:      {rouge_results['rougeL'] * 100:.2f}")
    print("-" * 60)
    print("\nДЕТАЛИЗАЦИЯ ПО АТРИБУТАМ (BERTScore F1):")
    
    for task, content in task_data.items():
        if not content['preds']:
            continue
        
        # Считаем BS для конкретной задачи (быстро, так как модель закеширована)
        try:
            res = bertscore.compute(
                predictions=content['preds'],
                references=content['refs'],
                model_type=BS_MODEL,
                lang="kz"
            )
            score = np.mean(res['f1'])
        except:
            score = 0.0
        
        print(f"Task: {task:<15} Score: {score:.4f}")
        # Показываем 3 случайных примера, а не всегда первый
        if len(content['preds']) > 0:
            import random
            idx = random.randint(0, len(content['preds']) - 1)
            print(f"  [Random Ref]:  {content['refs'][idx]}")
            print(f"  [Prediction]:  {content['preds'][idx]}")
        print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Валидация mT5 (казахский)")
    parser.add_argument("--model_dir", type=str, required=True, help="Путь к папке с моделью (например ./final_model)")
    parser.add_argument("--data_path", type=str, default="../data/dataset_kz_val.json", help="Путь к валидационному датасету")
    
    args = parser.parse_args()
    
    evaluate_model(args.model_dir, args.data_path)