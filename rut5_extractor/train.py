import json
import pandas as pd
import os
import subprocess
import torch
import numpy as np
import evaluate
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)
from transformers.trainer_callback import ProgressCallback, PrinterCallback

class SmartProgressCallback(ProgressCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        super().on_log(args, state, control, logs, **kwargs)
        if self.training_bar is not None and logs is not None:
            _postfix = {}
            if 'loss' in logs:
                _postfix['loss'] = f"{logs['loss']:.4f}"
            if 'eval_loss' in logs:
                _postfix['val_loss'] = f"{logs['eval_loss']:.4f}"
            if 'eval_rouge1' in logs:
                _postfix['rouge1'] = f"{logs['eval_rouge1']:.2f}"
            if 'epoch' in logs:
                _postfix['ep'] = f"{logs['epoch']:.2f}"
            self.training_bar.set_postfix(_postfix)

def launch_tensorboard(log_dir):
    try:
        print(f"[Info] TensorBoard: {log_dir}")
        tb_process = subprocess.Popen(
            ["tensorboard", "--logdir", log_dir, "--port", "6006"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("[Info] URL: http://localhost:6006")
        return tb_process
    except FileNotFoundError:
        return None

def prepare_dataset_rows(raw_data):
    rows = []
    for item in raw_data:
        text = item.get('text', '')
        if not text: continue
        
        tasks = [
            ('title', 'заголовок: '),
            ('key_events', 'событие: '),
            ('location', 'локация: '),
            ('key_names', 'имена: ')
        ]
        
        for json_key, prefix in tasks:
            if json_key in item:
                target = item[json_key]
                if json_key == 'location' and not target:
                    target = "Неизвестно"
                rows.append({
                    "input_text": f"{prefix}{text}", 
                    "target_text": str(target)
                })
    return rows

def load_data():
    path_train = 'dataset_train.json'
    path_val = 'dataset_val.json'
    path_single = 'dataset.json'

    if not os.path.exists(path_single) and not os.path.exists(path_train):
        parent_dir = os.path.dirname(__file__) if '__file__' in globals() else '..'
        path_train = os.path.join(parent_dir, 'dataset_train.json')
        path_val = os.path.join(parent_dir, 'dataset_val.json')
        path_single = os.path.join(parent_dir, 'dataset.json')

    if os.path.exists(path_train) and os.path.exists(path_val):
        print(f"[Data] Найдены разделенные файлы.")
        with open(path_train, 'r', encoding='utf-8') as f: train_data = json.load(f)
        with open(path_val, 'r', encoding='utf-8') as f: val_data = json.load(f)
        train_df = pd.DataFrame(prepare_dataset_rows(train_data))
        val_df = pd.DataFrame(prepare_dataset_rows(val_data))
        
    elif os.path.exists(path_single):
        print(f"[Data] Найден общий файл. Разбиение...")
        with open(path_single, 'r', encoding='utf-8') as f: raw_data = json.load(f)
        train_raw, val_raw = train_test_split(raw_data, test_size=0.1, random_state=42)
        
        with open('dataset_train.json', 'w', encoding='utf-8') as f: json.dump(train_raw, f, ensure_ascii=False, indent=2)
        with open('dataset_val.json', 'w', encoding='utf-8') as f: json.dump(val_raw, f, ensure_ascii=False, indent=2)
            
        train_df = pd.DataFrame(prepare_dataset_rows(train_raw))
        val_df = pd.DataFrame(prepare_dataset_rows(val_raw))
    else:
        raise FileNotFoundError("Датасет не найден")

    print(f"[Data] Train: {len(train_df)}, Val: {len(val_df)}")
    return train_df, val_df

def main():
    train_df, val_df = load_data()
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df)
    })

    MODEL_NAME = "ai-forever/ruT5-base"
    MAX_INPUT_LENGTH = 1024 
    MAX_TARGET_LENGTH = 128
    
    print(f"[Model] Загрузка {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    rouge = evaluate.load("rouge")

    def preprocess_function(examples):
        model_inputs = tokenizer(examples["input_text"], max_length=MAX_INPUT_LENGTH, padding="max_length", truncation=True)
        labels = tokenizer(examples["target_text"], max_length=MAX_TARGET_LENGTH, padding="max_length", truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(eval_pred):
        predictions, labels, inputs = eval_pred.predictions, eval_pred.label_ids, eval_pred.inputs

        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        
        print("\n" + "="*60)
        print("ВАЛИДАЦИЯ (Примеры):")
        indices = np.random.choice(len(decoded_preds), min(3, len(decoded_preds)), replace=False)
        for i in indices:
            print(f"\n[Пример {i}]")
            short_input = decoded_inputs[i][:50] + "..." if len(decoded_inputs[i]) > 50 else decoded_inputs[i]
            print(f"Вход:    {short_input}")
            print(f"Эталон:  {decoded_labels[i]}")
            print(f"Модель:  {decoded_preds[i]}")
        print("="*60 + "\n")

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return {k: float(round(v * 100, 4)) for k, v in result.items()}

    print("[Data] Токенизация...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    output_dir = "./rut5_news_attributes"
    tb_process = launch_tensorboard(output_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_only_model=True,
        
        learning_rate=4e-5,
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=4,
        save_total_limit=2,
        num_train_epochs=10, 
        
        predict_with_generate=True,
        include_inputs_for_metrics=True,
        fp16=True if torch.cuda.is_available() else False,
        
        logging_dir=output_dir,
        logging_strategy="steps", 
        logging_steps=10,
        report_to="tensorboard",
        
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        
        disable_tqdm=False 
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(SmartProgressCallback)

    print("[Train] Старт обучения...")
    trainer.train()

    final_model_dir = "./final_model"
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"[Info] Модель сохранена: {final_model_dir}")

    if tb_process:
        try:
            tb_process.wait()
        except KeyboardInterrupt:
            tb_process.terminate()

if __name__ == "__main__":
    main()