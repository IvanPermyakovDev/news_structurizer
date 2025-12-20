import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import evaluate
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase
)

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Конфигурация параметров обучения."""
    model_name: str = "google/t5gemma-2-270m-270m"
    output_dir: str = "./t5gemma_270m_kz_news_attributes_frozen"
    max_input_length: int = 1024
    max_target_length: int = 128
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 10
    warmup_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10
    seed: int = 42

def prepare_dataset_rows(raw_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Преобразует сырые данные JSON в формат prompt-completion для обучения.
    """
    rows = []
    # Маппинг ключей JSON на префиксы промптов
    task_mapping = [
        ('title', 'тақырып: '),
        ('key_events', 'оқиға: '),
        ('location', 'орын: '),
        ('key_names', 'есімдер: ')
    ]

    for item in raw_data:
        text = item.get('text', '')
        if not text:
            continue

        for json_key, prefix in task_mapping:
            if json_key not in item:
                continue
                
            target = item[json_key]
            
            # Обработка пустых значений
            if json_key == 'location':
                if not target or str(target).lower() == 'null':
                    target = "Белгісіз"
            
            if not target or str(target).lower() == 'null':
                continue

            rows.append({
                "input_text": f"{prefix}{text}",
                "target_text": str(target)
            })
    return rows

def load_data(data_dir: str = "../data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает данные из JSON файлов и возвращает DataFrame для train/val.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_data_dir = os.path.join(script_dir, data_dir)
    
    # Пути к файлам
    path_train = os.path.join(abs_data_dir, 'dataset_kz_train.json')
    path_val = os.path.join(abs_data_dir, 'dataset_kz_val.json')
    path_single = os.path.join(abs_data_dir, 'dataset_kz.json')

    # Локальный фоллбэк
    if not os.path.exists(path_train) and not os.path.exists(path_single):
        path_train = 'dataset_kz_train.json'
        path_val = 'dataset_kz_val.json'
        path_single = 'dataset_kz.json'

    if os.path.exists(path_train) and os.path.exists(path_val):
        logger.info("Loading split datasets.")
        with open(path_train, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(path_val, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        
    elif os.path.exists(path_single):
        logger.info("Loading single dataset and splitting.")
        with open(path_single, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        train_data, val_data = train_test_split(raw_data, test_size=0.1, random_state=42)
    else:
        raise FileNotFoundError(f"Dataset files not found in {abs_data_dir} or current directory.")

    train_df = pd.DataFrame(prepare_dataset_rows(train_data))
    val_df = pd.DataFrame(prepare_dataset_rows(val_data))
    
    logger.info(f"Dataset loaded. Train size: {len(train_df)}, Val size: {len(val_df)}")
    return train_df, val_df

def freeze_encoder(model: torch.nn.Module) -> None:
    """
    Замораживает веса энкодера для сохранения языковых знаний 
    и ускорения обучения.
    """
    for param in model.get_encoder().parameters():
        param.requires_grad = False
    
    for param in model.get_decoder().parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Encoder frozen. Trainable params: {trainable_params:,} / {all_params:,} "
                f"({100 * trainable_params / all_params:.2f}%)")

def get_metrics_compute_fn(tokenizer: PreTrainedTokenizerBase):
    """
    Фабрика для функции метрик ROUGE.
    """
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Декодинг с игнорированием паддингов (-100)
        decoded_preds = tokenizer.batch_decode(
            np.where(predictions != -100, predictions, tokenizer.pad_token_id),
            skip_special_tokens=True
        )
        decoded_labels = tokenizer.batch_decode(
            np.where(labels != -100, labels, tokenizer.pad_token_id),
            skip_special_tokens=True
        )

        # Логирование примеров генерации
        if len(decoded_preds) > 0:
            logger.info("Validation Example:")
            logger.info(f"Target: {decoded_labels[0]}")
            logger.info(f"Pred:   {decoded_preds[0]}")

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            tokenizer=lambda x: x.split() # Простой токенизатор для ROUGE
        )
        return {k: round(v * 100, 4) for k, v in result.items()}

    return compute_metrics

def main():
    conf = TrainingConfig()
    
    # 1. Подготовка данных
    train_df, val_df = load_data()
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df)
    })

    # 2. Инициализация модели и токенизатора
    logger.info(f"Initializing model: {conf.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(conf.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(conf.model_name)

    # 3. Заморозка энкодера
    freeze_encoder(model)

    # Патч совместимости для T5Gemma (обработка labels)
    if "t5gemma" in conf.model_name.lower() and hasattr(model, "prepare_decoder_input_ids_from_labels"):
        original_prep = model.prepare_decoder_input_ids_from_labels
        def fixed_prep(labels=None, input_ids=None, **kwargs):
            target_ids = labels if labels is not None else input_ids
            return original_prep(input_ids=target_ids, **kwargs)
        model.prepare_decoder_input_ids_from_labels = fixed_prep

    # 4. Препроцессинг
    def preprocess_function(examples):
        targets = [t + tokenizer.eos_token for t in examples["target_text"]]
        model_inputs = tokenizer(
            examples["input_text"], 
            max_length=conf.max_input_length, 
            padding="max_length", 
            truncation=True
        )
        labels = tokenizer(
            targets, 
            max_length=conf.max_target_length, 
            padding="max_length", 
            truncation=True
        )
        # Замена pad_token_id на -100 для корректного расчета loss
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] 
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    logger.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # 5. Параметры обучения
    training_args = Seq2SeqTrainingArguments(
        output_dir=conf.output_dir,
        eval_strategy="steps",
        eval_steps=conf.save_steps,
        save_strategy="steps",
        save_steps=conf.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        learning_rate=conf.learning_rate,
        per_device_train_batch_size=conf.batch_size,
        per_device_eval_batch_size=conf.batch_size,
        num_train_epochs=conf.num_epochs,
        warmup_steps=conf.warmup_steps,
        
        predict_with_generate=True,
        generation_max_length=conf.max_target_length,
        generation_num_beams=4,
        
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        
        logging_strategy="steps",
        logging_steps=conf.logging_steps,
        logging_dir=os.path.join(conf.output_dir, "runs"),
        report_to="tensorboard",
        disable_tqdm=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=get_metrics_compute_fn(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # 6. Запуск обучения
    logger.info("Starting training...")
    trainer.train()

    # 7. Сохранение артефактов
    save_path = os.path.join(conf.output_dir, "best_model")
    logger.info(f"Saving best model to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    main()