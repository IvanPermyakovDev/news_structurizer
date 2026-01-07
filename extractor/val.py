import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizer, PreTrainedModel
import evaluate

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class EvalConfig:
    """Конфигурация параметров валидации."""
    batch_size: int = 8
    max_input_length: int = 1024
    max_target_length: int = 128
    num_beams: int = 4
    no_repeat_ngram_size: int = 2
    repetition_penalty: float = 1.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    bertscore_model: str = "bert-base-multilingual-cased"

class Evaluator:
    """Класс для выполнения валидации Seq2Seq модели."""

    TASK_MAPPING = {
        "title": "тақырып: ",
        "key_events": "оқиға: ",
        "location": "орын: ",
        "key_names": "есімдер: "
    }

    def __init__(self, model_path: str, config: EvalConfig):
        self.config = config
        self.model_path = model_path
        self._load_model()
        self._init_metrics()

    def _load_model(self) -> None:
        logger.info(f"Loading model from: {self.model_path}")
        try:
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(self.config.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _init_metrics(self) -> None:
        logger.info("Initializing metrics (BERTScore, ROUGE)...")
        self.bertscore = evaluate.load("bertscore")
        self.rouge = evaluate.load("rouge")

    @staticmethod
    def load_data(file_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_batch(self, texts: List[str], task_prefix: str) -> List[str]:
        """Генерация предсказаний для батча текстов."""
        input_texts = [f"{task_prefix}{text}" for text in texts]
        
        inputs = self.tokenizer(
            input_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.config.max_input_length
        ).to(self.config.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.config.max_target_length,
                num_beams=self.config.num_beams,
                early_stopping=True,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                repetition_penalty=self.config.repetition_penalty
            )
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def evaluate(self, data_path: str) -> None:
        """Основной цикл валидации."""
        logger.info(f"Loading data from: {data_path}")
        try:
            data = self.load_data(data_path)
        except FileNotFoundError as e:
            logger.error(e)
            return

        all_preds: List[str] = []
        all_refs: List[str] = []
        
        # Структура для хранения результатов по задачам
        task_results: Dict[str, Dict[str, List[str]]] = {
            k: {'preds': [], 'refs': []} for k in self.TASK_MAPPING.keys()
        }

        logger.info(f"Starting inference on {self.config.device}...")
        
        for i in tqdm(range(0, len(data), self.config.batch_size), desc="Processing batches"):
            batch_items = data[i : i + self.config.batch_size]
            batch_texts = [item.get('text', '') for item in batch_items]
            
            for task_key, prefix in self.TASK_MAPPING.items():
                current_refs = []
                valid_indices = []
                
                for idx, item in enumerate(batch_items):
                    target = item.get(task_key)
                    
                    # Business logic: handle empty locations
                    if task_key == 'location' and not target:
                        target = "Белгісіз"
                    
                    if target:
                        current_refs.append(str(target))
                        valid_indices.append(idx)

                if not valid_indices:
                    continue

                texts_to_process = [batch_texts[j] for j in valid_indices]
                preds = self.generate_batch(texts_to_process, prefix)
                
                # Aggregation
                all_preds.extend(preds)
                all_refs.extend(current_refs)
                task_results[task_key]['preds'].extend(preds)
                task_results[task_key]['refs'].extend(current_refs)

        self._compute_and_log_metrics(all_preds, all_refs, task_results)

    def _compute_and_log_metrics(self, preds: List[str], refs: List[str], task_results: Dict) -> None:
        logger.info("Computing metrics...")

        # 1. BERTScore
        try:
            bs_res = self.bertscore.compute(
                predictions=preds,
                references=refs,
                model_type=self.config.bertscore_model,
                lang="kz"
            )
            bs_f1_mean = np.mean(bs_res['f1'])
        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {e}")
            bs_f1_mean = 0.0

        # 2. ROUGE
        rouge_res = self.rouge.compute(
            predictions=preds, 
            references=refs,
            tokenizer=lambda x: x.split()
        )

        print("-" * 60)
        print("OVERALL RESULTS")
        print("-" * 60)
        print(f"BERTScore F1: {bs_f1_mean:.4f}")
        print(f"ROUGE-1:      {rouge_res['rouge1'] * 100:.2f}")
        print(f"ROUGE-2:      {rouge_res['rouge2'] * 100:.2f}")
        print(f"ROUGE-L:      {rouge_res['rougeL'] * 100:.2f}")
        print("-" * 60)
        
        print("\nPER-TASK BREAKDOWN (BERTScore F1):")
        for task, content in task_results.items():
            if not content['preds']:
                continue
            
            try:
                res = self.bertscore.compute(
                    predictions=content['preds'],
                    references=content['refs'],
                    model_type=self.config.bertscore_model,
                    lang="kz"
                )
                score = np.mean(res['f1'])
            except Exception:
                score = 0.0
            
            print(f"Task: {task:<15} Score: {score:.4f}")
            
            if content['preds']:
                idx = random.randint(0, len(content['preds']) - 1)
                print(f"  [Ref]:  {content['refs'][idx]}")
                print(f"  [Pred]: {content['preds'][idx]}")
            print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mT5 Validation Script")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--data_path", type=str, default="../data/dataset_kz_val.json", help="Path to validation dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    
    args = parser.parse_args()
    
    config = EvalConfig(batch_size=args.batch_size)
    evaluator = Evaluator(args.model_dir, config)
    evaluator.evaluate(args.data_path)