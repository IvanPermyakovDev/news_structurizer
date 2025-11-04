from __future__ import annotations

import copy
import json
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback


MODEL_NAME = "DeepPavlov/rubert-base-cased"
TOPIC_LABELS = [
    "политика",
    "спорт",
    "экономика",
    "технологии и наука",
    "культура и искусство",
    "экология и климат",
    "mixed",
]
SCALE_LABELS = ["local", "global"]


def set_seed(seed: int) -> None:
    """Apply the seed across random, numpy and torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch might be unavailable in CPU-only environments, which is fine
        pass


def build_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    return tokenizer


def load_and_split_dataset(
    json_path: str | Path,
    label_column: str,
    allowed_labels: Sequence[str],
    seed: int,
    test_size: float = 0.10,
) -> DatasetDict:
    """Load the JSON dataset and create train/validation splits."""
    path = Path(json_path)
    records = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(records)

    required_columns = {"text", label_column}
    missing = required_columns - set(frame.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    unexpected = sorted(set(frame[label_column].unique()) - set(allowed_labels))
    if unexpected:
        unexpected_labels = ", ".join(unexpected)
        raise ValueError(
            f"Dataset contains unexpected labels for '{label_column}': {unexpected_labels}"
        )

    train_frame, test_frame = train_test_split(
        frame,
        test_size=test_size,
        stratify=frame[label_column],
        random_state=seed,
    )
    validation_size = test_size / (1 - test_size)
    train_frame, val_frame = train_test_split(
        train_frame,
        test_size=validation_size,
        stratify=train_frame[label_column],
        random_state=seed,
    )

    val_frame = pd.concat([val_frame, test_frame], ignore_index=True)

    return DatasetDict(
        train=Dataset.from_pandas(train_frame.reset_index(drop=True)),
        validation=Dataset.from_pandas(val_frame.reset_index(drop=True)),
    )


def _remove_unused_columns(dataset: Dataset, keep_columns: Iterable[str]) -> Dataset:
    keep = set(keep_columns)
    to_remove = [column for column in dataset.column_names if column not in keep]
    return dataset.remove_columns(to_remove)


def prepare_tokenized_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    label_column: str,
    labels: Sequence[str],
    max_length: int,
) -> tuple[DatasetDict, dict[str, int], dict[int, str]]:
    """Tokenize the dataset and attach label ids."""
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    def encode_labels(example: dict) -> dict:
        example["labels"] = label2id[example[label_column]]
        return example

    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    encoded = dataset.map(encode_labels)
    tokenized = encoded.map(tokenize_batch, batched=True)

    keep_columns = ("input_ids", "attention_mask", "labels")
    cleaned = DatasetDict(
        {split: _remove_unused_columns(ds_split, keep_columns) for split, ds_split in tokenized.items()}
    )
    return cleaned, label2id, id2label


def compute_metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


class BestModelTracker(TrainerCallback):
    """Track best checkpoints and persist checkpoints on each evaluation."""

    def __init__(
        self,
        metric_name: str,
        greater_is_better: bool,
        tokenizer: AutoTokenizer,
        best_dir: Path,
        last_dir: Path,
    ) -> None:
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.tokenizer = tokenizer
        self.best_dir = Path(best_dir)
        self.last_dir = Path(last_dir)
        self.best_metric: float | None = None
        self.best_state: dict | None = None
        self.saving_enabled = True
        self.best_dir.mkdir(parents=True, exist_ok=True)
        self.last_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(self, model, directory: Path) -> None:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # type: ignore[override]
        if not metrics or self.metric_name not in metrics:
            return

        metric_value = metrics[self.metric_name]
        if self.best_metric is None:
            should_update = True
        elif self.greater_is_better:
            should_update = metric_value > self.best_metric
        else:
            should_update = metric_value < self.best_metric

        if should_update:
            self.best_metric = metric_value
            model = kwargs.get("model")
            if model is not None:
                self.best_state = copy.deepcopy(model.state_dict())
                if self.saving_enabled:
                    self._save_checkpoint(model, self.best_dir)

        model = kwargs.get("model")
        if model is not None and self.saving_enabled:
            self._save_checkpoint(model, self.last_dir)


class LogTableCallback(TrainerCallback):
    """Use Trainer.log_metrics to display evaluation metrics in table form."""

    def __init__(self, trainer: Trainer, best_tracker: BestModelTracker) -> None:
        self._trainer = trainer
        self._best_tracker = best_tracker

    @staticmethod
    def _infer_split(metrics: dict[str, float]) -> str:
        for key in metrics:
            if "_" in key:
                return key.split("_", 1)[0]
        return "eval"

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # type: ignore[override]
        if not metrics or not self._trainer.is_world_process_zero():
            return
        split = self._infer_split(metrics)
        if split not in {"eval", "validation"}:
            return

        table_metrics = dict(metrics)
        best_metric = self._best_tracker.best_metric
        if best_metric is not None:
            table_metrics[f"{split}_best_macro_f1"] = best_metric

        self._trainer.log_metrics(split, table_metrics)


def train_model(
    dataset: DatasetDict,
    model_name: str,
    tokenizer: AutoTokenizer,
    id2label: dict[int, str],
    label2id: dict[str, int],
    output_dir: Path,
    logging_dir: Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    use_fp16: bool,
    log_to_tensorboard: bool,
) -> tuple[dict[str, float], Path, Path]:
    """Fine-tune the sequence classification model, returning metrics and checkpoint paths."""
    output_dir = Path(output_dir)
    logging_dir = Path(logging_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)
    best_dir = output_dir / "best"
    last_dir = output_dir / "last"
    best_dir.mkdir(parents=True, exist_ok=True)
    last_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="no",
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=50,
        seed=seed,
        fp16=use_fp16,
        logging_dir=str(logging_dir),
        report_to="tensorboard" if log_to_tensorboard else "none",
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if use_fp16 else None,
    )

    best_tracker = BestModelTracker(
        metric_name="eval_macro_f1",
        greater_is_better=True,
        tokenizer=tokenizer,
        best_dir=best_dir,
        last_dir=last_dir,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[best_tracker],
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(LogTableCallback(trainer, best_tracker))

    train_result = trainer.train()
    if train_result.metrics:
        trainer.log_metrics("train", train_result.metrics)

    last_state = copy.deepcopy(model.state_dict())

    best_tracker.saving_enabled = False
    if best_tracker.best_state is not None:
        model.load_state_dict(best_tracker.best_state)
    metrics = trainer.evaluate(dataset["validation"], metric_key_prefix="validation")
    metrics["validation_best_macro_f1"] = best_tracker.best_metric
    trainer.log_metrics("validation", metrics)

    model.load_state_dict(last_state)

    return metrics, best_dir, last_dir


@contextmanager
def tensorboard_server(log_dir: Path, port: int, enabled: bool) -> Iterator[bool]:
    """Launch TensorBoard as a background process and stop it on exit."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not enabled:
        yield False
        return

    import importlib.util

    if importlib.util.find_spec("tensorboard") is None:
        print("[TensorBoard] Module not installed. Skip auto-launch.")
        yield False
        return
    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(log_dir),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]

    process: subprocess.Popen | None = None
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        print("[TensorBoard] Executable not found. Skip auto-launch.")
        yield False
        return
    except OSError as exc:
        print(f"[TensorBoard] Failed to start ({exc}).")
        yield False
        return

    time.sleep(1.0)
    if process.poll() is not None:
        stdout_data, stderr_data = process.communicate()
        output = (stdout_data or b"") + (b"\n" if stdout_data and stderr_data else b"") + (stderr_data or b"")
        if output:
            preview = output.decode("utf-8", errors="replace").strip()
            preview = "\n".join(preview.splitlines()[:10])
            print(f"[TensorBoard] Process exited prematurely:\n{preview}")
        else:
            print("[TensorBoard] Process exited prematurely with no output.")
        yield False
        return

    print(f"[TensorBoard] Running on http://localhost:{port} (logdir: {log_dir})")
    try:
        yield True
    finally:
        if process and process.poll() is None:
            process.terminate()
