from dataclasses import dataclass, field
import torch


def _detect_device() -> str:
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
    except Exception:
        return "cpu"
    return "cpu"


@dataclass
class Config:
    # Paths
    train_path: str = "data/train.jsonl"
    val_path: str = "data/val.jsonl" # Если есть, иначе сплит внутри
    log_dir: str = "./runs/experiment_v1"
    save_dir: str = "./checkpoints/best_model"

    # Model
    model_name: str = "DeepPavlov/rubert-base-cased" 
    # model_name: str = "google/embeddinggemma-300m"
    architecture: str = "cross_encoder" # "cross_encoder" or "bi_encoder"
    embedding_dim: int = 768
    max_len: int = 128
    freeze_base: bool = False
    use_embedding_prompt: bool = False
    embedding_task: str = "sentence similarity"  # e.g. "search result", "classification"
    
    # Training
    batch_size: int = 16
    epochs: int = 100
    lr: float = 5e-5
    warmup_steps: int = 200
    weight_decay: float = 0.01
    
    # Augmentation & Jittering
    seed: int = 42
    jitter_prob: float = 0.5  # Вероятность того, что Label 1 останется 1 (иначе сдвиг)
    max_jitter_shift: int = 10 # Максимальный сдвиг границы (в словах)
    aug_prob: float = 0.5     # Вероятность применения текстовых аугментаций
    mix_prob: float = 0.3     # Вероятность смешивания двух разных контекстов (Synthetic Mixing)
    
    # Hardware
    device: str = field(default_factory=_detect_device)
    num_workers: int = 4
