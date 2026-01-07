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
    val_path: str = "data/val.jsonl"
    log_dir: str = "./runs/experiment_fix_oversegmentation" # Новая папка логов
    save_dir: str = "./checkpoints/best_model_robust"

    # Model
    model_name: str = "kz-transformers/kaz-roberta-conversational" 
    # model_name: str = "google/embeddinggemma-300m"
    architecture: str = "cross_encoder" # "cross_encoder" or "bi_encoder"
    embedding_dim: int = 768
    max_len: int = 128
    freeze_base: bool = False
    use_embedding_prompt: bool = False
    embedding_task: str = "sentence similarity"
    
    # Training
    batch_size: int = 16
    epochs: int = 100 # Дообучаем 5 эпох
    lr: float = 2e-5 # Пониженный LR для дообучения
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # --- БАЛАНСИРОВКА И АУГМЕНТАЦИЯ (CRITICAL FIX) ---
    seed: int = 42
    
    # 1. Смещаем баланс в сторону Label 0 (продолжение темы)
    synthetic_split_prob: float = 0.1  # Только 30% примеров будут Label 1 (разрыв)
    
    # 2. Джиттеринг (сдвиг границ)
    jitter_prob: float = 0.4           # Меньше шансов оставить идеальную границу
    max_jitter_shift: int = 10
    extreme_jitter_prob: float = 0.2   # Шанс сильного сдвига (Label 0)
    
    # 3. Вероятности аугментаций
    aug_prob: float = 0.8              # Общий шанс применения аугментаций
    mix_prob: float = 0.2              # Mixing тем
    
    # 4. Ловушки (Trap Words) - самое важное для лечения "мысалы"
    false_anchor_prob: float = 0.7     # Часто вставлять "бірақ", "мысалы" в Label 0
    extreme_noise_prob: float = 0.2    # Шум (Label 1)
    
    # Hardware
    device: str = field(default_factory=_detect_device)
    num_workers: int = 4