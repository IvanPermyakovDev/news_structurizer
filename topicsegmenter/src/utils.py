import random
import os
import numpy as np
import torch

import re

def normalize_text(text: str) -> str:
    # Keep only letters (Cyrillic + Kazakh extensions, and Latin) and spaces, lowercase everything
    text = text.lower()
    # NOTE: Kazakh Cyrillic letters: ә і ң ғ ү ұ қ ө һ
    text = re.sub(r'[^a-zа-яёәіңғүұқөһ\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Для детерминизма (может замедлить обучение)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
