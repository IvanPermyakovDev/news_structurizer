import torch
import re
import os
import inspect
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils import normalize_text

class NewsSegmenter:
    def __init__(self, model_path: str, base_model_name: str = "kz-transformers/kaz-roberta-conversationald"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        print(f"Loading weights from: {model_path}")
        
        # ЛЕЧИМ ОШИБКУ MISTRAL:
        # Грузим чистый токенизатор из интернета, игнорируя конфиги в папке
        # self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Грузим веса модели
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        # Расширяем embeddings если токенизатор больше модели
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.model.eval()
        self._accepts_token_type_ids = self._model_accepts_token_type_ids(self.model)

    @staticmethod
    def _model_accepts_token_type_ids(model) -> bool:
        try:
            return "token_type_ids" in inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            return False

    def _normalize(self, text: str) -> str:
        # Должно СТРОГО совпадать с тем, что в ASRAugmentor.apply
        return normalize_text(text)

    def _predict_batch(self, pairs: List[tuple]) -> List[float]:
        """Пакетное предсказание вероятностей."""
        if not pairs: return []
        
        batch_size = 32
        all_probs = []
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            lefts = [self._normalize(p[0]) for p in batch]
            rights = [self._normalize(p[1]) for p in batch]
            
            inputs = self.tokenizer(
                lefts, rights,
                add_special_tokens=True, max_length=128, # Совпадает с новым Config.max_len
                padding="max_length", truncation=True, return_tensors="pt"
            )

            if not self._accepts_token_type_ids and "token_type_ids" in inputs:
                inputs.pop("token_type_ids")

            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                all_probs.extend(probs[:, 1].cpu().tolist())
                
        return all_probs

    def _find_local_peak(self, words: List[str], center_idx: int, radius: int = 10) -> (int, float):
        """Сканирует область вокруг center_idx, ищет максимум вероятности."""
        start = max(1, center_idx - radius)
        end = min(len(words) - 1, center_idx + radius)
        
        candidates = []
        indices = []
        for i in range(start, end + 1):
            ctx_left = " ".join(words[max(0, i-50):i])
            ctx_right = " ".join(words[i:min(len(words), i+50)])
            candidates.append((ctx_left, ctx_right))
            indices.append(i)
            
        probs = self._predict_batch(candidates)
        
        best_prob = -1.0
        best_idx = center_idx
        for idx, prob in zip(indices, probs):
            if prob > best_prob:
                best_prob = prob
                best_idx = idx
        return best_idx, best_prob

    def segment_text(self, text: str) -> List[str]:
        words = text.split()
        print(f"Processing {len(words)} words...")
        
        # Настройки
        MIN_LEN = 10  # Увеличиваем минимальную длину новости
        CONFIRM_THR = 0.8  # Немного поднимаем порог
        
        # 1. Сбор всех точек (STEP=1)
        scan_indices = list(range(MIN_LEN, len(words) - MIN_LEN))
        if not scan_indices:
            return [text]

        candidates = []
        for i in scan_indices:
            # Используем 50 слов, чтобы влезло в max_len=128
            ctx_left = " ".join(words[max(0, i-50):i])
            ctx_right = " ".join(words[i:min(len(words), i+50)])
            candidates.append((ctx_left, ctx_right))
            
        # 2. Массовое предсказание (Batch Inference)
        print(f"Running full scan on {len(candidates)} points...")
        probs = self._predict_batch(candidates)
        
        # 3. Поиск пиков
        split_indices = [0]
        i = 0
        while i < len(probs):
            prob = probs[i]
            idx = scan_indices[i]
            
            # Проверяем, является ли это локальным пиком
            is_peak = True
            if i > 0 and probs[i-1] >= prob: is_peak = False
            if i < len(probs) - 1 and probs[i+1] > prob: is_peak = False
            
            if is_peak and prob > CONFIRM_THR:
                # Проверка на минимальную длину от предыдущего разрыва
                if idx - split_indices[-1] >= MIN_LEN:
                    print(f"    >>> CUT at {idx} ('{words[idx-1]}'), prob={prob:.4f}")
                    split_indices.append(idx)
                    
                    # Пропускаем MIN_LEN слов
                    while i < len(scan_indices) and scan_indices[i] < idx + MIN_LEN:
                        i += 1
                    continue
            
            i += 1

        split_indices.append(len(words))
        
        # Сборка
        segments = []
        for k in range(len(split_indices) - 1):
            segments.append(" ".join(words[split_indices[k]:split_indices[k+1]]))
            
        return segments

if __name__ == "__main__":
    # Чтобы убрать ошибку Mistral навсегда, удалите старые конфиги
    checkpoint_dir = "./checkpoints/best_model_1"
    # for junk in ["tokenizer_config.json", "special_tokens_map.json", "tokenizer.json"]:
    #     p = os.path.join(checkpoint_dir, junk)
    #     if os.path.exists(p):
    #         os.remove(p)


    segmenter = NewsSegmenter(checkpoint_dir)

    long_text = """
  бүгін астанада ауа райы суық болады түнде температура минус он бес градусқа 
  дейін түседі ал күндіз минус тоғыз градус болады метеорологтар жел те қатты 
  соғатынын ескертеді батыс өңірлерде қар жауады деп хабарлайды тұрғындарға сақ 
  болу керек дейді мамандар 

  ал енді спорт жаңалықтарына көшейік алматыдағы футбол ойынында қайрат 
  командасы актобені үш бір есебімен ұтты шешуші голды екінші таймда жасаған 
  болатын бұл жеңіс қайратты турнир кестесінде екінші орынға көтерді командада 
  қуанышты көңіл күй билейді келесі ойын келер аптада өтеді

  қазір экономикалық жаңалықтарды талқылайық ұлттық банк базалық мөлшерлемені он
   екі пайызға көтерді бұл инфляцияны тежеу үшін қажет деп түсіндіреді 
  сарапшылар валюта нарығында тенге бір доллар үшін төрт жүз жетпіс теңгеден 
  сатылып тұр аналитиктер алдағы айларда тұрақтылық болады деп болжайды

  мәдениет саласынан хабарлар шымкентте халықаралық театр фестивалі басталды он 
  екі елден қонақтар келді әр түрлі қойылымдармен таныстырады фестиваль екі апта
   бойы жалғасады билеттер интернет арқылы сатылады қала әкімі бұл үлкен мәдени 
  оқиға деп атады тамашаларға барлық қалаушыларды шақырады
  """

    results = segmenter.segment_text(long_text)
    
    print("\n" + "="*30)
    for i, seg in enumerate(results):
        print(f"NEWS {i+1}: {seg}")
    print("="*30)
