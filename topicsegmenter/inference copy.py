import torch
import re
import os
import inspect
from dataclasses import dataclass
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils import normalize_text

@dataclass(frozen=True)
class InferenceConfig:
    architecture: str = "cross_encoder" # "cross_encoder" or "bi_encoder"
    max_len: int = 128
    use_embedding_prompt: bool = True
    embedding_task: str = "sentence similarity"
    left_words: int = 60
    right_words: int = 40
    min_segment_words: int = 50
    confirm_thr: float = 0.5
    refine_radius: int = 12
    use_fast_tokenizer: bool = False
    prominence_radius: int = 40
    min_prominence: float = 0.08
    max_splits: int = 20
    fallback_delta: float = 0.1
    fallback_min_thr: float = 0.3
    fallback_rel_margin: float = 0.05
    overlap_window: int = 20
    scan_step: int = 1

class NewsSegmenter:
    def __init__(self, model_path: str, base_model_name: str = "cointegrated/rubert-tiny2", cfg: Optional[InferenceConfig] = None, verbose: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.verbose = verbose
        if self.verbose:
            print(f"Loading weights from: {model_path}")
        self.cfg = cfg or InferenceConfig()
        
        self.tokenizer = self._load_tokenizer(model_path=model_path, base_model_name=base_model_name)
        
        # Грузим веса модели
        if self.cfg.architecture == "bi_encoder":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
            
        self.model.to(self.device)
        self.model.eval()
        self._accepts_token_type_ids = self._model_accepts_token_type_ids(self.model)

    def _load_tokenizer(self, model_path: str, base_model_name: str):
        # Prefer local tokenizer; use_fast=False avoids a known regex-warning path in some environments.
        try:
            tok = AutoTokenizer.from_pretrained(model_path, use_fast=self.cfg.use_fast_tokenizer)
        except Exception:
            # Last resort: load tokenizer from the base model (may require network / cache)
            tok = AutoTokenizer.from_pretrained(base_model_name)
        
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token if tok.eos_token else tok.sep_token
        return tok

    @staticmethod
    def _model_accepts_token_type_ids(model) -> bool:
        try:
            return "token_type_ids" in inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            return False

    def _normalize(self, text: str) -> str:
        return normalize_text(text)

    def _apply_embedding_prompt(self, texts: List[str]) -> List[str]:
        if not self.cfg.use_embedding_prompt:
            return texts
        prefix = f"task: {self.cfg.embedding_task} | query: "
        return [prefix + t for t in texts]

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def _predict_batch(self, pairs: List[tuple]) -> List[float]:
        """Пакетное предсказание вероятностей."""
        if not pairs: return []
        
        batch_size = 32
        all_probs = []
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            lefts = [self._normalize(p[0]) for p in batch]
            rights = [self._normalize(p[1]) for p in batch]
            if self.cfg.architecture == "bi_encoder":
                lefts = self._apply_embedding_prompt(lefts)
                rights = self._apply_embedding_prompt(rights)
            
            if self.cfg.architecture == "bi_encoder":
                # Bi-Encoder: Embed separately and calc similarity
                inputs_l = self.tokenizer(lefts, add_special_tokens=True, max_length=self.cfg.max_len, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
                inputs_r = self.tokenizer(rights, add_special_tokens=True, max_length=self.cfg.max_len, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    out_l = self.model(input_ids=inputs_l['input_ids'], attention_mask=inputs_l['attention_mask'])
                    out_r = self.model(input_ids=inputs_r['input_ids'], attention_mask=inputs_r['attention_mask'])
                    
                    emb_l = torch.nn.functional.normalize(
                        self._mean_pool(out_l.last_hidden_state, inputs_l["attention_mask"]),
                        p=2,
                        dim=1,
                    )
                    emb_r = torch.nn.functional.normalize(
                        self._mean_pool(out_r.last_hidden_state, inputs_r["attention_mask"]),
                        p=2,
                        dim=1,
                    )
                    
                    sim = torch.nn.functional.cosine_similarity(emb_l, emb_r)
                    # Convert similarity to "split probability"
                    # If sim=1.0 (same), prob=0.0. If sim=-1.0 (different), prob=1.0.
                    probs = (1.0 - sim) / 2.0
                    all_probs.extend(probs.cpu().tolist())
            else:
                # Cross-Encoder: Paired input
                inputs = self.tokenizer(
                    lefts, rights,
                    add_special_tokens=True,
                    max_length=self.cfg.max_len,
                    padding="max_length",
                    truncation="only_first",
                    return_tensors="pt"
                )

                if not self._accepts_token_type_ids and "token_type_ids" in inputs:
                    inputs.pop("token_type_ids")

                inputs = inputs.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                    all_probs.extend(probs[:, 1].cpu().tolist())
                
        return all_probs

    def _ctx_pair_at(self, words: List[str], split_idx: int) -> tuple[str, str]:
        left = " ".join(words[max(0, split_idx - self.cfg.left_words):split_idx])
        right = " ".join(words[split_idx:min(len(words), split_idx + self.cfg.right_words)])
        return left, right

    def _refine_split(self, words: List[str], center_idx: int) -> tuple[int, float]:
        """Уточняет split_idx, ищет максимум вероятности в окрестности."""
        start = max(1, center_idx - self.cfg.refine_radius)
        end = min(len(words) - 1, center_idx + self.cfg.refine_radius)
        
        candidates = []
        indices = []
        for i in range(start, end + 1):
            candidates.append(self._ctx_pair_at(words, i))
            indices.append(i)
            
        probs = self._predict_batch(candidates)
        
        best_prob = -1.0
        best_idx = center_idx
        for idx, prob in zip(indices, probs):
            if prob > best_prob:
                best_prob = prob
                best_idx = idx
        return best_idx, best_prob

    def _select_splits_nms(self, probs: List[float], scan_indices: List[int], total_words: int, thr: float, scores: Optional[List[float]] = None) -> List[int]:
        if not probs:
            return []

        prefix = [0.0]
        for p in probs:
            prefix.append(prefix[-1] + float(p))

        def window_mean(center_i: int) -> float:
            r = max(1, self.cfg.prominence_radius // max(1, self.cfg.scan_step))
            lo = max(0, center_i - r)
            hi = min(len(probs) - 1, center_i + r)
            count = (hi - lo + 1) - 1
            if count <= 0:
                return 0.0
            total = (prefix[hi + 1] - prefix[lo]) - float(probs[center_i])
            return total / count

        if scores is None:
            scores = probs

        candidates = []
        for i, (p, idx, score) in enumerate(zip(probs, scan_indices, scores)):
            if p < thr:
                continue
            if (p - window_mean(i)) < self.cfg.min_prominence:
                continue
            candidates.append((score, p, idx))

        candidates.sort(reverse=True, key=lambda x: x[0])

        chosen: List[int] = []
        for score, prob, idx in candidates:
            if idx < self.cfg.min_segment_words or (total_words - idx) < self.cfg.min_segment_words:
                continue
            if any(abs(idx - c) < self.cfg.min_segment_words for c in chosen):
                continue
            chosen.append(idx)
            if len(chosen) >= self.cfg.max_splits:
                break

        chosen.sort()
        return chosen

    def segment_text(self, text: str) -> List[str]:
        words = text.split()
        if self.verbose:
            print(f"Processing {len(words)} words...")
        
        # 1. Сбор всех точек (STEP=1)
        scan_indices = list(range(self.cfg.min_segment_words, len(words) - self.cfg.min_segment_words, max(1, self.cfg.scan_step)))
        if not scan_indices:
            return [text]

        candidates = []
        for i in scan_indices:
            candidates.append(self._ctx_pair_at(words, i))
            
        # 2. Массовое предсказание (Batch Inference)
        if self.verbose:
            print(f"Running full scan on {len(candidates)} points...")
        probs = self._predict_batch(candidates)
        prob_by_idx = dict(zip(scan_indices, probs))

        overlaps = []
        for idx in scan_indices:
            w = self.cfg.overlap_window
            left_set = set(words[max(0, idx - w):idx])
            right_set = set(words[idx:min(len(words), idx + w)])
            union = left_set | right_set
            overlap = (len(left_set & right_set) / len(union)) if union else 0.0
            overlaps.append(overlap)
        scores = [p * (1.0 - ov) for p, ov in zip(probs, overlaps)]

        if probs:
            best_i = max(range(len(probs)), key=probs.__getitem__)
            max_prob = probs[best_i]
            best_idx = scan_indices[best_i]
            if self.verbose:
                print(f"Max prob={max_prob:.4f} at idx={best_idx} ('{words[best_idx-1]}')")
            best_score_i = max(range(len(scores)), key=scores.__getitem__)
            if self.verbose:
                print(f"Max score={scores[best_score_i]:.4f} at idx={scan_indices[best_score_i]} ('{words[scan_indices[best_score_i]-1]}')")
        else:
            max_prob = 0.0
        
        # 3. Выбор разрывов (NMS по вероятностям)
        chosen = self._select_splits_nms(probs, scan_indices, len(words), thr=self.cfg.confirm_thr, scores=scores)

        # Fallback: если вообще ничего не нашли, но модель где-то уверена — понижаем порог
        if not chosen:
            fallback_thr = max(self.cfg.fallback_min_thr, self.cfg.confirm_thr - self.cfg.fallback_delta)
            if max_prob >= fallback_thr:
                chosen = self._select_splits_nms(probs, scan_indices, len(words), thr=fallback_thr, scores=scores)
                if chosen:
                    rel_thr = max_prob - self.cfg.fallback_rel_margin
                    chosen = [idx for idx in chosen if prob_by_idx.get(idx, 0.0) >= rel_thr]
            if not chosen and probs:
                topk = sorted(zip(probs, scan_indices), reverse=True)[:5]
                if self.verbose:
                    print("Top-5 probs:", ", ".join([f"{p:.3f}@{idx}" for p, idx in topk]))

        refined = []
        for idx in chosen:
            best_idx, best_prob = self._refine_split(words, idx)
            refined.append((best_idx, best_prob))

        # Дедуп после refine
        refined.sort()
        split_indices = [0]
        for idx, prob in refined:
            if idx - split_indices[-1] < self.cfg.min_segment_words:
                continue
            if self.verbose:
                print(f"    >>> CUT at {idx} ('{words[idx-1]}'), prob={prob:.4f}")
            split_indices.append(idx)

        split_indices.append(len(words))
        
        # Сборка
        segments = []
        for k in range(len(split_indices) - 1):
            segments.append(" ".join(words[split_indices[k]:split_indices[k+1]]))
            
        return segments

if __name__ == "__main__":
    # Чтобы убрать ошибку Mistral навсегда, удалите старые конфиги
    checkpoint_dir = "./checkpoints/best_model"
    # for junk in ["tokenizer_config.json", "special_tokens_map.json", "tokenizer.json"]:
    #     p = os.path.join(checkpoint_dir, junk)
    #     if os.path.exists(p):
    #         os.remove(p)


    segmenter = NewsSegmenter(checkpoint_dir)

    long_text = (
        "сегодня в северном техническом университете представили прототип новой батареи для портативной электроники и дронов заявленная плотность энергии выше чем у предыдущих образцов и как бы это привлекло много студентов и инженеров мы работаем из лаборатории номер двенадцать здесь шум вытяжки и поэтому речь может пропадать руководитель проекта аллина жукова извините алина говорит что команда перешла на гибридный катод на основе никеля и марганца и добавила добавила что электролит использует солевой раствор с пониженой летучестью при комнатной температуре стенд показывает быстрое восстановлен е более семидесяти процентов зарядки за пятнадцать минут но это предварительные данные они подчёркивают что нужно ещё десять циклов тестов на деградацию компонетов в корридоре мы встретили аспиранта роман руднева он объясняет что в прототипе есть тонкая мембрана идущая из композита и что её производство может быть дорогим пока учёные ищут локального подрядчика в пресс релизе который раздавали на входе было написано латиницей battery proto v3 и там же ошибка в фамилии авторов жуково без а теперь о безопасности представители лаборатории подчёркнули что ячейки не содержат кобальта и это снижает экологический след но всё равно требуется утилизац по правилам эээ ведущий спрашивает а как насчёт совместимости с существующими устройствами ответ такой форм фактор будет повторять популярные стандарты 18650 и pouch варианты а также планируют модульный блок для квадрокоптеров организаторы показали небольшой дрон который поднялся в воздух на три минуты это не рекорд но демонстрация стабильности напряжения без провалов и перегрева температурный профиль снимали тепловизором график выглядит ровно между двадцать восемь и тридцать два градуса в соседнем кабинете обсуждали технологию покрытия токосъёмных пластин там инженер петрович простите нет точного имени он говорит что порошок поставляют из ростех хим но возможно скоро переключатся на местного дилера из колпино тут в репортаже всплыла заминка камера ушла в расфокус и мы на секунду потеряли картинку зато слышали как студенты хлопают и кричат ура ура вот такие эмоции презентация шла почти час на вопросы об интеллектуальной собственности жукова ответила что патент уже подан и ожидается публикация заявки зимой а пока партнёрам дают доступ под nda давайте добавим комментарий независимого специалиста из инж центра волна он говорит сказал что показатель плотности энергии выглядит перспективно но важно проверить поведение на морозе и в циклах быстрая зарядка иногда вызывает литиевое платирование цитирую аккуратно не спешите бежать в магазин конец цитаты в целом университет объявил о пилотной линии к следующей весне если финансирование подтвердят ректор отметил что переговоры с индустриальными партнёрами из питерского тех парка идут в рабочем режиме и есть шанс получить грант городского уровня мы остаёмся на площадке и будем следить за испытаниями итог дня таков прототип есть цифры есть впереди валидац тестов и много работы но настроение бодрое студенты уже сделали фото и выкладывают live истории в сети "
        "на финансовых рынках сегодня заметно оживление хотя волатильност осталась высокой аналитики центра орбита говорять что инвесторы реагируют на квартальные отчеты и на ожидания по ставкам крупнейших центробанков глобально индекс мби растёт на один и две десятых процента по предварительным подсчётам а технологический сектор лидирует по сохраненню спроса эээ вот наш обозреватель никита раев он на связи из студии он говорит что крупные игроки перекладываются из защитных активов в акции роста и это видно по оборотам в бумагах полупроводников и облачных сервисов при этом по словам по словам экспертов любые намёки на замедление прибыли могут быстро изменить настроение у нас есть комментарий с площадки международной фондовой биржи трейдер представился как сергей х не расслышал фамилию извините он сказал что роботы включились раньше чем ожидали и утром был всплеск алгоритмных покупок а затем консолидация к полудню курс некоторых валют распилился извините распилился это он так сказал там был фоновый шум и просто не разобрал я думаю речь про распил по диапазону в коридоре один тридцать один тридцать два к доллару кстати на сырьевом рынке нефть показала умеренный рост примерно на ноль точка восемь процента металллы разнонаправлено золото плюс маленький процент серебро около нуля а медь снижается что может быть связано с ожиданием данных по строй сектору здесь мы делаем ремарку что в прошлом месяце участники рынка переоценили шансы на скорое смягчение политики и сейчас риторика стала осторожнеи многие говорят дождёмся протокола комитета и потом уже решим добавили что значительная часть ралли вызвана закрытием коротких позиций отдельные бумаги растут на новостях о бай бек программах вот например крупный разработчик софта объявил о расширении обратного выкупа и рынок отреагировал всплеском объёмов но наш эксперт напоминает что байбек не равен росту выручки так что не стоит путать эти вещи эээ важный момент по облигациям доходности на длинном конце слегка снизились что даёт сектору недвижимости глоток воздуха однако риски остаются потому что рефинансирование торговых центров под вопросом мы слышим от управляющих что они закладывают стресс сценарии в модели оценки кэша и так по инфляции предварительный консенсус говорит про умеренное замедление в год к году но базовая составляюща может удивить и если она окажется упорной то реакция рынка будет резкой как говорят трейдеры флип сценарий здесь ещё что заметили участники азиатской сессии утром там главное движение прошло в технологических гигантах и на слухах о новых контрмерах регулятора по листингам некоторые издания упоминали площадку с неправильным названием нью йорк борд видимо опечатка да чтобы не вводить в заблуждение укажем это как ошибку редакции пока итоги дня выглядят зелеными но помните дисклеймер инвестиционные решения несут риски и это не рекомендац я мы продолжаем следить за динамикой если капитал потечёт обратно в облигации значит участники испугались очередных сюрпризов пока же у мониторов бодро мигают объёмы и алгоритмы балан сируют книгу заявок на этом всё к текущему часу услышымся позже"
    )

    results = segmenter.segment_text(long_text)
    
    print("\n" + "="*30)
    for i, seg in enumerate(results):
        print(f"NEWS {i+1}: {seg}")
    print("="*30)
