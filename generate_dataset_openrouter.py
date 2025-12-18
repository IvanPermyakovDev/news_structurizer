#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Any, Dict, List, Optional, Sequence


PROMPT_RU = """Ты — профессиональный редактор и диктор радионовостей, а также симулятор ошибок автоматического распознавания речи (ASR) для русского языка.

Твоя задача — сгенерировать датасет для обучения модели офлайн-сегментации радионовостей из ASR-текста на отдельные новости.

ОБЩИЕ ТРЕБОВАНИЯ
1. Весь текст должен быть:
   - без знаков препинания
   - без заглавных букв
   - в разговорном дикторском стиле
   - с возможными повторами слов и переформулировками
2. Тематики новостей:
   - политика
   - экономика
   - происшествия
   - погода
   - спорт
   - культура
   - международные события
3. Длина одной новости: от 30 до 120 слов
4. Переходы между новостями должны быть разнообразными:
   - явные
   - мягкие
   - ложные без смены темы
   - неожиданные
   - возврат к предыдущей теме

ASR ШУМ (ОБЯЗАТЕЛЬНО)
Добавляй реалистичные ошибки ASR:
- пропуски служебных слов
- склейку слов например вслучае вместо в случае
- замены близких по звучанию слов
- повторы слов
- неправильные окончания
- потерю редких имен и чисел
- иногда резкие обрывы фраз
Не используй одинаковые шаблоны ошибок и не применяй шум равномерно

СЛОЖНЫЕ СЛУЧАИ (КРИТИЧНО)
Обязательно включай:
- слова между тем также при этом кроме того которые иногда не означают новую новость
- смену имен и организаций без смены темы
- начало новой новости без вводных слов
- микс спорта и политики внутри одного блока
- смену диктора без смены темы
- паузу без смены темы
- смену темы без паузы

ФОРМАТ ВЫВОДА
Верни результат только в формате JSONL без пояснений

Каждая строка должна быть отдельным JSON объектом вида
{
  "context_left": "15–25 слов текста до границы",
  "context_right": "15–25 слов текста после границы",
  "label": 0 или 1
}

Где
label 1 означает начало новой новости
label 0 означает продолжение той же новости

БАЛАНС ДАННЫХ
Сделай 30–40 процентов примеров с label 1 и 60–70 процентов с label 0

КАЧЕСТВО
- не повторяй одни и те же переходы
- не используй фиксированные формулы
- делай тексты максимально разнообразными
- каждый пример должен выглядеть как реальный радио ASR

ОБЪЕМ
Сгенерируй N примеров где N задано пользователем

ПЕРЕД ВЫВОДОМ ВНУТРЕННЕ ПРОВЕРЬ
- что тексты выглядят как ASR
- что метка соответствует смыслу
- что присутствуют сложные и пограничные случаи
"""


PUNCT_RE = re.compile(r"[.,!?;:\"“”'’()\[\]{}<>«»—–…\-]")

LOGGER = logging.getLogger("openrouter_dataset")


@dataclass(frozen=True)
class Example:
    context_left: str
    context_right: str
    label: int

    def key(self) -> str:
        payload = f"{self.context_left}\n||\n{self.context_right}\n||\n{self.label}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "context_left": self.context_left,
                "context_right": self.context_right,
                "label": self.label,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


def _word_count(text: str) -> int:
    return len([w for w in text.strip().split() if w])


def _looks_like_asr(text: str) -> bool:
    if not text:
        return False
    if text != text.lower():
        return False
    if "\n" in text or "\r" in text or "\t" in text:
        return False
    if PUNCT_RE.search(text):
        return False
    return True


def _validate_example(obj: Any) -> Optional[Example]:
    if not isinstance(obj, dict):
        return None
    if not {"context_left", "context_right", "label"}.issubset(obj.keys()):
        return None

    left = obj.get("context_left")
    right = obj.get("context_right")
    label = obj.get("label")

    if not isinstance(left, str) or not isinstance(right, str):
        return None
    if not isinstance(label, int) or label not in (0, 1):
        return None

    left = " ".join(left.strip().split())
    right = " ".join(right.strip().split())

    if not (_looks_like_asr(left) and _looks_like_asr(right)):
        return None

    if not (15 <= _word_count(left) <= 25):
        return None
    if not (15 <= _word_count(right) <= 25):
        return None

    return Example(context_left=left, context_right=right, label=label)


def _extract_json_objects(text: str) -> List[Any]:
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()

    if not cleaned:
        return []

    if cleaned[0] == "[":
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []

    # Prefer JSONL (one object per line). Fall back to brace-balanced extraction.
    out: List[Any] = []
    for line in cleaned_lines:
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if out:
        return out

    buffer: List[str] = []
    depth = 0
    for line in cleaned_lines:
        for ch in line:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth = max(0, depth - 1)
        buffer.append(line)
        if depth == 0 and buffer:
            candidate = "".join(buffer).strip()
            buffer.clear()
            if not candidate:
                continue
            try:
                out.append(json.loads(candidate))
            except json.JSONDecodeError:
                continue
    return out


def _post_json(
    *,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"OpenRouter returned non-JSON payload: {raw[:500]}") from e


def openrouter_chat(
    *,
    api_key: str,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    base_url: str,
    http_referer: Optional[str] = None,
    x_title: Optional[str] = None,
    max_retries: int = 6,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if x_title:
        headers["X-Title"] = x_title

    payload = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        if attempt == 0:
            LOGGER.info("openrouter request: model=%s temperature=%.2f max_tokens=%d", model, temperature, max_tokens)
        try:
            data = _post_json(url=url, headers=headers, payload=payload, timeout_s=timeout_s)
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"OpenRouter response has no choices: {data}")
            message = (choices[0] or {}).get("message") or {}
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError(f"OpenRouter returned empty content: {data}")
            return content
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            status = getattr(e, "code", None)
            if status in (401, 403):
                raise RuntimeError(f"OpenRouter auth failed ({status}). Body: {body[:500]}")
            if status in (400,):
                raise RuntimeError(f"OpenRouter request rejected ({status}). Body: {body[:500]}")
            sleep_s = min(30.0, 1.5 * (2**attempt) + random.random())
            LOGGER.warning("openrouter http error status=%s retry_in=%.1fs body=%s", status, sleep_s, body[:200])
            time.sleep(sleep_s)
        except urllib.error.URLError:
            sleep_s = min(30.0, 1.5 * (2**attempt) + random.random())
            LOGGER.warning("openrouter url error retry_in=%.1fs", sleep_s)
            time.sleep(sleep_s)

    raise RuntimeError("OpenRouter request failed after retries.")


def _build_messages(n: int, bias: Optional[str]) -> List[Dict[str, str]]:
    user = (
        PROMPT_RU
        + f"\nСгенерируй ровно {n} примеров\n"
        + "ВЫВОД СТРОГО JSONL каждая строка один объект без кода блоков и без пустых строк\n"
        + 'Каждый JSON объект должен быть в одну строку и содержать только ключи "context_left" "context_right" "label"\n'
    )
    return [
        {
            "role": "system",
            "content": "Ты пишешь только данные в требуемом формате без объяснений и без лишнего текста.",
        },
        {"role": "user", "content": user},
    ]


def _pick_target_label1(n: int, available_1: int, available_0: int) -> int:
    min_1 = int((0.30 * n) + 0.00001)
    max_1 = int((0.40 * n) + 0.99999)
    desired = int(round(0.35 * n))

    feasible = [
        k
        for k in range(min_1, max_1 + 1)
        if k <= available_1 and (n - k) <= available_0
    ]
    if not feasible:
        return min(available_1, n)
    return min(feasible, key=lambda k: abs(k - desired))


def _append_examples(path: str, examples: Sequence[Example], *, flush: bool) -> None:
    if not examples:
        return
    with open(path, "a", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.to_jsonl())
            f.write("\n")
        if flush:
            f.flush()
            os.fsync(f.fileno())


def _load_existing_examples(path: str) -> List[Example]:
    if not os.path.exists(path):
        return []
    examples: List[Example] = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            ex = _validate_example(obj)
            if not ex:
                bad += 1
                continue
            examples.append(ex)
    if bad:
        LOGGER.warning("resume: skipped %d invalid lines from %s", bad, path)
    return examples


def generate_dataset(
    *,
    api_key: str,
    model: str,
    batch_size: int,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    http_referer: Optional[str],
    x_title: Optional[str],
    max_rounds: int,
    workers: int = 1,
    existing: Sequence[Example] = (),
    output_path: Optional[str] = None,
    flush_progress: bool = True,
) -> List[Example]:
    if workers <= 0:
        raise ValueError("--workers must be positive")
    if max_rounds < 0:
        raise ValueError("--max-rounds must be >= 0 (0 means infinite)")

    lock = threading.Lock()
    stop_event = threading.Event()
    seen: set[str] = set()
    label1_count = 0
    label0_count = 0
    requests_made = 0
    appended_lines = 0
    newly_added: List[Example] = []

    for ex in existing:
        k = ex.key()
        if k in seen:
            continue
        seen.add(k)
        if ex.label == 1:
            label1_count += 1
        else:
            label0_count += 1

    LOGGER.info(
        "generation start: batch_size=%d max_rounds=%s workers=%d existing=%d (label1=%d label0=%d)",
        batch_size,
        "infinite" if max_rounds == 0 else str(max_rounds),
        workers,
        label1_count + label0_count,
        label1_count,
        label0_count,
    )
    if output_path and not os.path.exists(output_path):
        open(output_path, "w", encoding="utf-8").close()
        LOGGER.info("output created: %s", output_path)

    def should_stop() -> bool:
        if max_rounds == 0:
            return False
        return requests_made >= max_rounds

    def _worker(worker_idx: int) -> int:
        nonlocal label1_count, label0_count, requests_made, appended_lines
        local_appended = 0

        while not stop_event.is_set():
            with lock:
                if should_stop():
                    stop_event.set()
                    break
                requests_made += 1
                request_id = requests_made
                total = label1_count + label0_count

            LOGGER.info(
                "req %d: worker=%d have=%d (label1=%d label0=%d) request_n=%d",
                request_id,
                worker_idx,
                total,
                label1_count,
                label0_count,
                batch_size,
            )

            messages = _build_messages(batch_size, None)
            t0 = time.monotonic()
            content = openrouter_chat(
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                base_url=base_url,
                http_referer=http_referer,
                x_title=x_title,
            )
            dt = time.monotonic() - t0

            parsed = _extract_json_objects(content)
            valid = 0
            added1 = 0
            added0 = 0
            round_new: List[Example] = []
            for item in parsed:
                ex = _validate_example(item)
                if not ex:
                    continue
                valid += 1
                round_new.append(ex)

            with lock:
                accepted: List[Example] = []
                for ex in round_new:
                    k = ex.key()
                    if k in seen:
                        continue
                    seen.add(k)
                    accepted.append(ex)
                    newly_added.append(ex)
                    if ex.label == 1:
                        label1_count += 1
                        added1 += 1
                    else:
                        label0_count += 1
                        added0 += 1

                total_after = label1_count + label0_count

            LOGGER.info(
                "req %d: worker=%d api_time=%.1fs parsed=%d valid=%d accepted=%d (accepted1=%d accepted0=%d) totals=%d (label1=%d label0=%d)",
                request_id,
                worker_idx,
                dt,
                len(parsed),
                valid,
                len(accepted),
                added1,
                added0,
                total_after,
                label1_count,
                label0_count,
            )

            if output_path and accepted and flush_progress:
                with lock:
                    _append_examples(output_path, accepted, flush=True)
                    appended_lines += len(accepted)
                    try:
                        size = os.path.getsize(output_path)
                    except OSError:
                        size = -1
                    LOGGER.info(
                        "file append: +%d lines to %s (appended_total=%d size_bytes=%d)",
                        len(accepted),
                        output_path,
                        appended_lines,
                        size,
                    )
                local_appended += len(accepted)

        return local_appended

    # If write-progress is disabled, we still generate but delay writes until the end.
    if workers == 1:
        _worker(1)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="or") as pool:
            futures = [pool.submit(_worker, i + 1) for i in range(workers)]
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc is not None:
                    stop_event.set()
                    raise exc

    if output_path and newly_added and not flush_progress:
        with lock:
            _append_examples(output_path, newly_added, flush=True)
            appended_lines += len(newly_added)
            try:
                size = os.path.getsize(output_path)
            except OSError:
                size = -1
            LOGGER.info(
                "file append (final): +%d lines to %s (appended_total=%d size_bytes=%d)",
                len(newly_added),
                output_path,
                appended_lines,
                size,
            )

    # For append-mode, we intentionally return only the newly added examples.
    return newly_added


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a JSONL segmentation dataset via OpenRouter.")
    parser.add_argument(
        "--output",
        type=str,
        default="dataset_openrouter.json",
        help="Output path (JSONL content, one object per line).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        help="OpenRouter model id (or set OPENROUTER_MODEL).",
    )
    parser.add_argument("--batch-size", type=int, default=25, help="Examples requested per API call.")
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=1,
        help="Maximum OpenRouter requests per run (total, across all workers). Use 0 for infinite until Ctrl+C.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker threads (each sends requests).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=120000, help="Max tokens per response.")
    parser.add_argument("--timeout", type=float, default=400.0, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        help="OpenRouter base URL.",
    )
    parser.add_argument("--http-referer", type=str, default=os.environ.get("OPENROUTER_HTTP_REFERER"))
    parser.add_argument("--x-title", type=str, default=os.environ.get("OPENROUTER_X_TITLE"))
    parser.add_argument("--seed", type=int, default=42, help="Seed for retry jitter/determinism helpers.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity (stderr).",
    )
    parser.add_argument(
        "--write-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continuously update --output during generation (after each API round).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call API; read response body from stdin and validate/print stats.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
        stream=sys.stderr,
    )
    random.seed(args.seed)

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.max_rounds < 0:
        raise SystemExit("--max-rounds must be >= 0 (0 means infinite)")

    if args.dry_run:
        content = sys.stdin.read()
        objs = _extract_json_objects(content)
        examples = [ex for ex in (_validate_example(o) for o in objs) if ex]
        n1 = sum(1 for e in examples if e.label == 1)
        n0 = sum(1 for e in examples if e.label == 0)
        print(f"valid={len(examples)} label1={n1} label0={n0}", file=sys.stderr)
        for ex in examples:
            print(ex.to_jsonl())
        return 0

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY env var.")

    out_path = args.output
    existing = _load_existing_examples(out_path)
    existing_total = len(existing)
    existing_label1 = sum(1 for e in existing if e.label == 1)
    existing_label0 = existing_total - existing_label1
    LOGGER.info(
        "resume: output=%s existing_valid=%d (label1=%d label0=%d)",
        out_path,
        existing_total,
        existing_label1,
        existing_label0,
    )
    new_examples = generate_dataset(
        api_key=api_key,
        model=args.model,
        batch_size=args.batch_size,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout,
        http_referer=args.http_referer,
        x_title=args.x_title,
        max_rounds=args.max_rounds,
        workers=args.workers,
        existing=existing,
        output_path=out_path,
        flush_progress=bool(args.write_progress),
    )

    final = _load_existing_examples(out_path)
    label1 = sum(1 for e in final if e.label == 1)
    label0 = len(final) - label1
    print(
        f"appended {len(new_examples)} new examples to {out_path} (file_total_valid={len(final)}, label1={label1}, label0={label0})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
