import re


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zа-яё\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

