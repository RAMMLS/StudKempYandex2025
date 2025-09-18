# Препроцессинг
import re
import unicodedata

def strip_html_tags(text: str) -> str:
    clean = re.sub(r"<.*?>", "", text)  # убираем теги
    return clean


def remove_invisible_chars(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

def clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def preprocess_text(text: str) -> str:
    text = strip_html_tags(text)
    text = remove_invisible_chars(text)
    text = normalize_unicode(text)
    text = clean_spaces(text)
    return text