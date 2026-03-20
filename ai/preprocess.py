"""
Text preprocessing untuk deteksi komentar promosi judi online (Bahasa Indonesia).

Tujuan utama:
- Menormalkan unicode/font obfuscation (NFKC + penghilangan karakter format/invisible)
- Menangani emoji sebagai sinyal (dipetakan ke label)
- Menghapus noise simbol/ASCII art tanpa menghilangkan sinyal penting
- Mengatasi obfuscation dengan separator (mis. S L O T, s.l.o.t, g-a-c-o-r)
- Leetspeak normalization (angka/simbol -> huruf) secara selektif
- Normalisasi huruf berulang dan whitespace

Catatan desain:
- Preprocessing ini sengaja tidak melakukan stemming/slang dictionary agresif.
- Output mempertahankan token <...> untuk fitur yang stabil pada model transformer.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Tuple


# -----------------------------
# Emoji label mapping (selective)
# -----------------------------
_EMOJI_GROUPS: Dict[str, Tuple[str, ...]] = {
    "<MONEY>": ("💰", "🤑", "💸", "🏦", "💵", "💴", "💶", "💷", "🪙"),
    "<HYPE>": ("🔥", "🚀", "⚡", "💥", "✨", "⭐", "🌟"),
    "<OK>": ("✅", "✔", "☑"),
    "<WARN>": ("⚠", "🚨", "❗", "‼"),
    "<HEART>": ("❤️", "❤", "💖", "💗", "💓", "💞", "💘"),
    "<LAUGH>": ("😂", "🤣", "😹"),
    "<HAND>": ("👉", "👈", "👇", "👆", "🤝", "🙏"),
}

_EMOJI_TO_LABEL: Dict[str, str] = {
    emoji: label for label, emojis in _EMOJI_GROUPS.items() for emoji in emojis
}

_RE_ANY_LETTER = re.compile(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF\u0400-\u04FF\u0600-\u06FF\u0900-\u097F\u4E00-\u9FFF]")
_RE_HAS_ALNUM = re.compile(r"[A-Za-z0-9]")
_RE_INVISIBLE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]")
_RE_COMBINING = re.compile(r"[\u0300-\u036F]")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_NON_WORD_NO_TAG = re.compile(r"[^\w\s<>]")

_RE_TOKEN_TAG = re.compile(r"<[A-Z_]+>")
_RE_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

_RE_ASCII_ART_BLOCKS = re.compile(r"[▀▄█▌▐░▒▓■□◆◇●○◼◻◾◽╬═║╔╗╚╝╦╩╠╣╭╮╰╯┼─│]+")
_RE_LONG_SYMBOL_RUN = re.compile(r"([^\w\s<>])\1{2,}")

_RE_OBFUSCATED_SPACED = re.compile(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b")
_RE_OBFUSCATED_DOT = re.compile(r"\b(?:[A-Za-z]\.){2,}[A-Za-z]\b")
_RE_OBFUSCATED_MIXSEP = re.compile(r"\b(?:[A-Za-z][\.\-_/\\\|\s]){2,}[A-Za-z]\b")

_RE_TIMESTAMP_ONLY = re.compile(
    r"^\s*(?:\d{1,2}:)?\d{1,2}:\d{2}\s*$"
)

_RE_REPEATED_CHARS = re.compile(r"([a-z])\1{2,}", re.IGNORECASE)

_LEET_MAP = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "6": "g",
        "7": "t",
        "8": "b",
        "9": "g",
        "@": "a",
        "$": "s",
    }
)


def unicode_nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def remove_invisible_chars(text: str) -> str:
    text = _RE_INVISIBLE.sub("", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return text


def fix_font_obfuscation(text: str) -> str:
    text = text.translate({0x00A0: 0x0020})
    text = _RE_COMBINING.sub("", text)
    return text


def emoji_to_labels(text: str) -> str:
    for emoji, label in _EMOJI_TO_LABEL.items():
        if emoji in text:
            text = text.replace(emoji, f" {label} ")
    return text

def is_timestamp_only(text: str) -> bool:
    if text is None:
        return False
    return bool(_RE_TIMESTAMP_ONLY.match(str(text)))

def remove_ascii_art_and_symbol_noise(text: str) -> str:
    text = _RE_ASCII_ART_BLOCKS.sub(" ", text)
    text = _RE_LONG_SYMBOL_RUN.sub(" ", text)
    return text


def _merge_obfuscated_word(match_text: str) -> str:
    merged = re.sub(r"[\s\.\-_/\\\|]+", "", match_text)
    return merged


def fix_separator_obfuscation(text: str) -> str:
    def merge_if_short_segments(m: re.Match) -> str:
        raw = m.group(0)
        segments = re.split(r"[\s\.\-_/\\\|]+", raw)
        segments = [s for s in segments if s]
        if not segments:
            return raw
        if all(len(s) <= 2 for s in segments) and len(segments) >= 3:
            return _merge_obfuscated_word(raw)
        return raw

    text = _RE_OBFUSCATED_MIXSEP.sub(merge_if_short_segments, text)
    return text


def normalize_leet_selective(token: str) -> str:
    digit_count = sum(c.isdigit() for c in token)
    alpha_count = sum(c.isalpha() for c in token)

    if digit_count >= 2:
        return token

    if len(token) > 5:
        return token

    if alpha_count == 0:
        return token

    return token.translate(_LEET_MAP)


def normalize_leetspeak(text: str) -> str:
    def repl_word(m: re.Match) -> str:
        w = m.group(0)
        return normalize_leet_selective(w)

    return re.sub(r"\b[\w@$]+\b", repl_word, text)


def normalize_repeated_chars(text: str) -> str:
    return _RE_REPEATED_CHARS.sub(r"\1\1", text)


def remove_meaningless_text(text: str) -> str:
    if not _RE_ANY_LETTER.search(text):
        return ""
    return text


def lowercase_text(text: str) -> str:
    return text.lower()


def cleanup_whitespace(text: str) -> str:
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


def remove_symbol_noise_keep_tags(text: str) -> str:
    return _RE_NON_WORD_NO_TAG.sub(" ", text)


def preprocess_comment(text: str) -> str:
    """
    Main preprocessing pipeline untuk satu komentar.
    Urutan mengikuti requirement:
    1) NFKC
    2) remove invisible
    3) remove ascii art / symbol noise
    4) emoji -> label
    5) remove meaningless text (tanpa huruf)
    6) fix separator / obfuscation
    7) normalize angka/simbol -> huruf (leetspeak)
    8) normalize repeated chars
    9) lowercase
    10) rapikan whitespace
    """
    if text is None:
        return ""

    text = str(text)

    if _RE_TIMESTAMP_ONLY.match(text):
        return ""
    text = unicode_nfkc(text)
    text = fix_font_obfuscation(text)
    text = remove_invisible_chars(text)

    text = emoji_to_labels(text)

    text = remove_ascii_art_and_symbol_noise(text)
    text = remove_symbol_noise_keep_tags(text)

    text = cleanup_whitespace(text)
    text = remove_meaningless_text(text)
    if not text:
        return ""

    text = fix_separator_obfuscation(text)
    text = normalize_leetspeak(text)
    text = normalize_repeated_chars(text)

    text = lowercase_text(text)
    text = cleanup_whitespace(text)

    return text


def preprocess_many(texts: Iterable[str]) -> List[str]:
    return [preprocess_comment(t) for t in texts]