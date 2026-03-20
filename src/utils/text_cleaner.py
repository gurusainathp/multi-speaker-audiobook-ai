"""
src/utils/text_cleaner.py
────────────────────────────────────────────────────────────────
Central text normalization for the audiobook pipeline.

All parsers return raw extracted text. This module is the single
place where cleaning happens — no parser should do its own
normalization beyond basic extraction.

Usage:
    from src.utils.text_cleaner import clean_text

    raw = "Some   messy\n\n\n\ntext with- \nhyphen breaks."
    clean = clean_text(raw)
"""

import re
import unicodedata
import logging

logger = logging.getLogger(__name__)


# ── Public API ───────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Run the full normalization pipeline on raw extracted text.

    Steps applied in order:
      1. Unicode normalization (NFC)
      2. Fix smart / curly quotes → straight ASCII equivalents
      3. Fix common unicode punctuation (dashes, ellipsis, etc.)
      4. Rejoin hyphenated line-break splits  (e.g. "amaz-\ning" → "amazing")
      5. Remove standalone page numbers
      6. Remove repeated header/footer lines
      7. Collapse whitespace (tabs, multiple spaces → single space)
      8. Collapse excessive blank lines (3+ → 2)
      9. Strip leading/trailing whitespace

    Args:
        text: Raw text string as returned by any parser.

    Returns:
        Cleaned, normalized plain-text string.
    """
    if not text or not text.strip():
        return ""

    text = _normalize_unicode(text)
    text = _fix_quotes(text)
    text = _fix_unicode_punctuation(text)
    text = _rejoin_hyphen_breaks(text)
    text = _remove_page_numbers(text)
    text = _remove_repeated_lines(text)
    text = _collapse_whitespace(text)
    text = _collapse_blank_lines(text)

    return text.strip()


def clean_page(text: str) -> str:
    """
    Lighter cleaning for a single extracted page (used by PDF parser).
    Skips operations that need full-document context (repeated lines, etc.).
    """
    if not text:
        return ""

    text = _normalize_unicode(text)
    text = _fix_quotes(text)
    text = _fix_unicode_punctuation(text)
    text = _rejoin_hyphen_breaks(text)
    text = _collapse_whitespace(text)

    # Remove form-feed characters
    text = text.replace("\f", "\n")

    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


# ── Normalization steps ───────────────────────────────────────────────────────

def _normalize_unicode(text: str) -> str:
    """
    Normalize to NFC (composed form).
    Fixes encoding artifacts like split accented characters.
    """
    return unicodedata.normalize("NFC", text)


def _fix_quotes(text: str) -> str:
    """
    Replace curly / smart quotes with their ASCII equivalents.
    This prevents TTS from misreading fancy quote characters.

    Mapping:
      " "  →  "
      ' '  →  '   (also covers apostrophes)
      „    →  "
      «»   →  "
    """
    replacements = {
        "\u201c": '"',   # left double quotation mark
        "\u201d": '"',   # right double quotation mark
        "\u201e": '"',   # double low-9 quotation mark
        "\u00ab": '"',   # left-pointing double angle quotation mark
        "\u00bb": '"',   # right-pointing double angle quotation mark
        "\u2018": "'",   # left single quotation mark
        "\u2019": "'",   # right single quotation mark / apostrophe
        "\u201a": "'",   # single low-9 quotation mark
        "\u2039": "'",   # single left-pointing angle quotation mark
        "\u203a": "'",   # single right-pointing angle quotation mark
        "\u0060": "'",   # grave accent used as quote
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def _fix_unicode_punctuation(text: str) -> str:
    """
    Normalize unicode punctuation to ASCII equivalents.

      — (em dash)   →  --
      – (en dash)   →  -
      … (ellipsis)  →  ...
      ‐ (hyphen)    →  -
    """
    replacements = {
        "\u2014": "--",   # em dash
        "\u2013": "-",    # en dash
        "\u2012": "-",    # figure dash
        "\u2011": "-",    # non-breaking hyphen
        "\u2010": "-",    # hyphen
        "\u2026": "...",  # horizontal ellipsis
        "\u00ad": "",     # soft hyphen (invisible, just remove)
        "\u200b": "",     # zero-width space
        "\u00a0": " ",    # non-breaking space → regular space
        "\ufeff": "",     # BOM (byte order mark)
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def _rejoin_hyphen_breaks(text: str) -> str:
    """
    Rejoin words broken across lines with a hyphen.

    PDFs often split long words at line boundaries:
      "amaz-\ning" → "amazing"
      "some-\nthing" → "something"

    Only merges when the hyphen is at the very end of a line,
    followed by a newline and a lowercase letter (to avoid
    merging legitimate hyphenated phrases like "well-\nknown"
    that appear mid-sentence).
    """
    # Pattern: word chars, hyphen, newline, lowercase continuation
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)
    return text


def _remove_page_numbers(text: str) -> str:
    """
    Remove standalone page number lines.

    Matches lines that contain only:
      - A bare integer:          "42"
      - "Page N" variants:       "Page 42", "page 42"
      - "- N -" style:           "- 42 -"
      - Roman numerals (up to ~50 pages of front matter)
    """
    # Bare integers on their own line
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)

    # "Page N" or "PAGE N"
    text = re.sub(r"(?mi)^\s*page\s+\d+\s*$", "", text)

    # "- N -" or "– N –"
    text = re.sub(r"(?m)^\s*[-–]\s*\d+\s*[-–]\s*$", "", text)

    # Roman numerals standing alone (i, ii, iii ... xlviii)
    text = re.sub(
        r"(?mi)^\s*M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\s*$",
        "",
        text,
    )

    return text


def _remove_repeated_lines(text: str) -> str:
    """
    Remove lines that appear suspiciously often throughout the document
    (typically running headers or footers like the book title or author name).

    Threshold: a line appearing in more than 20% of paragraphs (and at
    least 5 times absolute) is considered a repeated header/footer.

    Note: This is conservative — it won't remove lines that appear just
    twice or three times, which could be legitimate repeated dialogue.
    """
    lines = text.split("\n")
    total = len(lines)

    if total < 20:
        # Not enough content to safely detect repeated lines
        return text

    # Count non-empty line occurrences
    from collections import Counter
    line_counts = Counter(line.strip() for line in lines if line.strip())

    threshold_count = max(5, int(total * 0.20))
    repeated = {
        line for line, count in line_counts.items()
        if count >= threshold_count and len(line) > 3
    }

    if repeated:
        logger.debug(
            f"[text_cleaner] Removing {len(repeated)} repeated header/footer line(s): "
            + str(list(repeated)[:3])
        )

    cleaned = [line for line in lines if line.strip() not in repeated]
    return "\n".join(cleaned)


def _collapse_whitespace(text: str) -> str:
    """
    Collapse runs of spaces or tabs on each line into a single space.
    Does not touch newlines.
    """
    # Replace any sequence of spaces/tabs with a single space, per line
    lines = text.split("\n")
    return "\n".join(re.sub(r"[ \t]+", " ", line) for line in lines)


def _collapse_blank_lines(text: str) -> str:
    """
    Collapse 3 or more consecutive blank lines into exactly 2.
    Preserves intentional paragraph breaks (2 blank lines = chapter break).
    """
    return re.sub(r"\n{3,}", "\n\n", text)