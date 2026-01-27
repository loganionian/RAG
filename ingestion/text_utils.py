from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional

import tiktoken

MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
MULTI_SPACE_PATTERN = re.compile(r"[ \t]{2,}")
NON_ASCII_QUOTES = {
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u2013": "-",
    "\u2014": "-",
}

# Sentence boundary pattern - simple version
# Matches: . ! ? followed by space and uppercase letter
SENTENCE_END_PATTERN = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])'
)

# Common abbreviations to avoid splitting on
ABBREVIATIONS = frozenset([
    'mr', 'mrs', 'ms', 'dr', 'prof', 'jr', 'sr', 'inc', 'ltd', 'corp',
    'vs', 'etc', 'al', 'fig', 'no', 'vol', 'pp', 'ed', 'rev'
])


@lru_cache(maxsize=4)
def _get_tokenizer(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Get a cached tiktoken encoding."""
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Count tokens using tiktoken for accurate token counting.

    Args:
        text: The text to count tokens for.
        encoding: The tiktoken encoding to use (default: cl100k_base for GPT-4).

    Returns:
        Number of tokens in the text.
    """
    if not text:
        return 0
    tokenizer = _get_tokenizer(encoding)
    return len(tokenizer.encode(text))


def normalize_text(text: str) -> str:
    """Clean up boilerplate whitespace and punctuation."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for bad, replacement in NON_ASCII_QUOTES.items():
        normalized = normalized.replace(bad, replacement)
    normalized = MULTI_NEWLINE_PATTERN.sub("\n\n", normalized)
    normalized = MULTI_SPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def estimate_tokens(text: str) -> int:
    """Fast approximate token count using regex (for non-critical paths).

    For accurate token counting, use count_tokens() instead.
    """
    return max(1, len(re.findall(r"\w+|\S", text)))


def split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs on double newlines."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs or [text.strip()]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences respecting common boundaries.

    Handles:
    - Standard sentence endings (. ! ?)
    - Abbreviations (Mr., Dr., etc.)
    - Quoted text
    - Decimal numbers

    Args:
        text: The text to split into sentences.

    Returns:
        List of sentences. Returns [text] if no sentence boundaries found.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # Simple split on sentence boundaries
    sentences = SENTENCE_END_PATTERN.split(text)

    # Clean up and filter empty sentences
    result = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            result.append(sentence)

    # If no splits occurred, return the original text
    if not result:
        return [text]

    return result


def split_into_units(text: str, max_tokens: int, encoding: str = "cl100k_base") -> List[str]:
    """Split text into units that fit within max_tokens.

    First tries sentence splitting, then falls back to word splitting
    for very long sentences.

    Args:
        text: The text to split.
        max_tokens: Maximum tokens per unit.
        encoding: Tiktoken encoding to use.

    Returns:
        List of text units, each within max_tokens.
    """
    if not text:
        return []

    # If text fits, return as-is
    if count_tokens(text, encoding) <= max_tokens:
        return [text]

    # Try sentence splitting first
    sentences = split_sentences(text)
    if len(sentences) > 1:
        result = []
        for sentence in sentences:
            if count_tokens(sentence, encoding) <= max_tokens:
                result.append(sentence)
            else:
                # Sentence too long, split by words
                result.extend(_split_by_words(sentence, max_tokens, encoding))
        return result

    # Single long sentence - split by words
    return _split_by_words(text, max_tokens, encoding)


def _split_by_words(text: str, max_tokens: int, encoding: str = "cl100k_base") -> List[str]:
    """Split text by words to fit within max_tokens."""
    words = text.split()
    if not words:
        return []

    result = []
    current_chunk: List[str] = []
    current_tokens = 0

    for word in words:
        word_tokens = count_tokens(word, encoding)

        # If single word exceeds max, add it anyway (can't split further)
        if word_tokens > max_tokens:
            if current_chunk:
                result.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            result.append(word)
            continue

        # Check if adding this word exceeds limit
        if current_tokens + word_tokens + 1 > max_tokens:  # +1 for space
            if current_chunk:
                result.append(" ".join(current_chunk))
            current_chunk = [word]
            current_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_tokens += word_tokens + (1 if current_chunk else 0)

    if current_chunk:
        result.append(" ".join(current_chunk))

    return result
