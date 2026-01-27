"""Initialize tiktoken cache directory before any tiktoken imports.

This module MUST be imported before tiktoken is used anywhere in the codebase.
It configures the TIKTOKEN_CACHE_DIR environment variable to point to the
local cache directory, enabling offline use in corporate environments with
TLS-intercepting proxies.

Usage:
    # At the top of every entry point (before other imports):
    import core  # noqa: F401

    # Or explicitly:
    import core.tiktoken_init  # noqa: F401
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Expected cache file for cl100k_base encoding (SHA1 hash of URL)
CL100K_BASE_HASH = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
MIN_CACHE_FILE_SIZE = 100_000  # ~100KB minimum for valid encoding file


def _find_project_root() -> Optional[Path]:
    """Find project root by checking for marker files.

    Tries multiple strategies to robustly find the project root:
    1. Walk up from this file's location
    2. Check for common project markers (pyproject.toml, requirements.txt, models/)

    Returns:
        Path to project root, or None if not found.
    """
    # Start from this file's directory (core/)
    current = Path(__file__).resolve().parent

    # Walk up at most 5 levels looking for project markers
    for _ in range(5):
        # Check for project root markers
        if (current / "pyproject.toml").exists():
            return current
        if (current / "requirements.txt").exists():
            return current
        if (current / "models").is_dir():
            return current
        if (current / "CLAUDE.md").exists():
            return current

        parent = current.parent
        if parent == current:
            # Reached filesystem root
            break
        current = parent

    return None


def _find_cache_directory() -> Optional[Path]:
    """Find tiktoken cache directory.

    Returns:
        Path to cache directory if found and valid, None otherwise.
    """
    # Strategy 1: Explicit environment variable (user override)
    env_cache = os.environ.get("TIKTOKEN_CACHE_DIR")
    if env_cache:
        cache_path = Path(env_cache)
        if cache_path.is_dir():
            return cache_path.resolve()

    # Strategy 2: models/tiktoken_cache relative to project root
    project_root = _find_project_root()
    if project_root:
        cache_dir = project_root / "models" / "tiktoken_cache"
        if cache_dir.is_dir():
            return cache_dir.resolve()

    return None


def _validate_cache_directory(cache_dir: Path) -> bool:
    """Validate that cache directory has expected encoding files.

    Args:
        cache_dir: Path to the tiktoken cache directory.

    Returns:
        True if cache is valid and has at least cl100k_base encoding.
    """
    if not cache_dir.is_dir():
        return False

    # Check for cl100k_base encoding (the default used by this project)
    cl100k_file = cache_dir / CL100K_BASE_HASH
    if cl100k_file.exists():
        size = cl100k_file.stat().st_size
        if size >= MIN_CACHE_FILE_SIZE:
            return True
        logger.warning(
            "Tiktoken cache file %s is too small (%d bytes), may be corrupted",
            cl100k_file,
            size,
        )
        return False

    # Fallback: check for any hash-named files (40-char hex filenames)
    hash_files = [f for f in cache_dir.iterdir() if len(f.name) == 40 and f.is_file()]
    if hash_files:
        # Validate at least one file has reasonable size
        for f in hash_files:
            if f.stat().st_size >= MIN_CACHE_FILE_SIZE:
                return True

    return False


def configure_tiktoken_cache() -> bool:
    """Configure TIKTOKEN_CACHE_DIR if local cache exists.

    This function is idempotent - it can be called multiple times safely.
    It will only set the environment variable once.

    Returns:
        True if cache was found and configured, False otherwise.
    """
    # Already configured by a previous call
    existing = os.environ.get("TIKTOKEN_CACHE_DIR")
    if existing:
        logger.debug("TIKTOKEN_CACHE_DIR already set: %s", existing)
        return True

    cache_dir = _find_cache_directory()
    if cache_dir and _validate_cache_directory(cache_dir):
        # Use absolute path with forward slashes (cross-platform)
        cache_path = str(cache_dir.resolve())
        os.environ["TIKTOKEN_CACHE_DIR"] = cache_path
        logger.debug("Tiktoken cache configured: %s", cache_path)
        return True

    # Cache not found - log helpful message
    project_root = _find_project_root()
    if project_root:
        expected_path = project_root / "models" / "tiktoken_cache"
        logger.debug(
            "Tiktoken cache not found at %s. "
            "Run: python -m scripts.download_tiktoken_cache --output-dir %s",
            expected_path,
            expected_path,
        )
    else:
        logger.debug(
            "Could not find project root. "
            "Set TIKTOKEN_CACHE_DIR environment variable manually."
        )

    return False


# Run configuration at module import time
_cache_configured = configure_tiktoken_cache()
