"""Unit tests for tiktoken cache initialization."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from core.tiktoken_init import (
    CL100K_BASE_HASH,
    MIN_CACHE_FILE_SIZE,
    _find_cache_directory,
    _find_project_root,
    _validate_cache_directory,
    configure_tiktoken_cache,
)


class TestFindProjectRoot:
    """Tests for project root detection."""

    def test_find_project_root_from_core(self):
        """Test finding project root from core module location."""
        root = _find_project_root()
        assert root is not None
        # Should find the RAG project root
        assert (root / "requirements.txt").exists() or (root / "pyproject.toml").exists()

    def test_find_project_root_has_models(self):
        """Test that found root contains models directory."""
        root = _find_project_root()
        assert root is not None
        assert (root / "models").is_dir()

    def test_find_project_root_returns_absolute_path(self):
        """Test that project root is an absolute path."""
        root = _find_project_root()
        assert root is not None
        assert root.is_absolute()


class TestFindCacheDirectory:
    """Tests for cache directory detection."""

    def test_find_cache_directory_exists(self):
        """Test finding cache directory when it exists."""
        cache_dir = _find_cache_directory()
        # If the cache exists in the repo, it should be found
        if cache_dir is not None:
            assert cache_dir.is_dir()
            assert cache_dir.is_absolute()

    def test_find_cache_directory_respects_env_var(self, tmp_path: Path):
        """Test that explicit env var takes precedence."""
        # Create a temp cache directory
        temp_cache = tmp_path / "tiktoken_cache"
        temp_cache.mkdir()

        with patch.dict(os.environ, {"TIKTOKEN_CACHE_DIR": str(temp_cache)}):
            cache_dir = _find_cache_directory()
            assert cache_dir == temp_cache.resolve()

    def test_find_cache_directory_ignores_nonexistent_env_path(self, tmp_path: Path):
        """Test that non-existent env var path is handled."""
        nonexistent = tmp_path / "does_not_exist"

        with patch.dict(os.environ, {"TIKTOKEN_CACHE_DIR": str(nonexistent)}):
            # Should fall back to project-relative search
            cache_dir = _find_cache_directory()
            # Either finds project cache or returns None
            if cache_dir is not None:
                assert cache_dir.is_dir()


class TestValidateCacheDirectory:
    """Tests for cache directory validation."""

    def test_validate_nonexistent_directory(self, tmp_path: Path):
        """Test validation of non-existent directory."""
        nonexistent = tmp_path / "does_not_exist"
        assert _validate_cache_directory(nonexistent) is False

    def test_validate_empty_directory(self, tmp_path: Path):
        """Test validation of empty directory."""
        empty_dir = tmp_path / "empty_cache"
        empty_dir.mkdir()
        assert _validate_cache_directory(empty_dir) is False

    def test_validate_directory_with_small_file(self, tmp_path: Path):
        """Test validation rejects too-small files."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create a file that's too small
        small_file = cache_dir / CL100K_BASE_HASH
        small_file.write_text("too small")
        assert _validate_cache_directory(cache_dir) is False

    def test_validate_directory_with_valid_file(self, tmp_path: Path):
        """Test validation accepts valid-sized files."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create a file with sufficient size
        valid_file = cache_dir / CL100K_BASE_HASH
        valid_file.write_bytes(b"x" * (MIN_CACHE_FILE_SIZE + 1))
        assert _validate_cache_directory(cache_dir) is True

    def test_validate_directory_with_other_hash_file(self, tmp_path: Path):
        """Test validation accepts other 40-char hash files."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create a different hash file (not cl100k_base)
        other_hash = "a" * 40  # 40-char hex string
        other_file = cache_dir / other_hash
        other_file.write_bytes(b"x" * (MIN_CACHE_FILE_SIZE + 1))
        assert _validate_cache_directory(cache_dir) is True


class TestConfigureTiktokenCache:
    """Tests for the main configuration function."""

    def test_configure_is_idempotent(self):
        """Test that configure can be called multiple times safely."""
        # First call
        result1 = configure_tiktoken_cache()
        env1 = os.environ.get("TIKTOKEN_CACHE_DIR")

        # Second call should return same result
        result2 = configure_tiktoken_cache()
        env2 = os.environ.get("TIKTOKEN_CACHE_DIR")

        assert result1 == result2
        assert env1 == env2

    def test_configure_respects_existing_env_var(self, tmp_path: Path):
        """Test that existing env var is not overwritten."""
        custom_path = str(tmp_path / "custom_cache")

        # Clear any existing config first
        original = os.environ.pop("TIKTOKEN_CACHE_DIR", None)
        try:
            os.environ["TIKTOKEN_CACHE_DIR"] = custom_path
            result = configure_tiktoken_cache()

            # Should return True (already configured)
            assert result is True
            # Should not change the value
            assert os.environ.get("TIKTOKEN_CACHE_DIR") == custom_path
        finally:
            # Restore original
            if original:
                os.environ["TIKTOKEN_CACHE_DIR"] = original
            else:
                os.environ.pop("TIKTOKEN_CACHE_DIR", None)

    def test_configure_sets_absolute_path(self):
        """Test that configured path is absolute."""
        # Clear env var to force fresh detection
        original = os.environ.pop("TIKTOKEN_CACHE_DIR", None)
        try:
            configure_tiktoken_cache()
            cache_path = os.environ.get("TIKTOKEN_CACHE_DIR")

            if cache_path:
                assert Path(cache_path).is_absolute()
        finally:
            # Restore original
            if original:
                os.environ["TIKTOKEN_CACHE_DIR"] = original


class TestIntegration:
    """Integration tests for tiktoken cache initialization."""

    def test_import_core_configures_cache(self):
        """Test that importing core module configures tiktoken cache."""
        # The cache should already be configured since we imported core.tiktoken_init
        cache_path = os.environ.get("TIKTOKEN_CACHE_DIR")

        # In the test environment with the actual repo, cache should be configured
        if cache_path:
            assert Path(cache_path).is_absolute()
            assert Path(cache_path).is_dir()

    def test_tiktoken_uses_local_cache(self):
        """Test that tiktoken can load encoding from local cache."""
        import tiktoken

        # This should not require network access if cache is configured
        enc = tiktoken.get_encoding("cl100k_base")
        assert enc.name == "cl100k_base"

        # Verify basic functionality
        tokens = enc.encode("Hello world")
        assert len(tokens) >= 2

    def test_cache_file_hash_matches_expected(self):
        """Test that cache file name matches expected SHA1 hash."""
        import hashlib

        url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        expected_hash = hashlib.sha1(url.encode()).hexdigest()

        assert expected_hash == CL100K_BASE_HASH
