"""Core initialization module.

This package contains early initialization code that must run before
other modules are imported. Import this package first in all entry points.
"""

from __future__ import annotations

# Configure tiktoken cache at package import time
from . import tiktoken_init as _tiktoken_init  # noqa: F401
