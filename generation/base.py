"""Base protocol/interface for LLM clients.

This module defines the BaseLLMClient protocol that all LLM providers must implement.
Using Python's Protocol for structural subtyping allows existing classes to be
compatible without explicit inheritance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


@dataclass
class Message:
    """A chat message."""

    role: str
    content: str


@dataclass
class LLMResponse:
    """Response from any LLM provider."""

    content: str
    model: Optional[str] = None
    usage: Optional[dict] = None
    raw_response: Optional[dict] = field(default=None, repr=False)


@runtime_checkable
class BaseLLMClient(Protocol):
    """Protocol defining the interface for LLM clients.

    All LLM provider implementations must support these methods.
    Uses structural subtyping - any class with matching methods is compatible.
    """

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a response from a prompt.

        Args:
            prompt: The user prompt/question.
            system_prompt: Optional system instructions.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0-1.0).

        Returns:
            Generated text content.
        """
        ...

    def chat(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Send a chat request with multiple messages.

        Args:
            messages: List of chat messages with role and content.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0-1.0).

        Returns:
            LLM response with content and metadata.
        """
        ...
