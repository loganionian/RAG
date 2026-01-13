"""Anthropic (Claude) LLM client implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from ..api_client import LLMClientError
from ..base import LLMResponse, Message


@dataclass
class AnthropicConfig:
    """Configuration for Anthropic API client."""

    api_key: str
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 60

    @classmethod
    def from_env(cls) -> "AnthropicConfig":
        """Create config from environment variables."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        return cls(
            api_key=api_key,
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        )


class AnthropicClient:
    """Anthropic API client implementing BaseLLMClient protocol."""

    def __init__(self, config: Optional[AnthropicConfig] = None):
        """Initialize the Anthropic client.

        Args:
            config: Anthropic configuration. If None, loads from environment.
        """
        self.config = config or AnthropicConfig.from_env()
        self._client = None

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise LLMClientError(
                    "Anthropic package not installed. "
                    "Install with: pip install anthropic"
                )
            self._client = Anthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )
        return self._client

    def chat(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Send a chat request to Anthropic API.

        Args:
            messages: List of chat messages.
            max_tokens: Override default max tokens.
            temperature: Override default temperature.

        Returns:
            LLM response.

        Raises:
            LLMClientError: If the API request fails.
        """
        client = self._get_client()

        # Anthropic expects system prompt separately
        system_prompt = None
        anthropic_messages = []

        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                anthropic_messages.append({"role": m.role, "content": m.content})

        try:
            kwargs = {
                "model": self.config.model,
                "messages": anthropic_messages,
                "max_tokens": max_tokens or self.config.max_tokens,
                "temperature": temperature if temperature is not None else self.config.temperature,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = client.messages.create(**kwargs)
        except Exception as e:
            raise LLMClientError(f"Anthropic API request failed: {e}") from e

        # Extract content from response
        content = ""
        if response.content:
            content = response.content[0].text

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Simple generation interface.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            max_tokens: Override default max tokens.
            temperature: Override default temperature.

        Returns:
            Generated text content.
        """
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))

        response = self.chat(messages, max_tokens=max_tokens, temperature=temperature)
        return response.content
