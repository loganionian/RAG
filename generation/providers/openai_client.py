"""OpenAI LLM client implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from ..api_client import LLMClientError
from ..base import LLMResponse, Message


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI API client."""

    api_key: str
    model: str = "gpt-4o"
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 60

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        """Create config from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        return cls(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        )


class OpenAIClient:
    """OpenAI API client implementing BaseLLMClient protocol."""

    def __init__(self, config: Optional[OpenAIConfig] = None):
        """Initialize the OpenAI client.

        Args:
            config: OpenAI configuration. If None, loads from environment.
        """
        self.config = config or OpenAIConfig.from_env()
        self._client = None

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise LLMClientError(
                    "OpenAI package not installed. "
                    "Install with: pip install openai"
                )
            self._client = OpenAI(
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
        """Send a chat request to OpenAI API.

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

        openai_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]

        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=openai_messages,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature if temperature is not None else self.config.temperature,
            )
        except Exception as e:
            raise LLMClientError(f"OpenAI API request failed: {e}") from e

        content = response.choices[0].message.content or ""

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
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
