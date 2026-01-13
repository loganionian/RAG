"""Ollama LLM client implementation for local/offline use."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import requests

from ..api_client import LLMClientError
from ..base import LLMResponse, Message


@dataclass
class OllamaConfig:
    """Configuration for Ollama API client."""

    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 120  # Longer timeout for local models

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        """Create config from environment variables."""
        return cls(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        )


class OllamaClient:
    """Ollama API client implementing BaseLLMClient protocol.

    Ollama is a local LLM server for running models offline.
    See: https://ollama.ai/
    """

    def __init__(self, config: Optional[OllamaConfig] = None):
        """Initialize the Ollama client.

        Args:
            config: Ollama configuration. If None, loads from environment.
        """
        self.config = config or OllamaConfig.from_env()

    def chat(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Send a chat request to Ollama API.

        Args:
            messages: List of chat messages.
            max_tokens: Override default max tokens (maps to num_predict).
            temperature: Override default temperature.

        Returns:
            LLM response.

        Raises:
            LLMClientError: If the API request fails.
        """
        url = f"{self.config.base_url}/api/chat"

        ollama_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]

        request_body = {
            "model": self.config.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens or self.config.max_tokens,
                "temperature": temperature if temperature is not None else self.config.temperature,
            },
        }

        try:
            response = requests.post(
                url,
                json=request_body,
                timeout=self.config.timeout,
            )
        except requests.ConnectionError as e:
            raise LLMClientError(
                f"Could not connect to Ollama at {self.config.base_url}. "
                "Make sure Ollama is running: ollama serve"
            ) from e
        except requests.RequestException as e:
            raise LLMClientError(f"Ollama API request failed: {e}") from e

        if response.status_code != 200:
            raise LLMClientError(
                f"Ollama API returned status code {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise LLMClientError(f"Invalid JSON response from Ollama: {e}") from e

        try:
            content = data["message"]["content"]
        except KeyError as e:
            raise LLMClientError(f"Unexpected Ollama response format: {e}") from e

        # Extract usage info if available
        usage = None
        if "eval_count" in data or "prompt_eval_count" in data:
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            }

        return LLMResponse(
            content=content,
            model=data.get("model"),
            usage=usage,
            raw_response=data,
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

    def is_available(self) -> bool:
        """Check if Ollama server is running and accessible.

        Returns:
            True if Ollama is available, False otherwise.
        """
        try:
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        """List available models on the Ollama server.

        Returns:
            List of model names.

        Raises:
            LLMClientError: If the request fails.
        """
        try:
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=10,
            )
        except requests.RequestException as e:
            raise LLMClientError(f"Failed to list Ollama models: {e}") from e

        if response.status_code != 200:
            raise LLMClientError(f"Failed to list models: {response.text}")

        data = response.json()
        return [model["name"] for model in data.get("models", [])]
