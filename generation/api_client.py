"""HMAC-authenticated LLM API client."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import requests

from .base import LLMResponse, Message


@dataclass
class LLMConfig:
    """Configuration for LLM API client."""

    api_key: str
    api_secret: str
    base_url: str
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0
    presence_penalty: float = 0
    timeout: int = 60

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Create config from environment variables."""
        api_key = os.getenv("API_KEY")
        api_secret = os.getenv("API_SECRET")
        base_url = os.getenv("BASE_URL")

        if not api_key:
            raise ValueError("API_KEY environment variable is required")
        if not api_secret:
            raise ValueError("API_SECRET environment variable is required")
        if not base_url:
            raise ValueError("BASE_URL environment variable is required")

        return cls(api_key=api_key, api_secret=api_secret, base_url=base_url)


class LLMClientError(Exception):
    """Error from LLM API client."""

    pass


class LLMClient:
    """HMAC-authenticated LLM API client."""

    def __init__(self, config: LLMConfig):
        """Initialize the client.

        Args:
            config: LLM configuration.
        """
        self.config = config

    def _create_hmac_signature(
        self,
        request_body: dict,
        timestamp: float,
        request_id: uuid.UUID,
    ) -> str:
        """Create HMAC signature for request authentication.

        Args:
            request_body: The request payload.
            timestamp: Request timestamp in milliseconds.
            request_id: Unique request identifier.

        Returns:
            Base64-encoded HMAC signature.
        """
        hmac_source_data = (
            self.config.api_key
            + str(request_id)
            + str(timestamp)
            + json.dumps(request_body)
        )
        computed_hash = hmac.new(
            self.config.api_secret.encode(),
            hmac_source_data.encode(),
            hashlib.sha256,
        )
        return base64.b64encode(computed_hash.digest()).decode()

    def _build_request_body(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """Build the request body for the API.

        Args:
            messages: List of chat messages.
            max_tokens: Override default max tokens.
            temperature: Override default temperature.

        Returns:
            Request body dictionary.
        """
        return {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "frequency_penalty": self.config.frequency_penalty,
            "max_tokens": max_tokens or self.config.max_tokens,
            "n": 1,
            "presence_penalty": self.config.presence_penalty,
            "response_format": {"type": "text"},
            "stream": False,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "top_p": self.config.top_p,
        }

    def chat(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Send a chat request to the LLM API.

        Args:
            messages: List of chat messages.
            max_tokens: Override default max tokens.
            temperature: Override default temperature.

        Returns:
            LLM response.

        Raises:
            LLMClientError: If the API request fails.
        """
        request_body = self._build_request_body(messages, max_tokens, temperature)

        timestamp = time.time() * 1000
        request_id = uuid.uuid4()

        signature = self._create_hmac_signature(request_body, timestamp, request_id)

        headers = {
            "api-key": self.config.api_key,
            "Client-Request-Id": str(request_id),
            "Timestamp": str(timestamp),
            "Authorization": signature,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        url = f"{self.config.base_url}/chat/completions"

        try:
            response = requests.post(
                url,
                headers=headers,
                json=request_body,
                timeout=self.config.timeout,
            )
        except requests.RequestException as e:
            raise LLMClientError(f"Request failed: {e}") from e

        if response.status_code != 200:
            raise LLMClientError(
                f"API returned status code {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise LLMClientError(f"Invalid JSON response: {e}") from e

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMClientError(f"Unexpected response format: {e}") from e

        return LLMResponse(
            content=content,
            model=data.get("model"),
            usage=data.get("usage"),
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
