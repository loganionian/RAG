"""Generation module for RAG response generation.

This module provides LLM clients and the RAG chain for document-based Q&A.

Providers:
    - custom: HMAC-authenticated custom API (default, backwards compatible)
    - openai: OpenAI API (GPT-4, GPT-4o, etc.)
    - anthropic: Anthropic API (Claude)
    - ollama: Local Ollama server

Usage:
    # Use factory to get client based on LLM_PROVIDER env var
    from generation import create_llm_client, RAGChain
    client = create_llm_client()
    rag = RAGChain(client)

    # Or explicitly specify provider
    client = create_llm_client("openai")

    # Generate dataset summaries
    from generation import DatasetSummarizer, SummaryConfig
    summarizer = DatasetSummarizer(llm_client, sql_store, catalog)
    summary = summarizer.generate_summary("my_table")
"""

from .api_client import LLMClient, LLMClientError, LLMConfig
from .base import BaseLLMClient, LLMResponse, Message
from .cost_tracker import CostConfig, CostReport, CostTracker
from .dataset_summarizer import DatasetSummarizer, DatasetSummary, DatasetSummaryError, SummaryConfig
from .factory import ProviderError, create_llm_client, get_available_providers
from .rag_chain import RAGChain, RAGConfig, RAGResponse

__all__ = [
    # Core types
    "LLMClient",
    "LLMConfig",
    "LLMResponse",
    "LLMClientError",
    "Message",
    # Protocol
    "BaseLLMClient",
    # Factory
    "create_llm_client",
    "get_available_providers",
    "ProviderError",
    # RAG
    "RAGChain",
    "RAGConfig",
    "RAGResponse",
    # Dataset Summarization
    "DatasetSummarizer",
    "DatasetSummary",
    "DatasetSummaryError",
    "SummaryConfig",
    # Cost Tracking
    "CostConfig",
    "CostReport",
    "CostTracker",
]
