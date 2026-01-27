"""Pydantic request/response models for the RAG API."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for RAG query endpoint."""

    question: str = Field(..., description="The question to ask the RAG system")
    k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    agent_id: Optional[str] = Field(
        default=None,
        description="Optional agent ID for custom system prompt and parameters",
    )
    search_mode: str = Field(
        default="vector",
        description="Search mode: 'vector' (semantic), 'lexical' (BM25), or 'hybrid' (RRF fusion)",
    )
    rerank: Optional[bool] = Field(
        default=None,
        description="Enable cross-encoder reranking. None uses server config default.",
    )


class SourceInfo(BaseModel):
    """Information about a retrieved source chunk."""

    doc_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    chunk_id: str = Field(..., description="Chunk identifier")
    snippet: str = Field(..., description="Text snippet from the chunk")
    page: Optional[int] = Field(default=None, description="Page number if available")


class QueryMetadata(BaseModel):
    """Metadata about query execution."""

    retrieval_time_ms: float = Field(..., description="Time spent on retrieval in milliseconds")
    generation_time_ms: float = Field(..., description="Time spent on generation in milliseconds")
    agent_id: Optional[str] = Field(
        default=None, description="Agent ID used for this query (if any)"
    )
    search_mode: str = Field(
        default="vector", description="Search mode used: 'vector', 'lexical', or 'hybrid'"
    )
    reranking_applied: bool = Field(
        default=False, description="Whether cross-encoder reranking was applied"
    )


class QueryResponse(BaseModel):
    """Response model for RAG query endpoint."""

    answer: str = Field(..., description="Generated answer from the LLM")
    sources: list[SourceInfo] = Field(default_factory=list, description="Retrieved source chunks")
    metadata: QueryMetadata = Field(..., description="Query execution metadata")


class IngestResponse(BaseModel):
    """Response model for document ingestion endpoint."""

    doc_id: str = Field(..., description="Assigned document identifier")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Ingestion status (success, failed)")
    chunks_created: int = Field(..., description="Number of chunks created")
    message: Optional[str] = Field(default=None, description="Additional status message")


class DocumentInfo(BaseModel):
    """Information about an ingested document."""

    doc_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File extension/type")
    chunks: int = Field(..., description="Number of chunks")
    ingested_at: datetime = Field(..., description="Ingestion timestamp")


class DocumentsResponse(BaseModel):
    """Response model for documents list endpoint."""

    documents: list[DocumentInfo] = Field(default_factory=list, description="List of documents")
    total: int = Field(..., description="Total number of documents")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Overall health status")
    vectorstore: dict = Field(..., description="Vectorstore health details")
    llm_provider: dict = Field(..., description="LLM provider health details")
    bm25_index: Optional[dict] = Field(
        default=None, description="BM25 lexical index health details (if enabled)"
    )


class ErrorResponse(BaseModel):
    """Standard error response format."""

    detail: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error class name")


# ============================================================================
# Agent Schemas
# ============================================================================


class AgentBase(BaseModel):
    """Base agent fields for create/update operations."""

    name: str = Field(..., min_length=1, max_length=100, description="Unique agent name")
    description: Optional[str] = Field(
        default=None, max_length=500, description="Short description of the agent"
    )
    system_prompt: str = Field(
        ...,
        min_length=1,
        description="System prompt template with {context} placeholder",
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="LLM temperature (0.0-2.0)"
    )
    max_tokens: Optional[int] = Field(
        default=None, ge=1, le=32000, description="Maximum response tokens"
    )


class AgentCreate(AgentBase):
    """Request model for creating a new agent."""

    pass


class AgentUpdate(BaseModel):
    """Request model for updating an existing agent.

    All fields are optional - only provided fields will be updated.
    """

    name: Optional[str] = Field(
        default=None, min_length=1, max_length=100, description="New agent name"
    )
    description: Optional[str] = Field(
        default=None, max_length=500, description="New description"
    )
    system_prompt: Optional[str] = Field(
        default=None, min_length=1, description="New system prompt"
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="New temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None, ge=1, le=32000, description="New max tokens"
    )


class AgentResponse(BaseModel):
    """Response model for a single agent."""

    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent name")
    description: Optional[str] = Field(default=None, description="Agent description")
    system_prompt: str = Field(..., description="System prompt template")
    temperature: Optional[float] = Field(default=None, description="LLM temperature")
    max_tokens: Optional[int] = Field(default=None, description="Max response tokens")
    is_default: bool = Field(..., description="Whether this is the default agent")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class AgentsListResponse(BaseModel):
    """Response model for listing agents."""

    agents: list[AgentResponse] = Field(default_factory=list, description="List of agents")
    total: int = Field(..., description="Total number of agents")


class AgentDeleteResponse(BaseModel):
    """Response model for agent deletion."""

    deleted: bool = Field(..., description="Whether the agent was deleted")
