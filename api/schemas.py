"""Pydantic request/response models for the RAG API."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for RAG query endpoint."""

    question: str = Field(..., description="The question to ask the RAG system")
    k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


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


class ErrorResponse(BaseModel):
    """Standard error response format."""

    detail: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error class name")
