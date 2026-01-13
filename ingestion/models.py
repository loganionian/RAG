from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(slots=True)
class Document:
    """Represents a parsed document ready for chunking."""

    doc_id: str
    path: Path
    source_type: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass(slots=True)
class DocumentChunk:
    """Single chunk produced from a document."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    token_estimate: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "token_estimate": self.token_estimate,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class FailureInfo:
    """Structured information about a document processing failure."""

    source_path: str
    error_type: str
    error_message: str
    traceback: str
    timestamp: str
    doc_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "doc_id": self.doc_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback": self.traceback,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureInfo":
        return cls(
            source_path=data["source_path"],
            doc_id=data.get("doc_id"),
            error_type=data["error_type"],
            error_message=data["error_message"],
            traceback=data["traceback"],
            timestamp=data["timestamp"],
        )
