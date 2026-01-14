"""Chroma indexer for dataset summaries.

This module indexes LLM-generated dataset summaries in Chroma
for semantic discovery of tabular data sources.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import chromadb

from .chroma_store import get_collection
from .embeddings import EmbeddingService, build_embedding_function

if TYPE_CHECKING:
    from generation.dataset_summarizer import DatasetSummary

logger = logging.getLogger(__name__)

# Default collection name for dataset summaries
DEFAULT_SUMMARY_COLLECTION = "dataset-summaries"


@dataclass
class SummaryIndexConfig:
    """Configuration for summary indexing.

    Attributes:
        chroma_dir: Path to Chroma persistence directory.
        collection_name: Name of the Chroma collection.
        embedding_model: Model for generating embeddings.
    """

    chroma_dir: Path
    collection_name: str = DEFAULT_SUMMARY_COLLECTION
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class DatasetSearchResult:
    """Result from a dataset search query.

    Attributes:
        table_name: Name of the SQL table.
        summary: LLM-generated summary text.
        source_file: Original source file path.
        relevance_score: Similarity score (lower is more similar for distance).
        row_count: Number of rows in the dataset.
        column_count: Number of columns in the dataset.
        domain_labels: Domain labels for the dataset.
        column_descriptions: Column descriptions from summary.
    """

    table_name: str
    summary: str
    source_file: str
    relevance_score: float
    row_count: int = 0
    column_count: int = 0
    domain_labels: List[str] = field(default_factory=list)
    column_descriptions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "table_name": self.table_name,
            "summary": self.summary,
            "source_file": self.source_file,
            "relevance_score": self.relevance_score,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "domain_labels": self.domain_labels,
            "column_descriptions": self.column_descriptions,
        }


class SummaryIndexer:
    """Indexes dataset summaries in Chroma for semantic discovery.

    Usage:
        config = SummaryIndexConfig(chroma_dir=Path("data/vectorstore"))
        indexer = SummaryIndexer(config)

        # Index a summary
        indexer.index_summary(summary)

        # Search datasets
        results = indexer.search_datasets("What sales data do we have?")
        for result in results:
            print(f"{result.table_name}: {result.summary}")
    """

    def __init__(self, config: SummaryIndexConfig) -> None:
        """Initialize the summary indexer.

        Args:
            config: Configuration for summary indexing.
        """
        self.config = config
        self._collection = None
        self._embedding_service = EmbeddingService()

    @property
    def collection(self):
        """Lazy-load the Chroma collection."""
        if self._collection is None:
            embedding_fn = build_embedding_function(self.config.embedding_model)
            self._collection = get_collection(
                persist_path=self.config.chroma_dir,
                collection_name=self.config.collection_name,
                embedding_function=embedding_fn,
            )
            logger.info(
                "Connected to collection '%s' (%d documents)",
                self.config.collection_name,
                self._collection.count(),
            )
        return self._collection

    def _make_document_id(self, table_name: str) -> str:
        """Create a document ID for a dataset summary.

        Args:
            table_name: Name of the table.

        Returns:
            Document ID in format "dataset::{table_name}".
        """
        return f"dataset::{table_name}"

    def _build_document_text(self, summary: "DatasetSummary") -> str:
        """Build the document text for indexing.

        Combines summary text with column information for better retrieval.

        Args:
            summary: DatasetSummary object.

        Returns:
            Combined document text.
        """
        parts = [summary.summary]

        # Add column names for semantic matching
        if summary.column_descriptions:
            column_list = ", ".join(summary.column_descriptions.keys())
            parts.append(f"\nColumns: {column_list}")

            # Add column descriptions for richer semantic content
            col_descs = [f"- {col}: {desc}" for col, desc in summary.column_descriptions.items()]
            if col_descs:
                parts.append("\nColumn descriptions:")
                parts.extend(col_descs)

        return "\n".join(parts)

    def _build_metadata(self, summary: "DatasetSummary") -> dict:
        """Build metadata for a dataset summary document.

        Args:
            summary: DatasetSummary object.

        Returns:
            Metadata dictionary for Chroma.
        """
        return {
            "doc_type": "dataset_summary",
            "table_name": summary.table_name,
            "source_file": summary.source_file,
            "row_count": summary.row_count,
            "column_count": summary.column_count,
            "domain_labels": json.dumps(summary.domain_labels),
            "generated_at": summary.generated_at.isoformat(),
            "column_descriptions": json.dumps(summary.column_descriptions),
        }

    def index_summary(self, summary: "DatasetSummary") -> str:
        """Index a single dataset summary.

        Args:
            summary: DatasetSummary to index.

        Returns:
            Document ID of the indexed summary.
        """
        doc_id = self._make_document_id(summary.table_name)
        document_text = self._build_document_text(summary)
        metadata = self._build_metadata(summary)

        # Upsert to handle updates
        self.collection.upsert(
            ids=[doc_id],
            documents=[document_text],
            metadatas=[metadata],
        )

        logger.info("Indexed summary for '%s' (id=%s)", summary.table_name, doc_id)
        return doc_id

    def index_summaries_batch(self, summaries: List["DatasetSummary"]) -> int:
        """Index multiple dataset summaries.

        Args:
            summaries: List of DatasetSummary objects to index.

        Returns:
            Number of summaries indexed.
        """
        if not summaries:
            return 0

        ids = []
        documents = []
        metadatas = []

        for summary in summaries:
            ids.append(self._make_document_id(summary.table_name))
            documents.append(self._build_document_text(summary))
            metadatas.append(self._build_metadata(summary))

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info("Indexed %d summaries in batch", len(summaries))
        return len(summaries)

    def search_datasets(
        self,
        query: str,
        k: int = 5,
        filter_domain: Optional[str] = None,
    ) -> List[DatasetSearchResult]:
        """Search for datasets matching a query.

        Args:
            query: Natural language query.
            k: Number of results to return.
            filter_domain: Optional domain label to filter by.

        Returns:
            List of DatasetSearchResult objects, ordered by relevance.
        """
        where_filter = None
        if filter_domain:
            # Filter by domain label (stored as JSON array)
            where_filter = {"domain_labels": {"$contains": filter_domain}}

        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []

        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                document = results["documents"][0][i] if results["documents"] else ""
                distance = results["distances"][0][i] if results["distances"] else 0.0

                # Parse stored JSON fields
                domain_labels = []
                if metadata.get("domain_labels"):
                    try:
                        domain_labels = json.loads(metadata["domain_labels"])
                    except json.JSONDecodeError:
                        pass

                column_descriptions = {}
                if metadata.get("column_descriptions"):
                    try:
                        column_descriptions = json.loads(metadata["column_descriptions"])
                    except json.JSONDecodeError:
                        pass

                # Extract summary from document (first line is the summary)
                summary_text = document.split("\n")[0] if document else ""

                result = DatasetSearchResult(
                    table_name=metadata.get("table_name", doc_id.replace("dataset::", "")),
                    summary=summary_text,
                    source_file=metadata.get("source_file", ""),
                    relevance_score=distance,
                    row_count=metadata.get("row_count", 0),
                    column_count=metadata.get("column_count", 0),
                    domain_labels=domain_labels,
                    column_descriptions=column_descriptions,
                )
                search_results.append(result)

        return search_results

    def delete_summary(self, table_name: str) -> bool:
        """Delete a dataset summary from the index.

        Args:
            table_name: Name of the table.

        Returns:
            True if deleted, False if not found.
        """
        doc_id = self._make_document_id(table_name)

        try:
            # Check if exists first
            existing = self.collection.get(ids=[doc_id])
            if not existing["ids"]:
                logger.warning("Summary for '%s' not found in index", table_name)
                return False

            self.collection.delete(ids=[doc_id])
            logger.info("Deleted summary for '%s' from index", table_name)
            return True
        except Exception as e:
            logger.error("Failed to delete summary for '%s': %s", table_name, e)
            return False

    def get_summary(self, table_name: str) -> Optional[DatasetSearchResult]:
        """Get a specific dataset summary by table name.

        Args:
            table_name: Name of the table.

        Returns:
            DatasetSearchResult if found, None otherwise.
        """
        doc_id = self._make_document_id(table_name)

        try:
            result = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"],
            )

            if not result["ids"]:
                return None

            metadata = result["metadatas"][0] if result["metadatas"] else {}
            document = result["documents"][0] if result["documents"] else ""

            # Parse stored JSON fields
            domain_labels = []
            if metadata.get("domain_labels"):
                try:
                    domain_labels = json.loads(metadata["domain_labels"])
                except json.JSONDecodeError:
                    pass

            column_descriptions = {}
            if metadata.get("column_descriptions"):
                try:
                    column_descriptions = json.loads(metadata["column_descriptions"])
                except json.JSONDecodeError:
                    pass

            summary_text = document.split("\n")[0] if document else ""

            return DatasetSearchResult(
                table_name=table_name,
                summary=summary_text,
                source_file=metadata.get("source_file", ""),
                relevance_score=0.0,
                row_count=metadata.get("row_count", 0),
                column_count=metadata.get("column_count", 0),
                domain_labels=domain_labels,
                column_descriptions=column_descriptions,
            )
        except Exception as e:
            logger.error("Failed to get summary for '%s': %s", table_name, e)
            return None

    def list_all(self) -> List[DatasetSearchResult]:
        """List all indexed dataset summaries.

        Returns:
            List of all DatasetSearchResult objects.
        """
        try:
            # Get all documents from collection
            result = self.collection.get(
                include=["documents", "metadatas"],
            )

            summaries = []
            if result["ids"]:
                for i, doc_id in enumerate(result["ids"]):
                    metadata = result["metadatas"][i] if result["metadatas"] else {}
                    document = result["documents"][i] if result["documents"] else ""

                    domain_labels = []
                    if metadata.get("domain_labels"):
                        try:
                            domain_labels = json.loads(metadata["domain_labels"])
                        except json.JSONDecodeError:
                            pass

                    column_descriptions = {}
                    if metadata.get("column_descriptions"):
                        try:
                            column_descriptions = json.loads(metadata["column_descriptions"])
                        except json.JSONDecodeError:
                            pass

                    summary_text = document.split("\n")[0] if document else ""

                    summaries.append(
                        DatasetSearchResult(
                            table_name=metadata.get("table_name", doc_id.replace("dataset::", "")),
                            summary=summary_text,
                            source_file=metadata.get("source_file", ""),
                            relevance_score=0.0,
                            row_count=metadata.get("row_count", 0),
                            column_count=metadata.get("column_count", 0),
                            domain_labels=domain_labels,
                            column_descriptions=column_descriptions,
                        )
                    )

            return summaries
        except Exception as e:
            logger.error("Failed to list summaries: %s", e)
            return []

    def count(self) -> int:
        """Get the number of indexed summaries.

        Returns:
            Count of indexed summaries.
        """
        return self.collection.count()
