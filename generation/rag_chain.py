"""RAG chain combining retrieval and generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import chromadb
from chromadb.utils import embedding_functions

from .base import BaseLLMClient
from indexing.bm25_store import BM25Config, BM25Store
from indexing.embeddings import get_model_path

logger = logging.getLogger(__name__)

# Type alias for search mode
SearchMode = Literal["vector", "lexical", "hybrid"]


DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Instructions:
- Answer the question using ONLY the information from the context below
- If the context doesn't contain enough information to answer, say so clearly
- Be concise and direct in your responses
- Cite specific details from the context when relevant

Context:
{context}"""


@dataclass
class RetrievalResult:
    """Result from retrieval step."""

    chunks: list[str]
    metadatas: list[dict]
    distances: list[float]
    ids: list[str]


@dataclass
class RAGResponse:
    """Response from RAG chain."""

    answer: str
    retrieved_chunks: list[str]
    metadatas: list[dict]
    distances: list[float]


@dataclass
class RAGConfig:
    """Configuration for RAG chain."""

    vectorstore_dir: Path = field(default_factory=lambda: Path("data/vectorstore"))
    collection_name: str = "pilot-docs"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    max_tokens: int = 1000
    temperature: float = 0.7
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # Lexical search settings
    enable_lexical: bool = False
    lexical_weight: float = 0.3  # Weight for lexical results in hybrid mode (0-1)
    bm25_dir: Optional[Path] = None  # Defaults to vectorstore_dir if None


class RAGChain:
    """RAG chain that retrieves context and generates responses."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        config: Optional[RAGConfig] = None,
    ):
        """Initialize the RAG chain.

        Args:
            llm_client: LLM client for generation. Can be any client
                       implementing the BaseLLMClient protocol.
            config: RAG configuration.
        """
        self.llm_client = llm_client
        self.config = config or RAGConfig()
        self._collection = None
        self._chroma_client = None
        self._bm25_store: Optional[BM25Store] = None

    def _get_collection(self):
        """Get or initialize the Chroma collection."""
        if self._collection is None:
            self._chroma_client = chromadb.PersistentClient(
                path=str(self.config.vectorstore_dir)
            )
            # Resolve model path (prefers local model if available)
            resolved_model = get_model_path(self.config.embedding_model)
            embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=resolved_model,
            )
            self._collection = self._chroma_client.get_collection(
                self.config.collection_name,
                embedding_function=embedding_fn,
            )
        return self._collection

    def _get_bm25_store(self) -> Optional[BM25Store]:
        """Get or initialize the BM25 store."""
        if self._bm25_store is None and self.config.enable_lexical:
            bm25_dir = self.config.bm25_dir or self.config.vectorstore_dir
            bm25_config = BM25Config(
                index_dir=bm25_dir,
                index_name=self.config.collection_name,
            )
            self._bm25_store = BM25Store(bm25_config)
        return self._bm25_store

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        mode: SearchMode = "vector",
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a query.

        Args:
            query: User query.
            k: Number of results to retrieve (overrides config).
            mode: Search mode - 'vector', 'lexical', or 'hybrid'.

        Returns:
            Retrieved chunks with metadata.
        """
        n_results = k or self.config.top_k

        if mode == "vector":
            return self._retrieve_vector(query, n_results)
        elif mode == "lexical":
            return self._retrieve_lexical(query, n_results)
        elif mode == "hybrid":
            return self._retrieve_hybrid(query, n_results)
        else:
            logger.warning("Unknown search mode '%s', falling back to vector.", mode)
            return self._retrieve_vector(query, n_results)

    def _retrieve_vector(self, query: str, k: int) -> RetrievalResult:
        """Retrieve using vector (semantic) search.

        Args:
            query: User query.
            k: Number of results.

        Returns:
            Retrieved chunks with metadata.
        """
        collection = self._get_collection()

        result = collection.query(
            query_texts=[query],
            n_results=k,
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0] if result.get("distances") else []

        return RetrievalResult(
            chunks=documents,
            metadatas=metadatas,
            distances=distances,
            ids=ids,
        )

    def _retrieve_lexical(self, query: str, k: int) -> RetrievalResult:
        """Retrieve using BM25 lexical search.

        Args:
            query: User query.
            k: Number of results.

        Returns:
            Retrieved chunks with metadata.
        """
        bm25_store = self._get_bm25_store()
        if bm25_store is None:
            logger.warning("BM25 store not initialized. Enable lexical search in config.")
            return RetrievalResult(chunks=[], metadatas=[], distances=[], ids=[])

        results = bm25_store.search(query, k=k)

        # Convert BM25 scores to pseudo-distances (lower is better, like Chroma)
        # BM25 scores are higher = better, so we invert
        max_score = max((r.score for r in results), default=1.0)
        distances = [
            1.0 - (r.score / max_score) if max_score > 0 else 0.0
            for r in results
        ]

        return RetrievalResult(
            chunks=[r.text for r in results],
            metadatas=[r.metadata for r in results],
            distances=distances,
            ids=[r.chunk_id for r in results],
        )

    def _retrieve_hybrid(self, query: str, k: int) -> RetrievalResult:
        """Retrieve using hybrid search with RRF fusion.

        Combines vector and lexical search results using Reciprocal Rank Fusion.

        Args:
            query: User query.
            k: Number of results to return.

        Returns:
            Fused results from both search methods.
        """
        # Get more results from each method to ensure good fusion
        fetch_k = k * 2

        vector_result = self._retrieve_vector(query, fetch_k)
        lexical_result = self._retrieve_lexical(query, fetch_k)

        if not lexical_result.ids:
            # Fall back to vector-only if lexical search unavailable
            return self._retrieve_vector(query, k)

        # Reciprocal Rank Fusion (RRF)
        # RRF score = sum(1 / (rrf_k + rank)) for each list
        rrf_k = 60  # Standard RRF constant

        chunk_scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        # Score vector results
        for rank, (chunk_id, chunk, metadata, distance) in enumerate(
            zip(
                vector_result.ids,
                vector_result.chunks,
                vector_result.metadatas,
                vector_result.distances,
            )
        ):
            rrf_score = (1.0 - self.config.lexical_weight) / (rrf_k + rank + 1)
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + rrf_score
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = {
                    "chunk": chunk,
                    "metadata": metadata,
                    "distance": distance,
                }

        # Score lexical results
        for rank, (chunk_id, chunk, metadata, distance) in enumerate(
            zip(
                lexical_result.ids,
                lexical_result.chunks,
                lexical_result.metadatas,
                lexical_result.distances,
            )
        ):
            rrf_score = self.config.lexical_weight / (rrf_k + rank + 1)
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + rrf_score
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = {
                    "chunk": chunk,
                    "metadata": metadata,
                    "distance": distance,
                }

        # Sort by RRF score (higher is better) and take top k
        sorted_ids = sorted(chunk_scores.keys(), key=lambda x: chunk_scores[x], reverse=True)[:k]

        # Build result
        chunks = []
        metadatas = []
        distances = []
        ids = []

        for chunk_id in sorted_ids:
            data = chunk_data[chunk_id]
            ids.append(chunk_id)
            chunks.append(data["chunk"])
            metadatas.append(data["metadata"])
            # Use inverse RRF score as pseudo-distance (lower = better)
            distances.append(1.0 / (chunk_scores[chunk_id] + 0.001))

        return RetrievalResult(
            chunks=chunks,
            metadatas=metadatas,
            distances=distances,
            ids=ids,
        )

    def _format_context(self, retrieval: RetrievalResult) -> str:
        """Format retrieved chunks as context string.

        Args:
            retrieval: Retrieved chunks.

        Returns:
            Formatted context string.
        """
        context_parts = []
        for i, (chunk, metadata) in enumerate(
            zip(retrieval.chunks, retrieval.metadatas), start=1
        ):
            # Build source info
            source_parts = []
            if "relative_path" in metadata:
                source_parts.append(metadata["relative_path"])
            if "page" in metadata:
                source_parts.append(f"page {metadata['page']}")
            if "section" in metadata:
                source_parts.append(f"section: {metadata['section']}")

            source_info = " | ".join(source_parts) if source_parts else "unknown source"
            context_parts.append(f"[{i}] ({source_info})\n{chunk}")

        return "\n\n---\n\n".join(context_parts)

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        search_mode: SearchMode = "vector",
    ) -> RAGResponse:
        """Query the RAG system.

        Args:
            question: User question.
            k: Number of chunks to retrieve.
            max_tokens: Override default max tokens.
            temperature: Override default temperature.
            search_mode: Search mode - 'vector', 'lexical', or 'hybrid'.

        Returns:
            RAG response with answer and retrieved context.
        """
        # Retrieve relevant chunks
        retrieval = self.retrieve(question, k=k, mode=search_mode)

        if not retrieval.chunks:
            return RAGResponse(
                answer="I couldn't find any relevant information in the knowledge base.",
                retrieved_chunks=[],
                metadatas=[],
                distances=[],
            )

        # Format context and build prompt
        context = self._format_context(retrieval)
        system_prompt = self.config.system_prompt.format(context=context)

        # Generate response
        answer = self.llm_client.generate(
            prompt=question,
            system_prompt=system_prompt,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
        )

        return RAGResponse(
            answer=answer,
            retrieved_chunks=retrieval.chunks,
            metadatas=retrieval.metadatas,
            distances=retrieval.distances,
        )

    def chat_loop(self, k: Optional[int] = None, show_sources: bool = False):
        """Run an interactive chat loop.

        Args:
            k: Number of chunks to retrieve per query.
            show_sources: Whether to show retrieved sources.
        """
        print("=== RAG Chat ===")
        print("Type 'exit' or 'quit' to end the conversation\n")

        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if question.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if not question:
                continue

            try:
                response = self.query(question, k=k)
                print(f"\nAssistant: {response.answer}\n")

                if show_sources and response.retrieved_chunks:
                    print("--- Sources ---")
                    for i, (chunk, metadata) in enumerate(
                        zip(response.retrieved_chunks, response.metadatas), start=1
                    ):
                        source = metadata.get("relative_path", "unknown")
                        page = metadata.get("page", "")
                        page_str = f" (page {page})" if page else ""
                        snippet = chunk[:100].replace("\n", " ") + "..."
                        print(f"  [{i}] {source}{page_str}: {snippet}")
                    print()

            except Exception as e:
                print(f"\nError: {e}\n")
