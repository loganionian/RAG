"""RAG chain combining retrieval and generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import chromadb
from chromadb.utils import embedding_functions

from .base import BaseLLMClient
from indexing.bm25_store import BM25Config, BM25Store
from indexing.embeddings import get_model_path
from retrieval.score_normalizer import NormalizationConfig, ScoreNormalizer
from retrieval.reranker import CrossEncoderReranker, RerankerConfig, RerankResult

if TYPE_CHECKING:
    from security import ACLFilter, SecurityContext

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

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
    # Reranking settings
    enable_reranking: bool = False
    reranker_model: str = DEFAULT_RERANKER_MODEL
    reranker_top_k_multiplier: int = 3  # Fetch k * multiplier candidates before reranking
    normalize_scores: bool = True  # Normalize scores before RRF fusion


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
        self._reranker: Optional[CrossEncoderReranker] = None
        self._score_normalizer: Optional[ScoreNormalizer] = None

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

    def _get_reranker(self) -> Optional[CrossEncoderReranker]:
        """Get or initialize the cross-encoder reranker."""
        if self._reranker is None and self.config.enable_reranking:
            reranker_config = RerankerConfig(
                model_name=self.config.reranker_model,
            )
            self._reranker = CrossEncoderReranker(reranker_config)
        return self._reranker

    def _get_score_normalizer(self) -> ScoreNormalizer:
        """Get or initialize the score normalizer."""
        if self._score_normalizer is None:
            self._score_normalizer = ScoreNormalizer(NormalizationConfig())
        return self._score_normalizer

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        mode: SearchMode = "vector",
        rerank: Optional[bool] = None,
        security_context: Optional["SecurityContext"] = None,
        acl_filter: Optional["ACLFilter"] = None,
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a query.

        Args:
            query: User query.
            k: Number of results to retrieve (overrides config).
            mode: Search mode - 'vector', 'lexical', or 'hybrid'.
            rerank: Whether to apply cross-encoder reranking. None uses config default.
            security_context: Optional security context for ACL filtering.
            acl_filter: Optional ACL filter instance for access control.

        Returns:
            Retrieved chunks with metadata.
        """
        n_results = k or self.config.top_k

        # Determine whether to rerank
        should_rerank = rerank if rerank is not None else self.config.enable_reranking

        # Build ACL where clause for vector search
        acl_where = None
        if acl_filter is not None and security_context is not None:
            acl_where = acl_filter.build_chroma_where(security_context)

        if mode == "vector":
            result = self._retrieve_vector(query, n_results, where=acl_where)
        elif mode == "lexical":
            result = self._retrieve_lexical(query, n_results)
            # Apply post-filtering for BM25 (doesn't support native filtering)
            if acl_filter is not None and security_context is not None:
                chunks, metadatas, ids, distances = acl_filter.filter_bm25_results(
                    result.chunks, result.metadatas, result.ids, result.distances,
                    security_context
                )
                result = RetrievalResult(
                    chunks=chunks,
                    metadatas=metadatas,
                    distances=distances,
                    ids=ids,
                )
        elif mode == "hybrid":
            result = self._retrieve_hybrid(
                query, n_results, rerank=should_rerank,
                acl_where=acl_where, acl_filter=acl_filter,
                security_context=security_context
            )
        else:
            logger.warning("Unknown search mode '%s', falling back to vector.", mode)
            result = self._retrieve_vector(query, n_results, where=acl_where)

        # Apply reranking if enabled (for non-hybrid modes - hybrid handles reranking internally)
        if should_rerank and mode != "hybrid" and result.chunks:
            result = self._apply_reranking(query, result, n_results)

        return result

    def _retrieve_vector(
        self, query: str, k: int, where: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Retrieve using vector (semantic) search.

        Args:
            query: User query.
            k: Number of results.
            where: Optional Chroma where clause for filtering.

        Returns:
            Retrieved chunks with metadata.
        """
        collection = self._get_collection()

        query_kwargs = {
            "query_texts": [query],
            "n_results": k,
        }
        if where is not None:
            query_kwargs["where"] = where

        result = collection.query(**query_kwargs)

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

    def _retrieve_hybrid(
        self,
        query: str,
        k: int,
        rerank: bool = False,
        acl_where: Optional[Dict[str, Any]] = None,
        acl_filter: Optional["ACLFilter"] = None,
        security_context: Optional["SecurityContext"] = None,
    ) -> RetrievalResult:
        """Retrieve using hybrid search with RRF fusion and optional reranking.

        Combines vector and lexical search results using Reciprocal Rank Fusion.
        Optionally applies cross-encoder reranking for improved relevance.

        Args:
            query: User query.
            k: Number of results to return.
            rerank: Whether to apply cross-encoder reranking.
            acl_where: Optional Chroma where clause for ACL filtering.
            acl_filter: Optional ACL filter for post-filtering BM25 results.
            security_context: Optional security context for ACL filtering.

        Returns:
            Fused results from both search methods.
        """
        # Determine how many candidates to fetch
        # If reranking, fetch more to ensure we have good candidates for the reranker
        multiplier = self.config.reranker_top_k_multiplier if rerank else 2
        fetch_k = k * multiplier

        vector_result = self._retrieve_vector(query, fetch_k, where=acl_where)
        lexical_result = self._retrieve_lexical(query, fetch_k)

        # Apply ACL post-filtering for lexical results
        if acl_filter is not None and security_context is not None:
            chunks, metadatas, ids, distances = acl_filter.filter_bm25_results(
                lexical_result.chunks, lexical_result.metadatas,
                lexical_result.ids, lexical_result.distances,
                security_context
            )
            lexical_result = RetrievalResult(
                chunks=chunks,
                metadatas=metadatas,
                distances=distances,
                ids=ids,
            )

        if not lexical_result.ids:
            # Fall back to vector-only if lexical search unavailable
            result = self._retrieve_vector(query, k)
            if rerank and result.chunks:
                result = self._apply_reranking(query, result, k)
            return result

        # Get normalized scores if enabled
        normalizer = self._get_score_normalizer() if self.config.normalize_scores else None

        # Build chunk data and compute RRF scores
        chunk_data: dict[str, dict] = {}
        chunk_rrf_scores: dict[str, float] = {}

        # Reciprocal Rank Fusion (RRF)
        # RRF score = sum(weight / (rrf_k + rank)) for each list
        rrf_k = 60  # Standard RRF constant

        # Process vector results
        if normalizer:
            # Normalize vector distances to similarity scores
            normalized_vector = normalizer.normalize_vector_distances(
                vector_result.distances, vector_result.ids
            )
            # Sort by normalized score (descending) to get ranks
            sorted_vector = sorted(
                range(len(normalized_vector)),
                key=lambda i: normalized_vector[i].normalized,
                reverse=True,
            )
            for rank, idx in enumerate(sorted_vector):
                chunk_id = vector_result.ids[idx]
                rrf_score = (1.0 - self.config.lexical_weight) / (rrf_k + rank + 1)
                chunk_rrf_scores[chunk_id] = chunk_rrf_scores.get(chunk_id, 0.0) + rrf_score
                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = {
                        "chunk": vector_result.chunks[idx],
                        "metadata": vector_result.metadatas[idx],
                        "distance": vector_result.distances[idx],
                        "vector_score": normalized_vector[idx].normalized,
                    }
        else:
            # Use original ranking (by distance, ascending)
            for rank, (chunk_id, chunk, metadata, distance) in enumerate(
                zip(
                    vector_result.ids,
                    vector_result.chunks,
                    vector_result.metadatas,
                    vector_result.distances,
                )
            ):
                rrf_score = (1.0 - self.config.lexical_weight) / (rrf_k + rank + 1)
                chunk_rrf_scores[chunk_id] = chunk_rrf_scores.get(chunk_id, 0.0) + rrf_score
                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = {
                        "chunk": chunk,
                        "metadata": metadata,
                        "distance": distance,
                    }

        # Process lexical results
        if normalizer:
            # Get original BM25 scores from the store
            bm25_store = self._get_bm25_store()
            bm25_results = bm25_store.search(query, k=fetch_k) if bm25_store else []
            bm25_scores = [r.score for r in bm25_results]
            bm25_ids = [r.chunk_id for r in bm25_results]

            # Normalize BM25 scores
            normalized_lexical = normalizer.normalize_bm25_scores(bm25_scores, bm25_ids)
            # Sort by normalized score (descending) to get ranks
            sorted_lexical = sorted(
                range(len(normalized_lexical)),
                key=lambda i: normalized_lexical[i].normalized,
                reverse=True,
            )
            for rank, idx in enumerate(sorted_lexical):
                chunk_id = lexical_result.ids[idx]
                rrf_score = self.config.lexical_weight / (rrf_k + rank + 1)
                chunk_rrf_scores[chunk_id] = chunk_rrf_scores.get(chunk_id, 0.0) + rrf_score
                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = {
                        "chunk": lexical_result.chunks[idx],
                        "metadata": lexical_result.metadatas[idx],
                        "distance": lexical_result.distances[idx],
                        "lexical_score": normalized_lexical[idx].normalized,
                    }
                else:
                    chunk_data[chunk_id]["lexical_score"] = normalized_lexical[idx].normalized
        else:
            # Use original ranking
            for rank, (chunk_id, chunk, metadata, distance) in enumerate(
                zip(
                    lexical_result.ids,
                    lexical_result.chunks,
                    lexical_result.metadatas,
                    lexical_result.distances,
                )
            ):
                rrf_score = self.config.lexical_weight / (rrf_k + rank + 1)
                chunk_rrf_scores[chunk_id] = chunk_rrf_scores.get(chunk_id, 0.0) + rrf_score
                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = {
                        "chunk": chunk,
                        "metadata": metadata,
                        "distance": distance,
                    }

        # Sort by RRF score (higher is better)
        sorted_ids = sorted(
            chunk_rrf_scores.keys(),
            key=lambda x: chunk_rrf_scores[x],
            reverse=True,
        )

        # If reranking, pass top candidates to reranker
        if rerank:
            reranker = self._get_reranker()
            if reranker:
                # Prepare candidates for reranking
                candidates_for_rerank = k * self.config.reranker_top_k_multiplier
                top_candidate_ids = sorted_ids[:candidates_for_rerank]

                candidates = [
                    {
                        "chunk_id": cid,
                        "text": chunk_data[cid]["chunk"],
                        "metadata": chunk_data[cid]["metadata"],
                        "score": chunk_rrf_scores[cid],
                    }
                    for cid in top_candidate_ids
                ]

                reranked = reranker.rerank(query, candidates, top_k=k)

                # Build result from reranked items
                return RetrievalResult(
                    chunks=[r.text for r in reranked],
                    metadatas=[r.metadata for r in reranked],
                    distances=[1.0 - r.rerank_score for r in reranked],  # Convert score to pseudo-distance
                    ids=[r.chunk_id for r in reranked],
                )

        # No reranking - take top k by RRF score
        top_k_ids = sorted_ids[:k]

        # Build result
        chunks = []
        metadatas = []
        distances = []
        ids = []

        for chunk_id in top_k_ids:
            data = chunk_data[chunk_id]
            ids.append(chunk_id)
            chunks.append(data["chunk"])
            metadatas.append(data["metadata"])
            # Use inverse RRF score as pseudo-distance (lower = better)
            distances.append(1.0 / (chunk_rrf_scores[chunk_id] + 0.001))

        return RetrievalResult(
            chunks=chunks,
            metadatas=metadatas,
            distances=distances,
            ids=ids,
        )

    def _apply_reranking(
        self, query: str, result: RetrievalResult, k: int
    ) -> RetrievalResult:
        """Apply cross-encoder reranking to retrieval results.

        Args:
            query: User query.
            result: Original retrieval result.
            k: Number of results to return.

        Returns:
            Reranked retrieval result.
        """
        reranker = self._get_reranker()
        if not reranker:
            return result

        # Prepare candidates for reranking
        candidates = [
            {
                "chunk_id": result.ids[i],
                "text": result.chunks[i],
                "metadata": result.metadatas[i],
                "score": result.distances[i],
            }
            for i in range(len(result.chunks))
        ]

        reranked = reranker.rerank(query, candidates, top_k=k)

        return RetrievalResult(
            chunks=[r.text for r in reranked],
            metadatas=[r.metadata for r in reranked],
            distances=[1.0 - r.rerank_score for r in reranked],  # Convert score to pseudo-distance
            ids=[r.chunk_id for r in reranked],
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
