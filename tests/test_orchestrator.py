"""Tests for the LangGraph orchestrator."""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from orchestrator import (
    CritiqueResult,
    Evidence,
    ExecutionStatus,
    GraphState,
    NodeError,
    OrchestratorConfig,
    PartialAnswer,
    RAGOrchestrator,
    RetrievalError,
    RoutingError,
    SynthesisError,
    create_initial_state,
    extract_errors_from_state,
    get_node_status_summary,
    graph_state_to_response,
    has_fatal_errors,
)
from router.models import QueryType, RouteDecision
from security.config import SecurityContext


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, response: str = "Mock answer"):
        self.response = response
        self.calls = []

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        self.calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        return self.response


class MockRAGChain:
    """Mock RAG chain for testing."""

    def __init__(
        self,
        chunks: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None,
        llm_client: Optional[MockLLMClient] = None,
    ):
        self.llm_client = llm_client or MockLLMClient("Document answer")
        self._chunks = chunks or ["Chunk 1", "Chunk 2"]
        self._metadatas = metadatas or [
            {"doc_id": "doc1", "relative_path": "doc1.pdf", "page": 1},
            {"doc_id": "doc1", "relative_path": "doc1.pdf", "page": 2},
        ]
        self.config = MagicMock()
        self.config.system_prompt = "Context: {context}"
        self.config.max_tokens = 1000
        self.config.temperature = 0.7
        self.config.enable_reranking = False

    def retrieve(self, query: str, k: int = 5, mode: str = "hybrid", **kwargs):
        result = MagicMock()
        result.chunks = self._chunks
        result.metadatas = self._metadatas
        result.ids = [f"doc1::chunk-{i:04d}" for i in range(len(self._chunks))]
        result.distances = [0.1 * i for i in range(len(self._chunks))]
        return result

    def _format_context(self, retrieval_result) -> str:
        return "\n".join(retrieval_result.chunks)


class MockSQLChain:
    """Mock SQL chain for testing."""

    def __init__(
        self,
        data: Optional[List[Dict]] = None,
        columns: Optional[List[str]] = None,
        sql: str = "SELECT * FROM test",
    ):
        self._data = data or [{"col1": "value1", "col2": 100}]
        self._columns = columns or ["col1", "col2"]
        self._sql = sql

    def query(
        self,
        question: str,
        max_rows: int = 100,
        summarize: bool = True,
        **kwargs,
    ):
        result = MagicMock()
        result.answer = "SQL answer: Found 1 row"
        result.data = self._data
        result.columns = self._columns
        result.row_count = len(self._data)
        result.generated_sql = self._sql
        result.tables_used = ["test"]
        result.generation_time_ms = 10.0
        result.execution_time_ms = 5.0
        result.summarization_time_ms = 15.0
        return result


class TestExecutionStatus:
    """Tests for ExecutionStatus enum."""

    def test_status_values(self):
        """ExecutionStatus should have expected values."""
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.IN_PROGRESS.value == "in_progress"
        assert ExecutionStatus.SUCCESS.value == "success"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.SKIPPED.value == "skipped"


class TestNodeError:
    """Tests for NodeError dataclass."""

    def test_from_exception(self):
        """NodeError.from_exception should extract exception details."""
        try:
            raise ValueError("Test error message")
        except Exception as e:
            error = NodeError.from_exception("test_node", e)

        assert error.node_name == "test_node"
        assert error.error_type == "ValueError"
        assert error.message == "Test error message"
        assert "ValueError" in error.traceback_str
        assert error.recoverable is True

    def test_to_dict(self):
        """NodeError should serialize to dict."""
        error = NodeError(
            node_name="router",
            error_type="RoutingError",
            message="Failed to route",
            traceback_str="traceback...",
            recoverable=False,
        )
        result = error.to_dict()
        assert result["node_name"] == "router"
        assert result["error_type"] == "RoutingError"
        assert result["recoverable"] is False


class TestEvidence:
    """Tests for Evidence dataclass."""

    def test_docs_evidence(self):
        """Evidence for docs should store chunks."""
        evidence = Evidence(
            source="docs",
            chunks=["chunk1", "chunk2"],
            metadatas=[{"doc_id": "doc1"}],
            chunk_ids=["id1", "id2"],
            distances=[0.1, 0.2],
        )
        assert evidence.source == "docs"
        assert len(evidence.chunks) == 2
        assert evidence.sql_data == []

    def test_sql_evidence(self):
        """Evidence for SQL should store query results."""
        evidence = Evidence(
            source="sql",
            sql_data=[{"col": "value"}],
            sql_columns=["col"],
            sql_query="SELECT * FROM test",
            tables_used=["test"],
        )
        assert evidence.source == "sql"
        assert len(evidence.sql_data) == 1
        assert evidence.sql_query == "SELECT * FROM test"


class TestPartialAnswer:
    """Tests for PartialAnswer dataclass."""

    def test_partial_answer(self):
        """PartialAnswer should store answer with metadata."""
        answer = PartialAnswer(
            source="docs",
            answer="The document says...",
            confidence=0.8,
            timing_ms=150.5,
        )
        assert answer.source == "docs"
        assert answer.confidence == 0.8
        assert answer.timing_ms == 150.5


class TestCreateInitialState:
    """Tests for create_initial_state function."""

    def test_basic_state(self):
        """create_initial_state should create valid state."""
        state = create_initial_state(
            question="What is the policy?",
        )
        assert state["question"] == "What is the policy?"
        assert state["evidence"] == []
        assert state["partial_answers"] == []
        assert state["errors"] == []
        assert "router" in state["node_status"]

    def test_state_with_security_context(self):
        """State should include security context."""
        ctx = SecurityContext(user_id="user1", roles=["admin"])
        state = create_initial_state(
            question="Query",
            security_context=ctx,
        )
        assert state["security_context"].user_id == "user1"
        assert "admin" in state["security_context"].roles

    def test_state_with_config(self):
        """State should use provided config."""
        config = OrchestratorConfig(
            default_k=10,
            default_search_mode="lexical",
        )
        state = create_initial_state(
            question="Query",
            config=config,
        )
        assert state["k"] == 10
        assert state["search_mode"] == "lexical"

    def test_state_with_overrides(self):
        """Explicit parameters should override config defaults."""
        config = OrchestratorConfig(default_k=5)
        state = create_initial_state(
            question="Query",
            config=config,
            k=20,
        )
        assert state["k"] == 20


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = OrchestratorConfig()
        assert config.enable_critic is False
        assert config.max_revisions == 1
        assert config.default_k == 5
        assert config.default_search_mode == "hybrid"
        assert config.fallback_on_error is True


class TestCustomExceptions:
    """Tests for custom exception classes."""

    def test_routing_error(self):
        """RoutingError should be an OrchestratorError."""
        with pytest.raises(RoutingError):
            raise RoutingError("Invalid route")

    def test_retrieval_error(self):
        """RetrievalError should include agent info."""
        error = RetrievalError("Failed", agent="docs", recoverable=True)
        assert error.agent == "docs"
        assert error.recoverable is True

    def test_synthesis_error(self):
        """SynthesisError should be an OrchestratorError."""
        with pytest.raises(SynthesisError):
            raise SynthesisError("Synthesis failed")


class TestRAGOrchestrator:
    """Tests for RAGOrchestrator class."""

    def test_init_with_rag_only(self):
        """Orchestrator should work with only RAG chain."""
        llm = MockLLMClient()
        rag = MockRAGChain(llm_client=llm)
        orchestrator = RAGOrchestrator(rag_chain=rag)
        assert orchestrator.rag_chain is rag
        assert orchestrator.sql_chain is None
        assert orchestrator.llm_client is llm

    def test_init_with_sql_only(self):
        """Orchestrator should work with only SQL chain."""
        llm = MockLLMClient()
        sql = MockSQLChain()
        orchestrator = RAGOrchestrator(sql_chain=sql, llm_client=llm)
        assert orchestrator.rag_chain is None
        assert orchestrator.sql_chain is sql

    def test_init_with_both(self):
        """Orchestrator should work with both chains."""
        llm = MockLLMClient()
        rag = MockRAGChain(llm_client=llm)
        sql = MockSQLChain()
        orchestrator = RAGOrchestrator(rag_chain=rag, sql_chain=sql)
        assert orchestrator.rag_chain is rag
        assert orchestrator.sql_chain is sql

    def test_query_document_routing(self):
        """Orchestrator should route document queries to RAG."""
        llm = MockLLMClient("Document response")
        rag = MockRAGChain(llm_client=llm)
        sql = MockSQLChain()
        orchestrator = RAGOrchestrator(rag_chain=rag, sql_chain=sql)

        state = orchestrator.query(
            question="What does the policy say about vacation?",
            force_route="documents",
        )

        assert state["route_decision"].query_type == QueryType.DOCUMENTS
        assert state["node_status"]["docs_agent"] == ExecutionStatus.SUCCESS.value
        # SQL agent is not called for document routes (stays pending or skipped)
        assert state["node_status"]["sql_agent"] in [
            ExecutionStatus.SKIPPED.value,
            ExecutionStatus.PENDING.value,
        ]
        assert "Document" in state["final_answer"] or len(state["final_answer"]) > 0

    def test_query_structured_routing(self):
        """Orchestrator should route SQL queries to SQL chain."""
        llm = MockLLMClient()
        rag = MockRAGChain(llm_client=llm)
        sql = MockSQLChain()
        orchestrator = RAGOrchestrator(rag_chain=rag, sql_chain=sql, llm_client=llm)

        state = orchestrator.query(
            question="How many rows in the sales table?",
            force_route="structured",
        )

        assert state["route_decision"].query_type == QueryType.STRUCTURED
        assert state["node_status"]["sql_agent"] == ExecutionStatus.SUCCESS.value
        # Docs agent is not called for structured routes (stays pending or skipped)
        assert state["node_status"]["docs_agent"] in [
            ExecutionStatus.SKIPPED.value,
            ExecutionStatus.PENDING.value,
        ]

    def test_query_hybrid_routing(self):
        """Orchestrator should run both agents for hybrid queries."""
        llm = MockLLMClient("Combined response")
        rag = MockRAGChain(llm_client=llm)
        sql = MockSQLChain()
        orchestrator = RAGOrchestrator(rag_chain=rag, sql_chain=sql, llm_client=llm)

        state = orchestrator.query(
            question="What does the report say about top products?",
            force_route="hybrid",
        )

        assert state["route_decision"].query_type == QueryType.HYBRID
        assert state["node_status"]["docs_agent"] == ExecutionStatus.SUCCESS.value
        assert state["node_status"]["sql_agent"] == ExecutionStatus.SUCCESS.value
        # Should have evidence from both sources
        sources = [e.source for e in state["evidence"]]
        assert "docs" in sources
        assert "sql" in sources

    def test_update_known_tables(self):
        """update_known_tables should update classifier."""
        llm = MockLLMClient()
        orchestrator = RAGOrchestrator(llm_client=llm)
        orchestrator.update_known_tables(["sales", "products"])
        assert orchestrator.classifier.config.known_tables == ["sales", "products"]

    def test_get_graph_visualization(self):
        """get_graph_visualization should return ASCII diagram."""
        llm = MockLLMClient()
        orchestrator = RAGOrchestrator(llm_client=llm)
        viz = orchestrator.get_graph_visualization()
        assert "[Router]" in viz
        assert "[Docs Agent]" in viz
        assert "[SQL Agent]" in viz
        assert "[Synthesizer]" in viz


class TestErrorHandling:
    """Tests for error handling in the orchestrator."""

    def test_docs_error_with_fallback(self):
        """With fallback enabled, docs error should not abort execution."""
        llm = MockLLMClient("Fallback answer")

        # Create a failing RAG chain
        rag = MockRAGChain(llm_client=llm)
        rag.retrieve = MagicMock(side_effect=RuntimeError("RAG failed"))

        sql = MockSQLChain()
        config = OrchestratorConfig(fallback_on_error=True)
        orchestrator = RAGOrchestrator(
            rag_chain=rag,
            sql_chain=sql,
            llm_client=llm,
            config=config,
        )

        state = orchestrator.query(
            question="Test query",
            force_route="hybrid",
        )

        # Docs failed but SQL should succeed
        assert state["node_status"]["docs_agent"] == ExecutionStatus.FAILED.value
        assert state["node_status"]["sql_agent"] == ExecutionStatus.SUCCESS.value
        assert len(state["errors"]) > 0
        assert state["errors"][0].node_name == "docs_agent"

    def test_error_traceability(self):
        """Errors should include full traceback."""
        llm = MockLLMClient()
        rag = MockRAGChain(llm_client=llm)
        rag.retrieve = MagicMock(side_effect=ValueError("Test error"))

        config = OrchestratorConfig(fallback_on_error=True)
        orchestrator = RAGOrchestrator(
            rag_chain=rag,
            llm_client=llm,
            config=config,
        )

        state = orchestrator.query(
            question="Test",
            force_route="documents",
        )

        assert len(state["errors"]) > 0
        error = state["errors"][0]
        assert "ValueError" in error.traceback_str
        assert "Test error" in error.message


class TestAPIAdapter:
    """Tests for the API adapter functions."""

    def test_graph_state_to_response_docs_only(self):
        """Adapter should convert docs-only state to response."""
        state = GraphState(
            question="Test question",
            security_context=None,
            config=OrchestratorConfig(),
            route_decision=RouteDecision(
                query_type=QueryType.DOCUMENTS,
                confidence=0.9,
                reasoning="Document keywords detected",
            ),
            force_route=None,
            k=5,
            search_mode="hybrid",
            max_rows=100,
            rerank=True,
            evidence=[
                Evidence(
                    source="docs",
                    chunks=["Test chunk content"],
                    metadatas=[{"doc_id": "doc1", "relative_path": "doc1.pdf", "page": 1}],
                    chunk_ids=["doc1::chunk-0000"],
                    distances=[0.1],
                ),
            ],
            partial_answers=[
                PartialAnswer(source="docs", answer="Test answer", confidence=0.8),
            ],
            final_answer="Final test answer",
            critique=None,
            errors=[],
            node_status={
                "router": "success",
                "docs_agent": "success",
                "sql_agent": "skipped",
                "synthesizer": "success",
            },
            retrieval_time_ms=50.0,
            generation_time_ms=100.0,
            routing_time_ms=5.0,
        )

        response = graph_state_to_response(state)

        assert response.answer == "Final test answer"
        assert len(response.sources) == 1
        assert response.sources[0].doc_id == "doc1"
        assert response.sql_result is None
        assert response.metadata.routing.query_type == "documents"
        assert response.metadata.routing.confidence == 0.9
        assert response.metadata.reranking_applied is True

    def test_graph_state_to_response_sql_only(self):
        """Adapter should convert SQL-only state to response."""
        state = GraphState(
            question="Test question",
            security_context=None,
            config=OrchestratorConfig(),
            route_decision=RouteDecision(
                query_type=QueryType.STRUCTURED,
                confidence=0.85,
                reasoning="SQL keywords detected",
            ),
            force_route=None,
            k=5,
            search_mode="hybrid",
            max_rows=100,
            rerank=None,
            evidence=[
                Evidence(
                    source="sql",
                    sql_data=[{"col1": "value1"}],
                    sql_columns=["col1"],
                    sql_query="SELECT * FROM test",
                    tables_used=["test"],
                ),
            ],
            partial_answers=[
                PartialAnswer(source="sql", answer="SQL answer", confidence=0.8),
            ],
            final_answer="SQL final answer",
            critique=None,
            errors=[],
            node_status={},
            retrieval_time_ms=10.0,
            generation_time_ms=50.0,
            routing_time_ms=2.0,
        )

        response = graph_state_to_response(state)

        assert response.answer == "SQL final answer"
        assert len(response.sources) == 0
        assert response.sql_result is not None
        assert response.sql_result.row_count == 1
        assert response.sql_result.generated_sql == "SELECT * FROM test"
        assert response.metadata.tables_used == ["test"]

    def test_extract_errors_from_state(self):
        """extract_errors_from_state should return error dicts."""
        state = GraphState(
            question="",
            errors=[
                NodeError(
                    node_name="router",
                    error_type="RoutingError",
                    message="Failed",
                    traceback_str="tb...",
                    recoverable=False,
                ),
            ],
            node_status={},
        )
        errors = extract_errors_from_state(state)
        assert len(errors) == 1
        assert errors[0]["node_name"] == "router"

    def test_has_fatal_errors(self):
        """has_fatal_errors should detect non-recoverable errors."""
        state_ok = GraphState(
            question="",
            errors=[
                NodeError("node", "Error", "msg", "tb", recoverable=True),
            ],
            node_status={},
        )
        assert has_fatal_errors(state_ok) is False

        state_fatal = GraphState(
            question="",
            errors=[
                NodeError("node", "Error", "msg", "tb", recoverable=False),
            ],
            node_status={},
        )
        assert has_fatal_errors(state_fatal) is True

    def test_get_node_status_summary(self):
        """get_node_status_summary should return status dict."""
        state = GraphState(
            question="",
            node_status={
                "router": "success",
                "docs_agent": "success",
            },
            errors=[],
        )
        summary = get_node_status_summary(state)
        assert summary["router"] == "success"
        assert summary["docs_agent"] == "success"
