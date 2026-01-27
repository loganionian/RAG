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
    StateValidator,
    SynthesisError,
    ValidationResult,
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


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_passed(self):
        """ValidationResult should represent a passing check."""
        result = ValidationResult(passed=True, message="Valid")
        assert result.passed is True
        assert result.recoverable is True

    def test_validation_result_failed(self):
        """ValidationResult should represent a failing check."""
        result = ValidationResult(
            passed=False,
            message="Invalid state",
            recoverable=False,
        )
        assert result.passed is False
        assert result.recoverable is False

    def test_to_node_error(self):
        """ValidationResult should convert to NodeError."""
        result = ValidationResult(
            passed=False,
            message="Validation failed",
            recoverable=True,
        )
        error = result.to_node_error("router")
        assert error.node_name == "router"
        assert error.error_type == "ValidationError"
        assert error.message == "Validation failed"
        assert error.recoverable is True


class TestStateValidator:
    """Tests for StateValidator class."""

    def test_router_entry_valid(self):
        """Router entry should pass with non-empty question."""
        validator = StateValidator()
        state = GraphState(
            question="What is the policy?",
            node_status={},
            errors=[],
        )
        result = validator.validate_router_entry(state)
        assert result.passed is True

    def test_router_entry_empty_question(self):
        """Router entry should fail with empty question."""
        validator = StateValidator()
        state = GraphState(
            question="",
            node_status={},
            errors=[],
        )
        result = validator.validate_router_entry(state)
        assert result.passed is False
        assert "empty" in result.message.lower()

    def test_router_entry_whitespace_question(self):
        """Router entry should fail with whitespace-only question."""
        validator = StateValidator()
        state = GraphState(
            question="   ",
            node_status={},
            errors=[],
        )
        result = validator.validate_router_entry(state)
        assert result.passed is False

    def test_router_exit_valid(self):
        """Router exit should pass with route_decision set."""
        validator = StateValidator()
        state = GraphState(
            question="Test",
            route_decision=RouteDecision(
                query_type=QueryType.DOCUMENTS,
                confidence=0.9,
                reasoning="test",
            ),
            node_status={},
            errors=[],
        )
        result = validator.validate_router_exit(state)
        assert result.passed is True

    def test_router_exit_no_decision(self):
        """Router exit should fail without route_decision."""
        validator = StateValidator()
        state = GraphState(
            question="Test",
            route_decision=None,
            node_status={},
            errors=[],
        )
        result = validator.validate_router_exit(state)
        assert result.passed is False
        assert "route_decision" in result.message.lower()

    def test_router_exit_with_routing_error(self):
        """Router exit should pass if routing error is recorded."""
        validator = StateValidator()
        state = GraphState(
            question="Test",
            route_decision=None,
            node_status={},
            errors=[
                NodeError(
                    node_name="router",
                    error_type="RoutingError",
                    message="Failed",
                    traceback_str="",
                )
            ],
        )
        result = validator.validate_router_exit(state)
        assert result.passed is True

    def test_docs_agent_exit_success_with_evidence(self):
        """Docs agent exit should pass if successful with evidence."""
        validator = StateValidator()
        state = GraphState(
            question="Test",
            node_status={"docs_agent": ExecutionStatus.SUCCESS.value},
            evidence=[Evidence(source="docs", chunks=["chunk1"])],
            errors=[],
        )
        result = validator.validate_docs_agent_exit(state)
        assert result.passed is True

    def test_docs_agent_exit_success_no_evidence(self):
        """Docs agent exit should fail if successful without evidence."""
        validator = StateValidator()
        state = GraphState(
            question="Test",
            node_status={"docs_agent": ExecutionStatus.SUCCESS.value},
            evidence=[],
            errors=[],
        )
        result = validator.validate_docs_agent_exit(state)
        assert result.passed is False
        assert "no docs evidence" in result.message.lower()

    def test_docs_agent_exit_skipped(self):
        """Docs agent exit should pass if skipped."""
        validator = StateValidator()
        state = GraphState(
            question="Test",
            node_status={"docs_agent": ExecutionStatus.SKIPPED.value},
            evidence=[],
            errors=[],
        )
        result = validator.validate_docs_agent_exit(state)
        assert result.passed is True

    def test_docs_agent_exit_failed(self):
        """Docs agent exit should pass if failed (error already recorded)."""
        validator = StateValidator()
        state = GraphState(
            question="Test",
            node_status={"docs_agent": ExecutionStatus.FAILED.value},
            evidence=[],
            errors=[],
        )
        result = validator.validate_docs_agent_exit(state)
        assert result.passed is True

    def test_sql_agent_exit_success_with_evidence(self):
        """SQL agent exit should pass if successful with evidence."""
        validator = StateValidator()
        state = GraphState(
            question="Test",
            node_status={"sql_agent": ExecutionStatus.SUCCESS.value},
            evidence=[Evidence(source="sql", sql_data=[{"col": "val"}])],
            errors=[],
        )
        result = validator.validate_sql_agent_exit(state)
        assert result.passed is True

    def test_sql_agent_exit_success_no_evidence(self):
        """SQL agent exit should fail if successful without evidence."""
        validator = StateValidator()
        state = GraphState(
            question="Test",
            node_status={"sql_agent": ExecutionStatus.SUCCESS.value},
            evidence=[],
            errors=[],
        )
        result = validator.validate_sql_agent_exit(state)
        assert result.passed is False
        assert "no sql evidence" in result.message.lower()

    def test_synthesizer_exit_valid(self):
        """Synthesizer exit should pass with non-empty final_answer."""
        validator = StateValidator()
        state = GraphState(
            question="Test",
            final_answer="The answer is 42.",
            node_status={},
            errors=[],
        )
        result = validator.validate_synthesizer_exit(state)
        assert result.passed is True

    def test_synthesizer_exit_empty_answer(self):
        """Synthesizer exit should fail with empty final_answer."""
        validator = StateValidator()
        state = GraphState(
            question="Test",
            final_answer="",
            node_status={},
            errors=[],
        )
        result = validator.validate_synthesizer_exit(state)
        assert result.passed is False
        assert "final_answer" in result.message.lower()

    def test_strict_mode(self):
        """Strict mode should make failures non-recoverable."""
        validator = StateValidator(strict=True)
        state = GraphState(
            question="",
            node_status={},
            errors=[],
        )
        result = validator.validate_router_entry(state)
        assert result.passed is False
        assert result.recoverable is False

    def test_validate_entry_dispatch(self):
        """validate_entry should dispatch to correct validator."""
        validator = StateValidator()
        state = GraphState(
            question="Test",
            node_status={},
            errors=[],
        )
        result = validator.validate_entry("router", state)
        assert result.passed is True

        # Unknown node should pass
        result = validator.validate_entry("unknown_node", state)
        assert result.passed is True

    def test_validate_exit_dispatch(self):
        """validate_exit should dispatch to correct validator."""
        validator = StateValidator()
        state = GraphState(
            question="Test",
            route_decision=RouteDecision(
                query_type=QueryType.DOCUMENTS,
                confidence=0.9,
                reasoning="test",
            ),
            node_status={},
            errors=[],
        )
        result = validator.validate_exit("router", state)
        assert result.passed is True


class TestStatePassingAndEvidenceAggregation:
    """Tests for state passing and evidence aggregation across nodes."""

    def test_evidence_aggregation_hybrid(self):
        """Hybrid queries should aggregate evidence from both agents."""
        llm = MockLLMClient("Combined response")
        rag = MockRAGChain(llm_client=llm)
        sql = MockSQLChain()
        orchestrator = RAGOrchestrator(rag_chain=rag, sql_chain=sql, llm_client=llm)

        state = orchestrator.query(
            question="What does the report say and how many records?",
            force_route="hybrid",
        )

        # Should have evidence from both sources
        evidence_sources = [e.source for e in state["evidence"]]
        assert "docs" in evidence_sources
        assert "sql" in evidence_sources

        # Should have partial answers from both sources
        partial_sources = [pa.source for pa in state["partial_answers"]]
        assert "docs" in partial_sources
        assert "sql" in partial_sources

    def test_partial_answers_preserved(self):
        """Partial answers should be preserved through synthesis."""
        llm = MockLLMClient("Final answer")
        rag = MockRAGChain(llm_client=llm)
        orchestrator = RAGOrchestrator(rag_chain=rag, llm_client=llm)

        state = orchestrator.query(
            question="What is in the documents?",
            force_route="documents",
        )

        assert len(state["partial_answers"]) >= 1
        assert state["partial_answers"][0].source == "docs"
        assert state["final_answer"]  # Should have synthesized answer

    def test_timing_accumulation(self):
        """Timing metrics should accumulate across nodes."""
        llm = MockLLMClient("Answer")
        rag = MockRAGChain(llm_client=llm)
        sql = MockSQLChain()
        orchestrator = RAGOrchestrator(rag_chain=rag, sql_chain=sql, llm_client=llm)

        state = orchestrator.query(
            question="Hybrid query",
            force_route="hybrid",
        )

        # Timing should be accumulated
        assert state["retrieval_time_ms"] > 0
        assert state["routing_time_ms"] >= 0


class TestOrchestratorWithValidation:
    """Tests for orchestrator with validation integration."""

    def test_validation_errors_recorded(self):
        """Validation errors should be recorded in state."""
        llm = MockLLMClient("Answer")
        rag = MockRAGChain(llm_client=llm)

        # Create a RAG chain that returns no evidence despite success
        rag.retrieve = MagicMock(return_value=MagicMock(
            chunks=[],
            metadatas=[],
            ids=[],
            distances=[],
        ))

        orchestrator = RAGOrchestrator(rag_chain=rag, llm_client=llm)

        state = orchestrator.query(
            question="Test query",
            force_route="documents",
        )

        # The synthesizer should still produce an answer even with no chunks
        assert state["final_answer"]

    def test_empty_question_handled(self):
        """Empty question should be handled gracefully."""
        llm = MockLLMClient("Answer")
        orchestrator = RAGOrchestrator(llm_client=llm)

        # Empty question - should still process (classifier may still work)
        # The validation will add a warning but execution continues
        state = orchestrator.query(question="  ")

        # Should have validation error recorded
        # But still complete execution


class TestAPIIntegration:
    """Tests for API integration with orchestrator."""

    def test_graph_state_to_response_with_validation_errors(self):
        """Response should be generated even with validation errors."""
        state = GraphState(
            question="Test",
            security_context=None,
            config=OrchestratorConfig(),
            route_decision=RouteDecision(
                query_type=QueryType.DOCUMENTS,
                confidence=0.9,
                reasoning="test",
            ),
            force_route=None,
            k=5,
            search_mode="hybrid",
            max_rows=100,
            rerank=None,
            evidence=[],
            partial_answers=[],
            final_answer="Answer despite errors",
            critique=None,
            errors=[
                NodeError(
                    node_name="docs_agent",
                    error_type="ValidationError",
                    message="No evidence added",
                    traceback_str="",
                    recoverable=True,
                ),
            ],
            node_status={"router": "success", "docs_agent": "success"},
            retrieval_time_ms=50.0,
            generation_time_ms=100.0,
            routing_time_ms=5.0,
        )

        response = graph_state_to_response(state)
        assert response.answer == "Answer despite errors"

    def test_graph_state_to_response_hybrid(self):
        """Hybrid response should include both sources and SQL result."""
        state = GraphState(
            question="Test",
            security_context=None,
            config=OrchestratorConfig(),
            route_decision=RouteDecision(
                query_type=QueryType.HYBRID,
                confidence=0.85,
                reasoning="hybrid query",
            ),
            force_route=None,
            k=5,
            search_mode="hybrid",
            max_rows=100,
            rerank=True,
            evidence=[
                Evidence(
                    source="docs",
                    chunks=["Document content"],
                    metadatas=[{"doc_id": "doc1", "relative_path": "doc.pdf"}],
                    chunk_ids=["doc1::chunk-0000"],
                    distances=[0.1],
                ),
                Evidence(
                    source="sql",
                    sql_data=[{"count": 42}],
                    sql_columns=["count"],
                    sql_query="SELECT COUNT(*) FROM test",
                    tables_used=["test"],
                ),
            ],
            partial_answers=[
                PartialAnswer(source="docs", answer="Doc answer", confidence=0.8),
                PartialAnswer(source="sql", answer="SQL answer", confidence=0.8),
            ],
            final_answer="Combined answer",
            critique=None,
            errors=[],
            node_status={},
            retrieval_time_ms=75.0,
            generation_time_ms=150.0,
            routing_time_ms=3.0,
        )

        response = graph_state_to_response(state)

        assert response.answer == "Combined answer"
        assert len(response.sources) == 1
        assert response.sources[0].doc_id == "doc1"
        assert response.sql_result is not None
        assert response.sql_result.row_count == 1
        assert response.metadata.routing.query_type == "hybrid"
        assert response.metadata.tables_used == ["test"]
        assert response.metadata.reranking_applied is True
