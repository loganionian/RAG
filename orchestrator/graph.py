"""LangGraph-based RAG orchestrator.

This module provides the RAGOrchestrator class that uses LangGraph
to orchestrate the flow between routing, retrieval agents, and synthesis.
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import TYPE_CHECKING, Optional

from langgraph.graph import END, StateGraph

from router.classifier import QuestionClassifier
from router.config import RouterConfig

from .config import OrchestratorConfig
from .nodes import (
    critic_node,
    docs_agent_node,
    get_next_after_critic,
    get_next_after_docs,
    get_next_after_router,
    get_next_after_sql,
    get_next_after_synthesizer,
    router_node,
    sql_agent_node,
    synthesizer_node,
)
from .state import GraphState, StateValidator, create_initial_state

if TYPE_CHECKING:
    from generation.base import BaseLLMClient
    from generation.rag_chain import RAGChain
    from security.acl_filter import ACLFilter
    from security.audit_logger import AuditLogger
    from security.config import SecurityContext
    from security.table_acl import TableACL
    from sql_agent.sql_chain import SQLChain

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """LangGraph-based RAG orchestrator.

    Orchestrates the flow between:
    - Router: Classifies questions and determines routing
    - Docs Agent: Retrieves documents and generates partial answers
    - SQL Agent: Executes SQL queries and generates partial answers
    - Synthesizer: Combines partial answers into final response
    - Critic: Evaluates answer quality (placeholder)

    The graph topology adapts based on the routing decision:
    - DOCUMENTS: Router -> Docs Agent -> Synthesizer -> [Critic] -> END
    - STRUCTURED: Router -> SQL Agent -> Synthesizer -> [Critic] -> END
    - HYBRID: Router -> Docs Agent -> SQL Agent -> Synthesizer -> [Critic] -> END
    """

    def __init__(
        self,
        rag_chain: Optional["RAGChain"] = None,
        sql_chain: Optional["SQLChain"] = None,
        classifier: Optional[QuestionClassifier] = None,
        llm_client: Optional["BaseLLMClient"] = None,
        acl_filter: Optional["ACLFilter"] = None,
        table_acl: Optional["TableACL"] = None,
        audit_logger: Optional["AuditLogger"] = None,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            rag_chain: RAG chain for document retrieval. Required for document queries.
            sql_chain: SQL chain for structured queries. Required for SQL queries.
            classifier: Question classifier. Created with defaults if not provided.
            llm_client: LLM client for synthesis. Falls back to rag_chain's client if not provided.
            acl_filter: ACL filter for document access control.
            table_acl: Table ACL for SQL access control.
            audit_logger: Audit logger for recording decisions.
            config: Orchestrator configuration.
        """
        self.rag_chain = rag_chain
        self.sql_chain = sql_chain
        self.classifier = classifier or QuestionClassifier(RouterConfig())
        self.llm_client = llm_client or (rag_chain.llm_client if rag_chain else None)
        self.acl_filter = acl_filter
        self.table_acl = table_acl
        self.audit_logger = audit_logger
        self.config = config or OrchestratorConfig()
        self.validator = StateValidator(strict=False)

        # Build the graph
        self._graph = self._build_graph()
        self._compiled_graph = self._graph.compile()

    def _wrap_with_validation(
        self,
        node_name: str,
        node_fn: callable,
    ) -> callable:
        """Wrap a node function with entry/exit validation.

        Validation failures are logged as warnings and add recoverable
        errors to the state, but do not abort execution.

        Args:
            node_name: Name of the node for validation context.
            node_fn: The node function to wrap.

        Returns:
            Wrapped function that performs validation.
        """
        def validated_node(state: GraphState) -> dict:
            # Entry validation
            entry_result = self.validator.validate_entry(node_name, state)
            if not entry_result.passed:
                logger.warning("Validation failed: %s", entry_result.message)
                if not entry_result.recoverable:
                    # Non-recoverable - add error and return without executing
                    from .state import NodeError
                    error = entry_result.to_node_error(node_name)
                    return {
                        "errors": state.get("errors", []) + [error],
                    }

            # Execute the node
            result = node_fn(state)

            # Merge result into state for exit validation
            merged_state = {**state, **result}

            # Exit validation
            exit_result = self.validator.validate_exit(node_name, merged_state)
            if not exit_result.passed:
                logger.warning("Validation failed: %s", exit_result.message)
                # Add validation error to result
                from .state import NodeError
                error = exit_result.to_node_error(node_name)
                existing_errors = result.get("errors", state.get("errors", []))
                result["errors"] = existing_errors + [error]

            return result

        return validated_node

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph.

        Returns:
            Configured StateGraph ready for compilation.
        """
        # Create the graph
        graph = StateGraph(GraphState)

        # Create bound node functions with dependencies
        router_fn = partial(
            router_node,
            classifier=self.classifier,
            audit_logger=self.audit_logger,
        )

        docs_fn = partial(
            docs_agent_node,
            rag_chain=self.rag_chain,
            acl_filter=self.acl_filter,
        ) if self.rag_chain else self._skip_docs_node

        sql_fn = partial(
            sql_agent_node,
            sql_chain=self.sql_chain,
            table_acl=self.table_acl,
        ) if self.sql_chain else self._skip_sql_node

        synth_fn = partial(
            synthesizer_node,
            llm_client=self.llm_client,
        )

        # Wrap node functions with validation
        router_fn = self._wrap_with_validation("router", router_fn)
        docs_fn = self._wrap_with_validation("docs_agent", docs_fn)
        sql_fn = self._wrap_with_validation("sql_agent", sql_fn)
        synth_fn = self._wrap_with_validation("synthesizer", synth_fn)

        # Add nodes
        graph.add_node("router", router_fn)
        graph.add_node("docs_agent", docs_fn)
        graph.add_node("sql_agent", sql_fn)
        graph.add_node("synthesizer", synth_fn)
        graph.add_node("critic", critic_node)

        # Set entry point
        graph.set_entry_point("router")

        # Add conditional edges from router
        graph.add_conditional_edges(
            "router",
            get_next_after_router,
            {
                "docs_agent": "docs_agent",
                "sql_agent": "sql_agent",
                "synthesizer": "synthesizer",
            },
        )

        # Add conditional edges from docs_agent
        graph.add_conditional_edges(
            "docs_agent",
            get_next_after_docs,
            {
                "sql_agent": "sql_agent",
                "synthesizer": "synthesizer",
            },
        )

        # Add edge from sql_agent to synthesizer
        graph.add_conditional_edges(
            "sql_agent",
            get_next_after_sql,
            {
                "synthesizer": "synthesizer",
            },
        )

        # Add conditional edges from synthesizer
        graph.add_conditional_edges(
            "synthesizer",
            get_next_after_synthesizer,
            {
                "critic": "critic",
                "__end__": END,
            },
        )

        # Add conditional edges from critic
        graph.add_conditional_edges(
            "critic",
            get_next_after_critic,
            {
                "__end__": END,
            },
        )

        return graph

    @staticmethod
    def _skip_docs_node(state: GraphState) -> dict:
        """Placeholder node when RAG chain is not available."""
        from .state import ExecutionStatus

        logger.warning("Docs agent: RAG chain not configured, skipping")
        return {
            "node_status": {
                **state.get("node_status", {}),
                "docs_agent": ExecutionStatus.SKIPPED.value,
            },
        }

    @staticmethod
    def _skip_sql_node(state: GraphState) -> dict:
        """Placeholder node when SQL chain is not available."""
        from .state import ExecutionStatus

        logger.warning("SQL agent: SQL chain not configured, skipping")
        return {
            "node_status": {
                **state.get("node_status", {}),
                "sql_agent": ExecutionStatus.SKIPPED.value,
            },
        }

    def query(
        self,
        question: str,
        security_context: Optional["SecurityContext"] = None,
        k: Optional[int] = None,
        search_mode: Optional[str] = None,
        force_route: Optional[str] = None,
        rerank: Optional[bool] = None,
        max_rows: Optional[int] = None,
    ) -> GraphState:
        """Execute a query through the orchestrator graph.

        Synchronous execution method.

        Args:
            question: The user's question.
            security_context: Optional security context for ACL.
            k: Number of chunks to retrieve.
            search_mode: Search mode for document queries.
            force_route: Force routing to specific type.
            rerank: Whether to apply reranking.
            max_rows: Maximum rows for SQL queries.

        Returns:
            Final GraphState with all results.
        """
        # Create initial state
        initial_state = create_initial_state(
            question=question,
            security_context=security_context,
            config=self.config,
            k=k,
            search_mode=search_mode,
            max_rows=max_rows,
            rerank=rerank,
            force_route=force_route,
        )

        # Execute the graph
        logger.info("Orchestrator: starting query execution")
        final_state = self._compiled_graph.invoke(initial_state)
        logger.info("Orchestrator: query execution complete")

        return final_state

    async def aquery(
        self,
        question: str,
        security_context: Optional["SecurityContext"] = None,
        k: Optional[int] = None,
        search_mode: Optional[str] = None,
        force_route: Optional[str] = None,
        rerank: Optional[bool] = None,
        max_rows: Optional[int] = None,
    ) -> GraphState:
        """Execute a query through the orchestrator graph asynchronously.

        Async execution method for FastAPI integration.

        Args:
            question: The user's question.
            security_context: Optional security context for ACL.
            k: Number of chunks to retrieve.
            search_mode: Search mode for document queries.
            force_route: Force routing to specific type.
            rerank: Whether to apply reranking.
            max_rows: Maximum rows for SQL queries.

        Returns:
            Final GraphState with all results.
        """
        # Create initial state
        initial_state = create_initial_state(
            question=question,
            security_context=security_context,
            config=self.config,
            k=k,
            search_mode=search_mode,
            max_rows=max_rows,
            rerank=rerank,
            force_route=force_route,
        )

        # Execute the graph asynchronously
        # LangGraph's ainvoke runs the graph in an async context
        logger.info("Orchestrator: starting async query execution")

        # Run synchronous graph in executor to avoid blocking
        loop = asyncio.get_event_loop()
        final_state = await loop.run_in_executor(
            None,
            self._compiled_graph.invoke,
            initial_state,
        )

        logger.info("Orchestrator: async query execution complete")

        return final_state

    def update_known_tables(self, tables: list[str]) -> None:
        """Update the classifier's known tables.

        Call this when tables change to improve routing accuracy.

        Args:
            tables: List of available table names.
        """
        self.classifier.set_known_tables(tables)
        logger.debug("Orchestrator: updated known tables (%d tables)", len(tables))

    def get_graph_visualization(self) -> str:
        """Get a text representation of the graph topology.

        Returns:
            ASCII diagram of the graph.
        """
        return """
    [START]
       │
       ▼
    [Router]
       │
       ├─── DOCUMENTS ───► [Docs Agent] ───┐
       │                                   │
       ├─── STRUCTURED ──► [SQL Agent] ────┤
       │                                   │
       └─── HYBRID ──────► [Docs Agent] ───┤
                          [SQL Agent] ─────┤
                                           ▼
                                    [Synthesizer]
                                           │
                                           ▼
                                    [Critic] (optional)
                                           │
                                        [END]
        """
