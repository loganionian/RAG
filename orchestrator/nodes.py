"""Node functions for the orchestrator graph.

This module defines the node functions that make up the LangGraph state machine.
Each node handles a specific responsibility in the orchestration flow.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

from router.models import QueryType, RouteDecision

from .errors import RetrievalError, RoutingError, SynthesisError
from .state import (
    CritiqueResult,
    Evidence,
    ExecutionStatus,
    GraphState,
    NodeError,
    PartialAnswer,
)

if TYPE_CHECKING:
    from generation.base import BaseLLMClient
    from generation.rag_chain import RAGChain
    from router.classifier import QuestionClassifier
    from security.acl_filter import ACLFilter
    from security.audit_logger import AuditLogger
    from security.table_acl import TableACL
    from sql_agent.sql_chain import SQLChain

logger = logging.getLogger(__name__)

# Synthesis prompt for combining multiple answers
SYNTHESIS_PROMPT = """You are a helpful assistant that synthesizes information from multiple sources.

The user asked: {question}

You have answers from the following sources:

{source_answers}

Synthesize these into a single, coherent response. If the sources provide different types of information (e.g., document context vs. data statistics), present both clearly labeled. If sources contradict each other, note the discrepancy.

Provide a clear, well-organized response:"""


def router_node(
    state: GraphState,
    classifier: "QuestionClassifier",
    audit_logger: Optional["AuditLogger"] = None,
) -> dict:
    """Classify the question and determine routing.

    Args:
        state: Current graph state.
        classifier: Question classifier instance.
        audit_logger: Optional audit logger for recording decisions.

    Returns:
        State updates with routing decision.
    """
    start_time = time.perf_counter()

    try:
        # Check for forced routing
        if state.get("force_route"):
            valid_routes = {"documents", "structured", "hybrid"}
            force_route = state["force_route"]

            if force_route not in valid_routes:
                raise RoutingError(
                    f"Invalid force_route: {force_route}. Must be one of: {valid_routes}"
                )

            query_type = QueryType(force_route)
            route_decision = RouteDecision(
                query_type=query_type,
                confidence=1.0,
                reasoning=f"Forced to {force_route}",
                signals={},
            )
        else:
            # Classify the question
            route_decision = classifier.classify(state["question"])

        routing_time_ms = (time.perf_counter() - start_time) * 1000

        # Log the decision
        if audit_logger:
            security_context = state.get("security_context")
            if security_context:
                audit_logger.log_route_decision(
                    security_context=security_context,
                    query_type=route_decision.query_type.value,
                    confidence=route_decision.confidence,
                    reasoning=route_decision.reasoning,
                    question=state["question"],
                    force_route=state.get("force_route"),
                )

        logger.info(
            "Router: classified as %s (confidence=%.2f)",
            route_decision.query_type.value,
            route_decision.confidence,
        )

        return {
            "route_decision": route_decision,
            "routing_time_ms": routing_time_ms,
            "node_status": {
                **state.get("node_status", {}),
                "router": ExecutionStatus.SUCCESS.value,
            },
        }

    except Exception as e:
        logger.exception("Router node failed")
        error = NodeError.from_exception("router", e, recoverable=False)
        return {
            "errors": state.get("errors", []) + [error],
            "node_status": {
                **state.get("node_status", {}),
                "router": ExecutionStatus.FAILED.value,
            },
        }


def docs_agent_node(
    state: GraphState,
    rag_chain: "RAGChain",
    acl_filter: Optional["ACLFilter"] = None,
) -> dict:
    """Retrieve documents and generate a partial answer.

    Args:
        state: Current graph state.
        rag_chain: RAG chain for retrieval and generation.
        acl_filter: Optional ACL filter for access control.

    Returns:
        State updates with evidence and partial answer.
    """
    start_time = time.perf_counter()

    # Check if we should skip this node
    route_decision = state.get("route_decision")
    if route_decision and route_decision.query_type == QueryType.STRUCTURED:
        logger.debug("Docs agent: skipping (STRUCTURED query)")
        return {
            "node_status": {
                **state.get("node_status", {}),
                "docs_agent": ExecutionStatus.SKIPPED.value,
            },
        }

    try:
        # Retrieve documents
        security_context = state.get("security_context")
        k = state.get("k", 5)
        search_mode = state.get("search_mode", "hybrid")
        rerank = state.get("rerank")

        retrieval_result = rag_chain.retrieve(
            query=state["question"],
            k=k,
            mode=search_mode,
            rerank=rerank,
            security_context=security_context,
            acl_filter=acl_filter,
        )

        retrieval_time = (time.perf_counter() - start_time) * 1000

        # Build evidence
        evidence = Evidence(
            source="docs",
            chunks=retrieval_result.chunks,
            metadatas=retrieval_result.metadatas,
            chunk_ids=retrieval_result.ids,
            distances=retrieval_result.distances,
        )

        # Generate partial answer if we have chunks
        partial_answer = None
        generation_time = 0.0

        if retrieval_result.chunks:
            gen_start = time.perf_counter()
            context = rag_chain._format_context(retrieval_result)
            system_prompt = rag_chain.config.system_prompt.format(context=context)
            answer_text = rag_chain.llm_client.generate(
                prompt=state["question"],
                system_prompt=system_prompt,
                max_tokens=rag_chain.config.max_tokens,
                temperature=rag_chain.config.temperature,
            )
            generation_time = (time.perf_counter() - gen_start) * 1000

            partial_answer = PartialAnswer(
                source="docs",
                answer=answer_text,
                confidence=0.8 if route_decision and route_decision.query_type == QueryType.DOCUMENTS else 0.6,
                timing_ms=retrieval_time + generation_time,
            )
        else:
            partial_answer = PartialAnswer(
                source="docs",
                answer="No relevant documents found.",
                confidence=0.3,
                timing_ms=retrieval_time,
            )

        logger.info(
            "Docs agent: retrieved %d chunks in %.1fms",
            len(retrieval_result.chunks),
            retrieval_time,
        )

        return {
            "evidence": state.get("evidence", []) + [evidence],
            "partial_answers": state.get("partial_answers", []) + [partial_answer],
            "retrieval_time_ms": state.get("retrieval_time_ms", 0.0) + retrieval_time,
            "generation_time_ms": state.get("generation_time_ms", 0.0) + generation_time,
            "node_status": {
                **state.get("node_status", {}),
                "docs_agent": ExecutionStatus.SUCCESS.value,
            },
        }

    except Exception as e:
        logger.exception("Docs agent failed")
        error = NodeError.from_exception("docs_agent", e, recoverable=True)

        # Check if we should continue despite error
        config = state.get("config")
        if config and config.fallback_on_error:
            return {
                "errors": state.get("errors", []) + [error],
                "node_status": {
                    **state.get("node_status", {}),
                    "docs_agent": ExecutionStatus.FAILED.value,
                },
            }
        else:
            raise RetrievalError(str(e), agent="docs", recoverable=False) from e


def sql_agent_node(
    state: GraphState,
    sql_chain: "SQLChain",
    table_acl: Optional["TableACL"] = None,
) -> dict:
    """Execute SQL query and generate a partial answer.

    Args:
        state: Current graph state.
        sql_chain: SQL chain for query execution.
        table_acl: Optional table ACL for access control.

    Returns:
        State updates with evidence and partial answer.
    """
    start_time = time.perf_counter()

    # Check if we should skip this node
    route_decision = state.get("route_decision")
    if route_decision and route_decision.query_type == QueryType.DOCUMENTS:
        logger.debug("SQL agent: skipping (DOCUMENTS query)")
        return {
            "node_status": {
                **state.get("node_status", {}),
                "sql_agent": ExecutionStatus.SKIPPED.value,
            },
        }

    try:
        security_context = state.get("security_context")
        max_rows = state.get("max_rows", 100)

        # Execute SQL query
        result = sql_chain.query(
            question=state["question"],
            max_rows=max_rows,
            summarize=True,
            security_context=security_context,
            table_acl=table_acl,
        )

        total_time = (time.perf_counter() - start_time) * 1000

        # Build evidence
        evidence = Evidence(
            source="sql",
            sql_data=result.data,
            sql_columns=result.columns,
            sql_query=result.generated_sql,
            tables_used=result.tables_used,
        )

        # Build partial answer
        partial_answer = PartialAnswer(
            source="sql",
            answer=result.answer,
            confidence=0.8 if route_decision and route_decision.query_type == QueryType.STRUCTURED else 0.6,
            timing_ms=total_time,
        )

        logger.info(
            "SQL agent: returned %d rows in %.1fms",
            result.row_count,
            total_time,
        )

        return {
            "evidence": state.get("evidence", []) + [evidence],
            "partial_answers": state.get("partial_answers", []) + [partial_answer],
            "retrieval_time_ms": state.get("retrieval_time_ms", 0.0) + result.execution_time_ms,
            "generation_time_ms": state.get("generation_time_ms", 0.0) + result.generation_time_ms + result.summarization_time_ms,
            "node_status": {
                **state.get("node_status", {}),
                "sql_agent": ExecutionStatus.SUCCESS.value,
            },
        }

    except Exception as e:
        logger.exception("SQL agent failed")
        error = NodeError.from_exception("sql_agent", e, recoverable=True)

        # Check if we should continue despite error
        config = state.get("config")
        if config and config.fallback_on_error:
            return {
                "errors": state.get("errors", []) + [error],
                "node_status": {
                    **state.get("node_status", {}),
                    "sql_agent": ExecutionStatus.FAILED.value,
                },
            }
        else:
            raise RetrievalError(str(e), agent="sql", recoverable=False) from e


def synthesizer_node(
    state: GraphState,
    llm_client: "BaseLLMClient",
) -> dict:
    """Synthesize partial answers into a final response.

    Args:
        state: Current graph state.
        llm_client: LLM client for synthesis.

    Returns:
        State updates with final answer.
    """
    start_time = time.perf_counter()

    try:
        partial_answers = state.get("partial_answers", [])

        if not partial_answers:
            # No answers to synthesize
            logger.warning("Synthesizer: no partial answers available")
            return {
                "final_answer": "I couldn't find relevant information to answer your question.",
                "node_status": {
                    **state.get("node_status", {}),
                    "synthesizer": ExecutionStatus.SUCCESS.value,
                },
            }

        # Filter to successful answers (exclude "no results" type answers)
        successful_answers = [
            pa for pa in partial_answers
            if pa.confidence > 0.3 and "no relevant" not in pa.answer.lower() and "no results" not in pa.answer.lower()
        ]

        if not successful_answers:
            # All answers indicate no results
            return {
                "final_answer": "I couldn't find relevant information from documents or data.",
                "node_status": {
                    **state.get("node_status", {}),
                    "synthesizer": ExecutionStatus.SUCCESS.value,
                },
            }

        if len(successful_answers) == 1:
            # Single source - no synthesis needed
            final_answer = successful_answers[0].answer
        else:
            # Multiple sources - synthesize
            source_texts = []
            for pa in successful_answers:
                source_label = "Documents" if pa.source == "docs" else "Data/SQL"
                source_texts.append(f"**{source_label}:**\n{pa.answer}")

            source_answers = "\n\n".join(source_texts)

            # Generate synthesized answer
            synthesis_prompt = SYNTHESIS_PROMPT.format(
                question=state["question"],
                source_answers=source_answers,
            )

            final_answer = llm_client.generate(
                prompt="Synthesize the above information.",
                system_prompt=synthesis_prompt,
                max_tokens=1500,
                temperature=0.7,
            )

        synthesis_time = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Synthesizer: combined %d answers in %.1fms",
            len(successful_answers),
            synthesis_time,
        )

        return {
            "final_answer": final_answer,
            "generation_time_ms": state.get("generation_time_ms", 0.0) + synthesis_time,
            "node_status": {
                **state.get("node_status", {}),
                "synthesizer": ExecutionStatus.SUCCESS.value,
            },
        }

    except Exception as e:
        logger.exception("Synthesizer failed")
        error = NodeError.from_exception("synthesizer", e, recoverable=False)

        # Try to return best available answer
        partial_answers = state.get("partial_answers", [])
        fallback_answer = "An error occurred while generating the response."
        if partial_answers:
            # Use the highest confidence partial answer
            best = max(partial_answers, key=lambda x: x.confidence)
            fallback_answer = best.answer

        return {
            "final_answer": fallback_answer,
            "errors": state.get("errors", []) + [error],
            "node_status": {
                **state.get("node_status", {}),
                "synthesizer": ExecutionStatus.FAILED.value,
            },
        }


def critic_node(state: GraphState) -> dict:
    """Evaluate the final answer (placeholder).

    This is a placeholder node for future critic functionality.
    Currently always passes without evaluation.

    Args:
        state: Current graph state.

    Returns:
        State updates with critique result.
    """
    # Check if critic is enabled
    config = state.get("config")
    if not config or not config.enable_critic:
        logger.debug("Critic: disabled, skipping")
        return {
            "critique": CritiqueResult(passed=True),
            "node_status": {
                **state.get("node_status", {}),
                "critic": ExecutionStatus.SKIPPED.value,
            },
        }

    # Placeholder: always pass
    # Future implementation would evaluate the answer quality
    logger.debug("Critic: placeholder - always passes")
    return {
        "critique": CritiqueResult(passed=True),
        "node_status": {
            **state.get("node_status", {}),
            "critic": ExecutionStatus.SUCCESS.value,
        },
    }


def should_run_docs_agent(state: GraphState) -> bool:
    """Check if docs agent should run based on routing.

    Args:
        state: Current graph state.

    Returns:
        True if docs agent should execute.
    """
    route_decision = state.get("route_decision")
    if not route_decision:
        return True  # Default to running

    return route_decision.query_type in (QueryType.DOCUMENTS, QueryType.HYBRID)


def should_run_sql_agent(state: GraphState) -> bool:
    """Check if SQL agent should run based on routing.

    Args:
        state: Current graph state.

    Returns:
        True if SQL agent should execute.
    """
    route_decision = state.get("route_decision")
    if not route_decision:
        return False  # Default to not running SQL

    return route_decision.query_type in (QueryType.STRUCTURED, QueryType.HYBRID)


def get_next_after_router(state: GraphState) -> str:
    """Determine next node after routing.

    Args:
        state: Current graph state.

    Returns:
        Name of next node to execute.
    """
    route_decision = state.get("route_decision")

    if not route_decision:
        # Routing failed - check errors
        errors = state.get("errors", [])
        if any(not e.recoverable for e in errors):
            return "synthesizer"  # Go to synthesizer to handle error
        return "docs_agent"  # Default fallback

    query_type = route_decision.query_type

    if query_type == QueryType.DOCUMENTS:
        return "docs_agent"
    elif query_type == QueryType.STRUCTURED:
        return "sql_agent"
    else:  # HYBRID
        return "docs_agent"  # Start with docs, then SQL


def get_next_after_docs(state: GraphState) -> str:
    """Determine next node after docs agent.

    Args:
        state: Current graph state.

    Returns:
        Name of next node to execute.
    """
    route_decision = state.get("route_decision")

    if route_decision and route_decision.query_type == QueryType.HYBRID:
        return "sql_agent"
    else:
        return "synthesizer"


def get_next_after_sql(state: GraphState) -> str:
    """Determine next node after SQL agent.

    Args:
        state: Current graph state.

    Returns:
        Name of next node to execute.
    """
    return "synthesizer"


def get_next_after_synthesizer(state: GraphState) -> str:
    """Determine next node after synthesizer.

    Args:
        state: Current graph state.

    Returns:
        Name of next node to execute or END.
    """
    config = state.get("config")

    if config and config.enable_critic:
        return "critic"
    else:
        return "__end__"


def get_next_after_critic(state: GraphState) -> str:
    """Determine next node after critic.

    Args:
        state: Current graph state.

    Returns:
        Name of next node to execute or END.
    """
    critique = state.get("critique")
    config = state.get("config")

    if critique and not critique.passed:
        # Check if we've exceeded max revisions
        max_revisions = config.max_revisions if config else 1
        if critique.revision_number < max_revisions:
            # Would loop back for revision - but not implemented yet
            pass

    return "__end__"
