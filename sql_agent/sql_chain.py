"""Main SQL Chain for end-to-end query processing.

This module provides the SQLChain class that orchestrates the full
pipeline: schema extraction, SQL generation, validation, execution,
and answer generation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from generation.base import BaseLLMClient
from storage.sql_store import SQLStore, SQLStoreConfig

from .config import SQLAgentConfig
from .errors import QueryExecutionError, QueryGenerationError, QueryValidationError
from .query_validator import QueryValidator
from .schema_extractor import SchemaExtractor
from .sql_generator import SQLGenerator

if TYPE_CHECKING:
    from security import SecurityContext, TableACL

logger = logging.getLogger(__name__)

# System prompt for summarizing SQL results
ANSWER_SYSTEM_PROMPT = """You are a helpful data analyst. The user asked a question about data in a database.
A SQL query was executed and returned the results shown below.

Summarize the results in a clear, concise answer. Include specific numbers and data points.
If the results are empty, explain that no matching data was found.

## Question: {question}

## SQL Query:
```sql
{sql}
```

## Results:
{results}

Provide a natural language summary of these results:"""


@dataclass
class SQLQueryResult:
    """Result from SQL query execution."""

    answer: str
    data: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    generated_sql: str
    tables_used: List[str] = field(default_factory=list)
    generation_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    summarization_time_ms: float = 0.0


@dataclass
class SQLHealthCheckResult:
    """Result from SQL agent health check."""

    healthy: bool
    message: str
    table_count: int = 0
    total_rows: int = 0


class SQLChain:
    """SQL Chain that processes natural language to SQL queries.

    This class orchestrates the full pipeline following the RAGChain pattern:
    1. Extract database schema
    2. Generate SQL from natural language
    3. Validate the generated SQL
    4. Execute the query
    5. Generate natural language answer

    All database access is read-only for safety.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        config: Optional[SQLAgentConfig] = None,
    ) -> None:
        """Initialize the SQL Chain.

        Args:
            llm_client: LLM client for generation.
            config: SQLAgentConfig with settings.
        """
        self.llm_client = llm_client
        self.config = config or SQLAgentConfig()

        # Lazy-initialized components
        self._sql_store: Optional[SQLStore] = None
        self._schema_extractor: Optional[SchemaExtractor] = None
        self._validator: Optional[QueryValidator] = None
        self._generator: Optional[SQLGenerator] = None

    def _get_sql_store(self) -> SQLStore:
        """Get or initialize SQLStore."""
        if self._sql_store is None:
            store_config = SQLStoreConfig(
                db_path=self.config.db_path,
                read_only=True,  # SAFETY: Always read-only
            )
            self._sql_store = SQLStore(store_config)
        return self._sql_store

    def _get_schema_extractor(
        self, excluded_tables: Optional[List[str]] = None
    ) -> SchemaExtractor:
        """Get or initialize SchemaExtractor.

        Args:
            excluded_tables: Optional list of tables to exclude. If provided,
                creates a fresh extractor with these exclusions.
        """
        if excluded_tables is not None:
            # Create a fresh extractor with dynamic exclusions
            return SchemaExtractor(
                config=self.config,
                sql_store=self._get_sql_store(),
                excluded_tables=excluded_tables,
            )

        if self._schema_extractor is None:
            self._schema_extractor = SchemaExtractor(
                config=self.config,
                sql_store=self._get_sql_store(),
            )
        return self._schema_extractor

    def _get_validator(self) -> QueryValidator:
        """Get or initialize QueryValidator."""
        if self._validator is None:
            self._validator = QueryValidator(self.config)
        return self._validator

    def _get_generator(
        self, excluded_tables: Optional[List[str]] = None
    ) -> SQLGenerator:
        """Get or initialize SQLGenerator.

        Args:
            excluded_tables: Optional list of tables to exclude. If provided,
                creates a fresh generator with these exclusions instead of
                using the cached one.
        """
        if excluded_tables is not None:
            # Create a fresh generator with dynamic exclusions
            schema_extractor = self._get_schema_extractor(
                excluded_tables=excluded_tables
            )
            return SQLGenerator(
                llm_client=self.llm_client,
                config=self.config,
                schema_extractor=schema_extractor,
            )

        if self._generator is None:
            self._generator = SQLGenerator(
                llm_client=self.llm_client,
                config=self.config,
                schema_extractor=self._get_schema_extractor(),
            )
        return self._generator

    def query(
        self,
        question: str,
        max_rows: Optional[int] = None,
        summarize: bool = True,
        security_context: Optional["SecurityContext"] = None,
        table_acl: Optional["TableACL"] = None,
    ) -> SQLQueryResult:
        """Execute a natural language query against the database.

        Args:
            question: Natural language question.
            max_rows: Maximum rows to return (overrides config).
            summarize: Whether to generate natural language answer.
            security_context: Optional security context for ACL filtering.
            table_acl: Optional table ACL for access control.

        Returns:
            SQLQueryResult with data and optional answer.

        Raises:
            QueryGenerationError: If SQL generation fails.
            QueryValidationError: If generated SQL is invalid.
            QueryExecutionError: If SQL execution fails.
        """
        effective_max_rows = max_rows or self.config.max_result_rows

        # Compute dynamic excluded tables based on ACL
        dynamic_excluded: Set[str] = set(self.config.excluded_tables)
        if table_acl is not None and security_context is not None:
            # Get all available tables
            sql_store = self._get_sql_store()
            all_tables = sql_store.list_tables()
            # Get tables blocked by ACL
            blocked_tables = table_acl.get_excluded_tables(all_tables, security_context)
            dynamic_excluded.update(blocked_tables)

        # Step 1: Generate SQL (with dynamic excluded tables)
        generation_start = time.perf_counter()
        generator = self._get_generator(excluded_tables=list(dynamic_excluded))
        sql = generator.generate(question, max_rows=effective_max_rows)
        generation_time_ms = (time.perf_counter() - generation_start) * 1000

        # Step 2: Validate SQL
        validator = self._get_validator()
        sql = validator.validate_or_raise(sql)

        # Extract tables used
        tables_used = validator.extract_tables(sql)

        # Step 2.5: Validate ACL on extracted tables
        if table_acl is not None and security_context is not None:
            is_valid, unauthorized = table_acl.validate_query_tables(
                tables_used, security_context
            )
            if not is_valid:
                raise QueryValidationError(
                    f"Access denied to tables: {unauthorized}"
                )

        # Step 3: Execute SQL
        execution_start = time.perf_counter()
        data, columns = self._execute_query(sql)
        execution_time_ms = (time.perf_counter() - execution_start) * 1000

        # Step 4: Generate answer (optional)
        summarization_time_ms = 0.0
        if summarize and data:
            summarization_start = time.perf_counter()
            answer = self._generate_answer(question, sql, data, columns)
            summarization_time_ms = (time.perf_counter() - summarization_start) * 1000
        elif not data:
            answer = "No results found for your query."
        else:
            # Just format the data as a simple response
            answer = f"Query returned {len(data)} row(s)."

        return SQLQueryResult(
            answer=answer,
            data=data,
            columns=columns,
            row_count=len(data),
            generated_sql=sql,
            tables_used=tables_used,
            generation_time_ms=round(generation_time_ms, 2),
            execution_time_ms=round(execution_time_ms, 2),
            summarization_time_ms=round(summarization_time_ms, 2),
        )

    def _execute_query(self, sql: str) -> tuple[List[Dict[str, Any]], List[str]]:
        """Execute a validated SQL query.

        Args:
            sql: Validated SQL query.

        Returns:
            Tuple of (data rows as dicts, column names).

        Raises:
            QueryExecutionError: If execution fails.
        """
        sql_store = self._get_sql_store()

        try:
            result = sql_store.execute(sql)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()

            # Convert to list of dicts
            data = []
            for row in rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    value = row[i]
                    # Convert complex types to strings for JSON serialization
                    if hasattr(value, "isoformat"):
                        value = value.isoformat()
                    row_dict[col] = value
                data.append(row_dict)

            logger.info("Query executed: %d rows, %d columns", len(data), len(columns))
            return data, columns

        except Exception as e:
            logger.exception("Query execution failed")
            raise QueryExecutionError(f"Query execution failed: {e}") from e

    def _generate_answer(
        self,
        question: str,
        sql: str,
        data: List[Dict[str, Any]],
        columns: List[str],
    ) -> str:
        """Generate natural language answer from query results.

        Args:
            question: Original user question.
            sql: Executed SQL query.
            data: Query result data.
            columns: Column names.

        Returns:
            Natural language summary of results.
        """
        # Format results as a table
        results_text = self._format_results_for_prompt(data, columns)

        system_prompt = ANSWER_SYSTEM_PROMPT.format(
            question=question,
            sql=sql,
            results=results_text,
        )

        try:
            answer = self.llm_client.generate(
                prompt="Summarize these query results.",
                system_prompt=system_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            return answer.strip()
        except Exception as e:
            logger.warning("Answer generation failed: %s", e)
            return f"Query returned {len(data)} row(s), but summarization failed."

    def _format_results_for_prompt(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
        max_rows: int = 20,
    ) -> str:
        """Format query results as markdown table for LLM.

        Args:
            data: Query result data.
            columns: Column names.
            max_rows: Maximum rows to include in prompt.

        Returns:
            Markdown-formatted table string.
        """
        if not data:
            return "No results."

        lines = []

        # Header
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

        # Rows (limit for prompt size)
        for row in data[:max_rows]:
            values = []
            for col in columns:
                val = row.get(col, "")
                # Truncate long values
                val_str = str(val) if val is not None else ""
                if len(val_str) > 50:
                    val_str = val_str[:47] + "..."
                values.append(val_str)
            lines.append("| " + " | ".join(values) + " |")

        if len(data) > max_rows:
            lines.append(f"\n... and {len(data) - max_rows} more rows")

        return "\n".join(lines)

    def health_check(self) -> SQLHealthCheckResult:
        """Check SQL agent health.

        Returns:
            SQLHealthCheckResult with status and statistics.
        """
        try:
            sql_store = self._get_sql_store()

            # Verify we can connect and query
            tables = sql_store.list_tables()

            # Filter out excluded tables
            excluded = set(t.lower() for t in self.config.excluded_tables)
            user_tables = [t for t in tables if t.lower() not in excluded]

            # Count total rows
            total_rows = 0
            for table in user_tables:
                try:
                    total_rows += sql_store.get_table_count(table)
                except Exception:
                    pass  # Ignore individual table errors

            if not user_tables:
                return SQLHealthCheckResult(
                    healthy=True,
                    message="SQL Agent healthy but no user tables found.",
                    table_count=0,
                    total_rows=0,
                )

            return SQLHealthCheckResult(
                healthy=True,
                message=f"SQL Agent healthy. {len(user_tables)} tables, {total_rows:,} total rows.",
                table_count=len(user_tables),
                total_rows=total_rows,
            )

        except Exception as e:
            logger.exception("SQL Agent health check failed")
            return SQLHealthCheckResult(
                healthy=False,
                message=f"SQL Agent error: {e}",
                table_count=0,
                total_rows=0,
            )

    def get_available_tables(self) -> List[str]:
        """Get list of available tables for querying.

        Returns:
            List of table names.
        """
        schema_extractor = self._get_schema_extractor()
        return schema_extractor.get_table_names()

    def close(self) -> None:
        """Close all connections and resources."""
        if self._schema_extractor is not None:
            self._schema_extractor.close()
            self._schema_extractor = None

        if self._sql_store is not None:
            self._sql_store.close()
            self._sql_store = None

        self._validator = None
        self._generator = None
