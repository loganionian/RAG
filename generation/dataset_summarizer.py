"""LLM-based dataset summary generation.

This module generates natural-language summaries for tabular datasets,
describing their contents, columns, and potential use cases.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from storage.metadata_catalog import CatalogEntry, MetadataCatalog
from storage.sql_store import SQLStore

from .base import BaseLLMClient
from .cost_tracker import CostConfig, CostTracker

logger = logging.getLogger(__name__)

# Default prompt template for summary generation
DEFAULT_SYSTEM_PROMPT = """You are a data documentation assistant. Your task is to generate concise, informative summaries for datasets.

Guidelines:
- Write clear, professional descriptions
- Focus on what the data contains and its likely use cases
- Describe key columns, especially non-obvious ones
- Keep summaries brief but informative (2-3 sentences)
- Format response as valid JSON"""

DEFAULT_USER_PROMPT_TEMPLATE = """Generate a summary for this dataset.

Table: {table_name}
Source File: {source_file}
Rows: {row_count:,}
Columns: {column_count}

Schema:
{schema_formatted}

Sample Data (first {sample_count} rows):
{sample_data_formatted}

Generate a JSON response with:
1. "summary": A 2-3 sentence description of what this dataset contains and its likely use cases
2. "column_descriptions": Brief descriptions for key columns (focus on non-obvious ones)

Response format:
{{
  "summary": "...",
  "column_descriptions": {{"column1": "description", "column2": "description"}}
}}"""


@dataclass
class SummaryConfig:
    """Configuration for dataset summary generation.

    Attributes:
        max_tokens: Maximum tokens for LLM response.
        temperature: Sampling temperature (lower = more deterministic).
        sample_rows: Number of sample rows to include in prompt.
        max_columns_in_prompt: Maximum columns to show in prompt.
        retry_count: Number of retries on failure.
        system_prompt: Custom system prompt (uses default if None).
        user_prompt_template: Custom user prompt template.
    """

    max_tokens: int = 400
    temperature: float = 0.3
    sample_rows: int = 5
    max_columns_in_prompt: int = 20
    retry_count: int = 3
    system_prompt: Optional[str] = None
    user_prompt_template: Optional[str] = None


@dataclass
class DatasetSummary:
    """Generated summary for a tabular dataset.

    Attributes:
        table_name: Name of the SQL table.
        summary: Natural-language description of the dataset.
        column_descriptions: Descriptions for key columns.
        sample_values: Sample values for each column.
        generated_at: Timestamp when summary was generated.
        token_usage: Token counts from LLM response.
        source_file: Original source file path.
        domain_labels: Domain labels for the dataset.
        row_count: Number of rows in the dataset.
        column_count: Number of columns in the dataset.
    """

    table_name: str
    summary: str
    column_descriptions: Dict[str, str] = field(default_factory=dict)
    sample_values: Dict[str, List[Any]] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)
    token_usage: Dict[str, int] = field(default_factory=dict)
    source_file: str = ""
    domain_labels: List[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "table_name": self.table_name,
            "summary": self.summary,
            "column_descriptions": self.column_descriptions,
            "sample_values": {k: [str(v) for v in vals] for k, vals in self.sample_values.items()},
            "generated_at": self.generated_at.isoformat(),
            "token_usage": self.token_usage,
            "source_file": self.source_file,
            "domain_labels": self.domain_labels,
            "row_count": self.row_count,
            "column_count": self.column_count,
        }


class DatasetSummaryError(Exception):
    """Error during dataset summary generation."""

    pass


class DatasetSummarizer:
    """Generates natural-language summaries for tabular datasets.

    Usage:
        from generation import create_llm_client
        from storage import SQLStore, SQLStoreConfig, MetadataCatalog

        llm_client = create_llm_client()
        sql_store = SQLStore(SQLStoreConfig(db_path=Path("data/tabular.db")))
        catalog = MetadataCatalog(sql_store)

        summarizer = DatasetSummarizer(
            llm_client=llm_client,
            sql_store=sql_store,
            catalog=catalog,
        )

        summary = summarizer.generate_summary("sales_2024")
        print(summary.summary)
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        sql_store: SQLStore,
        catalog: MetadataCatalog,
        config: Optional[SummaryConfig] = None,
        cost_config: Optional[CostConfig] = None,
    ) -> None:
        """Initialize the dataset summarizer.

        Args:
            llm_client: LLM client for generating summaries.
            sql_store: SQL store for querying table data.
            catalog: Metadata catalog for table information.
            config: Summary generation configuration.
            cost_config: Cost tracking configuration.
        """
        self.llm_client = llm_client
        self.sql_store = sql_store
        self.catalog = catalog
        self.config = config or SummaryConfig()
        self.cost_tracker = CostTracker(cost_config)

    def generate_summary(self, table_name: str) -> DatasetSummary:
        """Generate a summary for a single table.

        Args:
            table_name: Name of the table to summarize.

        Returns:
            DatasetSummary with generated content.

        Raises:
            DatasetSummaryError: If summary generation fails.
        """
        # Get catalog entry
        entry = self.catalog.get_entry(table_name)
        if entry is None:
            raise DatasetSummaryError(f"Table '{table_name}' not found in catalog")

        # Fetch sample data
        sample_data = self._fetch_sample_rows(table_name, self.config.sample_rows)

        # Build prompt
        prompt = self._build_prompt(entry, sample_data)

        # Generate summary with retry logic
        last_error = None
        for attempt in range(self.config.retry_count):
            try:
                response = self.llm_client.chat(
                    messages=[
                        {"role": "system", "content": self.config.system_prompt or DEFAULT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )

                # Track token usage
                usage = response.usage or {}
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                self.cost_tracker.track_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    operation=f"summarize:{table_name}",
                )

                # Parse response
                parsed = self._parse_response(response.content)

                return DatasetSummary(
                    table_name=table_name,
                    summary=parsed.get("summary", ""),
                    column_descriptions=parsed.get("column_descriptions", {}),
                    sample_values=self._extract_sample_values(sample_data, entry.column_schema),
                    generated_at=datetime.now(),
                    token_usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
                    source_file=entry.source_file,
                    domain_labels=entry.domain_labels,
                    row_count=entry.row_count,
                    column_count=entry.column_count,
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    "Summary generation attempt %d/%d failed for '%s': %s",
                    attempt + 1,
                    self.config.retry_count,
                    table_name,
                    e,
                )

        raise DatasetSummaryError(
            f"Failed to generate summary for '{table_name}' after {self.config.retry_count} attempts: {last_error}"
        )

    def generate_summaries_batch(
        self,
        table_names: Optional[List[str]] = None,
        skip_existing: bool = True,
    ) -> List[DatasetSummary]:
        """Generate summaries for multiple tables.

        Args:
            table_names: List of tables to summarize. If None, summarizes all tables.
            skip_existing: Skip tables that already have summaries.

        Returns:
            List of generated DatasetSummary objects.
        """
        # Get tables to process
        if table_names is None:
            entries = self.catalog.list_tables()
            table_names = [e.table_name for e in entries]

        summaries = []
        failed = []

        for table_name in table_names:
            # Check budget
            if not self.cost_tracker.can_proceed():
                logger.warning(
                    "Token budget exhausted after %d summaries. Stopping.",
                    len(summaries),
                )
                break

            try:
                summary = self.generate_summary(table_name)
                summaries.append(summary)
                logger.info(
                    "Generated summary for '%s' (%d tokens)",
                    table_name,
                    summary.token_usage.get("input_tokens", 0) + summary.token_usage.get("output_tokens", 0),
                )
            except DatasetSummaryError as e:
                failed.append((table_name, str(e)))
                logger.error("Failed to summarize '%s': %s", table_name, e)

        if failed:
            logger.warning("Failed to generate %d summaries: %s", len(failed), [f[0] for f in failed])

        return summaries

    def _build_prompt(self, entry: CatalogEntry, sample_data: List[Dict[str, Any]]) -> str:
        """Build the user prompt for summary generation.

        Args:
            entry: Catalog entry with table metadata.
            sample_data: Sample rows from the table.

        Returns:
            Formatted prompt string.
        """
        # Format schema
        schema = entry.column_schema[: self.config.max_columns_in_prompt]
        schema_lines = []
        for col in schema:
            col_name = col.get("name", "unknown")
            col_type = col.get("type", "unknown")
            schema_lines.append(f"  - {col_name}: {col_type}")
        schema_formatted = "\n".join(schema_lines)

        if len(entry.column_schema) > self.config.max_columns_in_prompt:
            schema_formatted += f"\n  ... and {len(entry.column_schema) - self.config.max_columns_in_prompt} more columns"

        # Format sample data
        sample_lines = []
        for i, row in enumerate(sample_data[: self.config.sample_rows], 1):
            row_str = ", ".join(f"{k}={self._format_value(v)}" for k, v in list(row.items())[:8])
            if len(row) > 8:
                row_str += f", ... (+{len(row) - 8} more)"
            sample_lines.append(f"  Row {i}: {row_str}")
        sample_formatted = "\n".join(sample_lines) if sample_lines else "  (no sample data available)"

        template = self.config.user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE

        return template.format(
            table_name=entry.table_name,
            source_file=entry.source_file,
            row_count=entry.row_count,
            column_count=entry.column_count,
            schema_formatted=schema_formatted,
            sample_count=len(sample_data),
            sample_data_formatted=sample_formatted,
        )

    def _fetch_sample_rows(self, table_name: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch sample rows from a table.

        Args:
            table_name: Name of the table.
            limit: Maximum rows to fetch.

        Returns:
            List of row dictionaries.
        """
        try:
            # Get column names
            schema = self.sql_store.get_table_schema(table_name)
            columns = [col["name"] for col in schema]

            # Fetch rows
            query = f'SELECT * FROM "{table_name}" LIMIT {limit}'
            rows = self.sql_store.fetchall(query)

            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.warning("Failed to fetch sample rows for '%s': %s", table_name, e)
            return []

    def _format_value(self, value: Any) -> str:
        """Format a value for display in the prompt.

        Args:
            value: Value to format.

        Returns:
            Formatted string representation.
        """
        if value is None:
            return "NULL"
        if isinstance(value, str):
            # Truncate long strings
            if len(value) > 50:
                return f'"{value[:47]}..."'
            return f'"{value}"'
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)[:50]

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response as JSON.

        Args:
            response_text: Raw response from LLM.

        Returns:
            Parsed dictionary with summary and column_descriptions.
        """
        # Try to extract JSON from response
        try:
            # First try direct JSON parse
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in response
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fall back to extracting summary from plain text
        logger.warning("Could not parse JSON response, extracting plain text summary")
        return {
            "summary": response_text.strip()[:500],
            "column_descriptions": {},
        }

    def _extract_sample_values(
        self,
        sample_data: List[Dict[str, Any]],
        column_schema: List[Dict[str, str]],
    ) -> Dict[str, List[Any]]:
        """Extract sample values for each column.

        Args:
            sample_data: Sample rows from the table.
            column_schema: Column schema from catalog.

        Returns:
            Dictionary mapping column names to sample values.
        """
        sample_values = {}
        column_names = [col.get("name", "") for col in column_schema]

        for col_name in column_names:
            values = []
            for row in sample_data:
                if col_name in row:
                    values.append(row[col_name])
            if values:
                sample_values[col_name] = values[:5]

        return sample_values

    def get_cost_report(self):
        """Get the cost tracking report.

        Returns:
            CostReport with usage statistics.
        """
        return self.cost_tracker.get_report()

    def estimate_tokens(self, table_name: str) -> int:
        """Estimate tokens needed to summarize a table.

        Args:
            table_name: Name of the table.

        Returns:
            Estimated token count.
        """
        entry = self.catalog.get_entry(table_name)
        if entry is None:
            return 0

        # Rough estimation: ~4 chars per token
        schema_chars = sum(len(col.get("name", "")) + len(col.get("type", "")) + 10 for col in entry.column_schema)
        sample_chars = entry.column_count * self.config.sample_rows * 20  # ~20 chars per cell
        base_prompt_chars = 500  # Template overhead

        estimated_input = (schema_chars + sample_chars + base_prompt_chars) // 4
        estimated_output = self.config.max_tokens

        return estimated_input + estimated_output
