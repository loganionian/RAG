"""SQL query generation using LLM.

This module provides LLM-based SQL generation from natural language questions.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from generation.base import BaseLLMClient

from .config import SQLAgentConfig
from .errors import QueryGenerationError
from .schema_extractor import SchemaExtractor

logger = logging.getLogger(__name__)

# Marker for LLM to indicate it cannot answer
CANNOT_ANSWER_PREFIX = "CANNOT_ANSWER:"


class SQLGenerator:
    """Generates SQL queries from natural language using LLM.

    Uses the configured LLM client to translate questions into SQL,
    providing database schema context for accurate generation.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        config: Optional[SQLAgentConfig] = None,
        schema_extractor: Optional[SchemaExtractor] = None,
    ) -> None:
        """Initialize the SQL generator.

        Args:
            llm_client: LLM client for generation.
            config: SQLAgentConfig with generation settings.
            schema_extractor: Optional pre-configured schema extractor.
        """
        self.llm_client = llm_client
        self.config = config or SQLAgentConfig()
        self._schema_extractor = schema_extractor

    def _get_schema_extractor(self) -> SchemaExtractor:
        """Get or create schema extractor."""
        if self._schema_extractor is None:
            self._schema_extractor = SchemaExtractor(self.config)
        return self._schema_extractor

    def generate(self, question: str, max_rows: Optional[int] = None) -> str:
        """Generate SQL from a natural language question.

        Args:
            question: Natural language question.
            max_rows: Maximum rows for LIMIT clause (overrides config).

        Returns:
            Generated SQL query string.

        Raises:
            QueryGenerationError: If generation fails or LLM cannot answer.
        """
        if not question or not question.strip():
            raise QueryGenerationError("Empty question")

        # Get schema for prompt
        schema_extractor = self._get_schema_extractor()
        schema_text = schema_extractor.format_schema_for_prompt()

        if "No tables available" in schema_text:
            raise QueryGenerationError("No tables available in the database")

        # Build prompt
        effective_max_rows = max_rows or self.config.max_result_rows
        system_prompt = self.config.system_prompt.format(
            schema=schema_text,
            max_rows=effective_max_rows,
            question=question,
        )

        logger.debug("Generating SQL for question: %s", question[:100])

        try:
            response = self.llm_client.generate(
                prompt=question,
                system_prompt=system_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
        except Exception as e:
            logger.exception("LLM generation failed")
            raise QueryGenerationError(f"LLM generation failed: {e}") from e

        # Parse response
        sql = self._parse_response(response)

        logger.debug("Generated SQL: %s", sql[:200] if len(sql) > 200 else sql)

        return sql

    def _parse_response(self, response: str) -> str:
        """Parse LLM response to extract SQL query.

        Args:
            response: Raw LLM response text.

        Returns:
            Extracted SQL query.

        Raises:
            QueryGenerationError: If response indicates cannot answer or is empty.
        """
        if not response or not response.strip():
            raise QueryGenerationError("Empty response from LLM")

        response = response.strip()

        # Check for CANNOT_ANSWER marker
        if response.upper().startswith(CANNOT_ANSWER_PREFIX.upper()):
            reason = response[len(CANNOT_ANSWER_PREFIX):].strip()
            raise QueryGenerationError(f"Cannot generate SQL: {reason}")

        # Extract SQL from markdown code blocks if present
        sql = self._extract_from_code_block(response)

        if not sql:
            # Try to use the response directly
            sql = response

        # Clean up the SQL
        sql = self._clean_sql(sql)

        if not sql:
            raise QueryGenerationError("Could not extract valid SQL from response")

        return sql

    def _extract_from_code_block(self, text: str) -> Optional[str]:
        """Extract SQL from markdown code blocks.

        Args:
            text: Text that may contain code blocks.

        Returns:
            Extracted SQL or None if no code block found.
        """
        # Match ```sql ... ``` or ``` ... ```
        patterns = [
            r"```sql\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _clean_sql(self, sql: str) -> str:
        """Clean and normalize SQL query.

        Args:
            sql: Raw SQL string.

        Returns:
            Cleaned SQL string.
        """
        # Remove leading/trailing whitespace
        sql = sql.strip()

        # Remove trailing semicolons (we'll add them if needed)
        sql = sql.rstrip(";")

        # Remove common LLM explanatory prefixes
        prefixes_to_remove = [
            "Here is the SQL query:",
            "Here's the SQL query:",
            "The SQL query is:",
            "SQL:",
            "Query:",
        ]

        sql_upper = sql.upper()
        for prefix in prefixes_to_remove:
            if sql_upper.startswith(prefix.upper()):
                sql = sql[len(prefix):].strip()
                break

        return sql
