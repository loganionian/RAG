# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG MVP - A Python-based Retrieval Augmented Generation system for semantic document search. The pipeline ingests documents, chunks them, generates embeddings, and stores them in a Chroma vector database for retrieval.

## Commands

### Setup

**Requires Python 3.12+**

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Run Ingestion Pipeline
```bash
python -m scripts.ingest --input-dir data/raw --output-dir data/processed --verbose
```

### Run Indexing Pipeline
```bash
python -m scripts.index_chunks --processed-dir data/processed --chroma-dir data/vectorstore --verbose
```

### Query the Vector Store
```bash
python -m scripts.query_chunks --question "Your question here" --k 3 --pretty
```

### Re-index Specific Documents
```bash
python -m scripts.index_chunks --doc-ids doc-id-1 doc-id-2 --verbose
```

### RAG Chat (Query with LLM Response)
```bash
# Single question
python -m scripts.rag_chat --question "What are the key findings?" --k 5

# Interactive chat mode
python -m scripts.rag_chat --interactive

# Show retrieved sources
python -m scripts.rag_chat -q "What skills are in demand?" --show-sources

# JSON output
python -m scripts.rag_chat -q "Summarize the report" --json
```

### SQL Chat (Query Tables with Natural Language)
```bash
# Single question
python -m scripts.sql_chat -q "What tables are available?"

# Interactive mode
python -m scripts.sql_chat --interactive

# Show generated SQL
python -m scripts.sql_chat -q "Top 5 products by price" --show-sql

# JSON output
python -m scripts.sql_chat -q "Count rows per table" --json

# List available tables
python -m scripts.sql_chat --list-tables
```

### REST API Server
```bash
# Start the API server (default: http://localhost:8080)
python -m uvicorn api.main:app --reload

# Start with custom host/port
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080
```

**Required environment variables** (in `.env`):
```
API_KEY=your_api_key
API_SECRET=your_api_secret
BASE_URL=https://your-llm-api-endpoint
```

## Architecture

### Data Flow
```
data/raw/ (PDF, DOC, DOCX, XLSX, XLS, CSV, TSV, MD, TXT)
    → [ingestion]  → data/processed/chunks/*.jsonl + manifest.json + failures.json + ingestion-report.json
    → [indexing]   → data/vectorstore/ (Chroma + BM25)
    → [query]      → Top-k matches (vector, lexical, or hybrid)
    → [generation] → LLM-powered answers with retrieved context
```

### Package Structure

**ingestion/** - Document parsing and chunking
- `loader.py`: Multi-format document loading (PDF via PyPDF2, DOC via antiword/win32com, DOCX via python-docx, Excel via openpyxl, CSV/TSV via csv module)
- `chunker.py`: Paragraph-aware chunking (~400 tokens, 80-token overlap)
- `storage.py`: JSONL chunk persistence, manifest tracking, failure/report storage
- `pipeline.py`: `IngestionPipeline` orchestrates discover → load → classify → chunk → store
- `normalizer.py`: Configurable text normalization with `TextNormalizer` and `NormalizationConfig`
- `normalization_rules.py`: Pre-defined regex patterns for page numbers, boilerplate, special chars
- `spreadsheet_classifier.py`: Heuristic-based classification of spreadsheets as tabular vs report-like
- `models.py`: Data models (`Document`, `DocumentChunk`, `FailureInfo`)

**indexing/** - Vector database operations
- `chroma_store.py`: Chroma persistence with cosine distance
- `bm25_store.py`: `BM25Store` for persistent BM25 lexical index with pickle storage
- `lexical_search.py`: `LexicalSearchService` wrapper for BM25 with RetrievalResult compatibility
- `embeddings.py`: `EmbeddingService` with retry logic, `EmbeddingConfig`, `EmbeddingError`
- `dataset.py`: Loads chunks from manifest/JSONL files
- `pipeline.py`: `ChromaIndexingPipeline` handles batch embedding and upsert to Chroma + BM25
- `summary_indexer.py`: `SummaryIndexer` for indexing dataset summaries in Chroma

**generation/** - LLM-powered response generation
- `api_client.py`: HMAC-authenticated LLM client with `LLMClient`, `LLMConfig`
- `rag_chain.py`: `RAGChain` combines retrieval (vector/lexical/hybrid) and generation
- `dataset_summarizer.py`: `DatasetSummarizer` generates LLM summaries for tabular datasets
- `cost_tracker.py`: `CostTracker` for token usage tracking and budget controls

**retrieval/** - Score normalization and reranking
- `score_normalizer.py`: `ScoreNormalizer` with min-max, z-score, and rank normalization methods
- `reranker.py`: `CrossEncoderReranker` for cross-encoder reranking with lazy model loading

**api/** - REST API (UI-agnostic)
- `main.py`: FastAPI application with CORS, lifespan events
- `schemas.py`: Pydantic request/response models
- `routes/query.py`: POST /api/query endpoint
- `routes/ingest.py`: POST /api/ingest endpoint
- `routes/documents.py`: GET /api/documents, GET /api/health endpoints
- `routes/sql_query.py`: POST /api/sql-query, GET /api/sql-tables endpoints

**sql_agent/** - Natural language to SQL agent
- `config.py`: `SQLAgentConfig` with database settings and guardrails
- `errors.py`: Custom exceptions (`QueryValidationError`, `QueryGenerationError`, `QueryExecutionError`)
- `query_validator.py`: `QueryValidator` for SELECT-only validation and auto-LIMIT
- `schema_extractor.py`: `SchemaExtractor` for database schema extraction
- `sql_generator.py`: `SQLGenerator` for LLM-based SQL generation
- `sql_chain.py`: `SQLChain` orchestrates generation → validation → execution → summarization

**scripts/** - CLI entry points for each pipeline stage

### Key Design Patterns

- **Idempotency**: Content SHA256 hashes in `manifest.json` prevent reprocessing unchanged documents
- **Chunk IDs**: Format `{doc_id}::chunk-{index:04d}` enables selective re-indexing
- **Batch Processing**: Configurable batch sizes for embedding and upsert operations
- **Fail-fast Mode**: Optional `--fail-fast` flag to stop on first error vs. continue with failures
- **Failure Tolerance**: Failed documents are logged with full details and can be retried independently

### Configuration Defaults
- Chunk size: 400 tokens
- Chunk overlap: 80 tokens
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimensions: 384
- Vector metric: Cosine distance
- Upsert batch size: 32 chunks
- Embedding retries: 3 (with exponential backoff)
- RAG retrieval: Top-5 chunks
- LLM temperature: 0.7
- LLM max tokens: 1000
- BM25 k1: 1.5 (term frequency saturation)
- BM25 b: 0.75 (length normalization)
- Hybrid lexical weight: 0.3 (30% lexical, 70% vector in RRF fusion)
- Reranking: Disabled by default
- Reranker model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Reranker top-k multiplier: 3 (fetch k*3 candidates before reranking)

### BM25 Lexical Search

The indexing pipeline creates a BM25 lexical index alongside the Chroma vector index, enabling keyword-based search for queries where exact term matching is important (e.g., IDs, names, acronyms).

**Architecture:**
```
Query → RAGChain.retrieve(mode="vector|lexical|hybrid")
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    [Chroma]    [BM25Store]  [RRF Fusion]
     vector      lexical       hybrid
        │           │           │
        └───────────┴───────────┘
                    ▼
              RetrievalResult
```

**Search Modes:**
- `vector`: Semantic search using Chroma embeddings (default)
- `lexical`: BM25 keyword search for exact term matching
- `hybrid`: Combines both using Reciprocal Rank Fusion (RRF)

**Index Location:** `data/vectorstore/bm25/{collection_name}.chunks.pkl`

**CLI Usage:**
```bash
# BM25 is enabled by default during indexing
python -m scripts.index_chunks --processed-dir data/processed --chroma-dir data/vectorstore --verbose

# Query BM25 index directly
python -m scripts.query_bm25 --question "WEF Future of Jobs 2025" --k 5 --pretty

# Query with search mode via rag_chat (requires code changes or API)
```

**API Usage:**
```bash
# Vector search (default)
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What skills are in demand?", "k": 5}'

# Lexical search (exact keyword matching)
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "WEF report 2025", "k": 5, "search_mode": "lexical"}'

# Hybrid search (RRF fusion of vector + lexical)
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "skills in demand", "k": 5, "search_mode": "hybrid"}'
```

**When to Use Each Mode:**
- **vector**: General questions, conceptual queries, paraphrased content
- **lexical**: Exact names, IDs, acronyms, technical terms, report titles
- **hybrid**: Best of both worlds, especially for mixed queries

**Programmatic Usage:**
```python
from generation.rag_chain import RAGChain, RAGConfig

# Enable lexical search in config
config = RAGConfig(
    vectorstore_dir=Path("data/vectorstore"),
    enable_lexical=True,  # Required for lexical/hybrid modes
    lexical_weight=0.3,   # Weight for lexical in hybrid mode (0-1)
)

chain = RAGChain(llm_client, config)

# Use different search modes
result = chain.retrieve("query", k=5, mode="vector")
result = chain.retrieve("query", k=5, mode="lexical")
result = chain.retrieve("query", k=5, mode="hybrid")

# Or via query() method
response = chain.query("question", search_mode="hybrid")
```

**BM25 Parameters:**
- `k1` (default: 1.5): Controls term frequency saturation. Higher values give more weight to term frequency.
- `b` (default: 0.75): Controls length normalization. 0 = no normalization, 1 = full normalization.

### Cross-Encoder Reranking

The RAG system supports cross-encoder reranking to improve retrieval relevance. After initial retrieval (vector, lexical, or hybrid), a cross-encoder model scores query-document pairs for better relevance ranking.

**Architecture:**
```
Query → Initial Retrieval (k*3 candidates)
              │
              ▼
      [ScoreNormalizer]
        normalize scores
              │
              ▼
      [RRF Fusion] (hybrid only)
              │
              ▼
      [CrossEncoderReranker]
        rerank top candidates
              │
              ▼
        Top-k Results
```

**Configuration:**
```python
from generation.rag_chain import RAGChain, RAGConfig

config = RAGConfig(
    vectorstore_dir=Path("data/vectorstore"),
    enable_lexical=True,
    # Reranking settings
    enable_reranking=True,                    # Enable by default
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",  # Default model
    reranker_top_k_multiplier=3,              # Fetch k*3 candidates before reranking
    normalize_scores=True,                    # Normalize scores before fusion
)

chain = RAGChain(llm_client, config)

# Reranking is applied automatically when enabled
result = chain.retrieve("query", k=5, mode="hybrid")

# Override per-query
result = chain.retrieve("query", k=5, mode="hybrid", rerank=True)
result = chain.retrieve("query", k=5, mode="hybrid", rerank=False)
```

**API Usage:**
```bash
# Enable reranking for a single query
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "skills in demand", "k": 5, "search_mode": "hybrid", "rerank": true}'

# Response includes reranking_applied in metadata
{
  "answer": "...",
  "sources": [...],
  "metadata": {
    "retrieval_time_ms": 150.5,
    "generation_time_ms": 1200.3,
    "search_mode": "hybrid",
    "reranking_applied": true
  }
}
```

**When to Use Reranking:**
- **Use reranking** when precision is critical (e.g., legal, medical, compliance queries)
- **Skip reranking** for latency-sensitive applications or simple queries
- **Hybrid + reranking** provides the best relevance but highest latency

**Default Reranker Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Trained on MS MARCO passage ranking dataset
- Good balance between speed and accuracy
- ~22M parameters

**Offline Model Download:**
```bash
# Download reranker model for offline use
python -m scripts.download_reranker_model --output-dir models

# The model will be saved to: models/cross-encoder_ms-marco-MiniLM-L-6-v2/
```

**Manual override:** Set environment variable:
```bash
set RERANKER_MODEL_PATH=models/cross-encoder_ms-marco-MiniLM-L-6-v2
```

### Local Embedding Model (Offline/Corporate Environments)

For environments with SSL issues or no internet access, download the model once and use locally:

**Download the model** (from a machine with internet):
```bash
python -m scripts.download_model --output-dir models
```

**The model will be saved to:** `models/sentence-transformers_all-MiniLM-L6-v2/`

**Usage:** The code automatically detects local models in `models/` directory. No configuration changes needed.

**Manual override:** Set environment variable:
```bash
set EMBEDDING_MODEL_PATH=models/sentence-transformers_all-MiniLM-L6-v2
```

### Tiktoken Cache (Offline/Corporate Environments)

The tiktoken library (used for token counting) downloads encoding files at runtime. For environments with SSL issues or no internet access, pre-download the cache:

**Download the cache** (from a machine with internet):
```bash
python -m scripts.download_tiktoken_cache --output-dir models/tiktoken_cache
```

**Download all encodings** (if using multiple models):
```bash
python -m scripts.download_tiktoken_cache --output-dir models/tiktoken_cache --all
```

**Skip SSL verification** (for corporate environments with SSL inspection):
```bash
python -m scripts.download_tiktoken_cache --output-dir models/tiktoken_cache --skip-ssl
```

**The cache will be saved to:** `models/tiktoken_cache/`

**Usage:** The code automatically detects the local cache in `models/tiktoken_cache/` directory. No configuration changes needed.

**Manual override:** Set environment variable:
```bash
set TIKTOKEN_CACHE_DIR=models/tiktoken_cache
```

**Available encodings:**
- `cl100k_base` (default) - Used by GPT-4, GPT-3.5-turbo
- `p50k_base` - Used by older models
- `o200k_base` - Used by GPT-4o

### Legacy DOC File Support

The ingestion pipeline supports legacy Word documents (.doc format, Word 97-2003) with multiple extraction strategies:

**Supported Formats:**
- `.doc` - Microsoft Word 97-2003 (Legacy binary format)
- `.docx` - Microsoft Word 2007+ (Office Open XML) - already supported via python-docx

**Extraction Methods (in order of priority):**

1. **antiword** (Cross-platform, recommended)
   - Command-line tool that extracts text from .doc files
   - Installation:
     - Windows: `choco install antiword` or download from http://antiword.cjb.net/
     - Linux: `apt-get install antiword` or `yum install antiword`
     - macOS: `brew install antiword`

2. **win32com** (Windows only)
   - Uses Microsoft Word via COM automation
   - Requires Microsoft Word to be installed
   - Installation: `pip install pywin32`

**Metadata Extracted:**
- `title`: Document title (or filename fallback)
- `author`: Document author
- `subject`: Document subject
- `keywords`: Document keywords
- `creation_date` / `modification_date`: Timestamps
- `extraction_method`: Which method was used (`antiword` or `win32com`)

**Example Usage:**
```bash
# Ingest DOC files along with other documents
python -m scripts.ingest --input-dir data/raw --output-dir data/processed --verbose

# The pipeline automatically detects and processes .doc files
```

**Error Handling:**
- If no extraction method is available, a helpful error message is displayed with installation instructions
- Corrupted files raise `DocumentParseError`
- Timeout handling for large files (60 second limit for antiword)

### Excel File Support

The ingestion pipeline supports Excel workbooks (.xlsx and .xls formats) via the `openpyxl` library:

**Supported Formats:**
- `.xlsx` - Excel 2010+ (Office Open XML)
- `.xls` - Excel 97-2003 (Legacy format)

**Sheet Handling:**
- All sheets in the workbook are processed sequentially
- Each sheet is prefixed with a `[SHEET:sheet_name]` marker (similar to `[PAGE:N]` for PDFs)
- Sheets are separated by double newlines in the output

**Row Format:**
- Each row is converted to pipe-separated values (e.g., `Col1 | Col2 | Col3`)
- Empty rows are skipped
- Cell values are trimmed of whitespace
- Formulas return their computed values (not the formula text)

**Metadata Extracted:**
- `title`: From document properties or filename fallback
- `author`: Document creator
- `creation_date` / `modification_date`: Timestamps
- `sheet_names`: List of all sheet names in the workbook
- `sheet_count`: Number of sheets
- `total_rows`: Sum of non-empty rows across all sheets
- `total_columns`: Maximum column count across sheets

**Example Usage:**
```bash
# Ingest Excel files along with other documents
python -m scripts.ingest --input-dir data/raw --output-dir data/processed --verbose

# The pipeline automatically detects and processes .xlsx and .xls files
```

**Error Handling:**
- Corrupted files raise `DocumentParseError`
- Password-protected workbooks are detected and reported
- Partial extraction continues if individual sheets fail

### CSV/TSV File Support

The ingestion pipeline supports CSV and TSV files with automatic delimiter detection:

**Supported Formats:**
- `.csv` - Comma-separated values (with automatic delimiter detection)
- `.tsv` - Tab-separated values

**Delimiter Detection:**
- Automatically detects: comma (`,`), semicolon (`;`), tab (`\t`), pipe (`|`)
- Uses consistency scoring across first 10 rows to choose best delimiter
- Falls back to comma if no multi-column delimiter is detected

**Output Format:**
- Content prefixed with `[SHEET:filename]` marker (consistent with Excel format)
- Rows converted to pipe-separated values (e.g., `Col1 | Col2 | Col3`)
- Empty rows are skipped
- Cell values are trimmed of whitespace

**Metadata Extracted:**
- `title`: From filename
- `detected_delimiter`: The delimiter used for parsing
- `total_rows`: Number of non-empty rows
- `total_columns`: Maximum column count
- `sheet_names`: List containing the filename (for consistency with Excel)
- `sheet_count`: Always 1 for CSV files

**Encoding Handling:**
- Primary: UTF-8
- Fallback chain: cp1252 → iso-8859-1 → latin-1
- Encoding used is recorded in metadata if fallback was needed

**Example Usage:**
```bash
# Ingest CSV files along with other documents
python -m scripts.ingest --input-dir data/raw --output-dir data/processed --verbose

# The pipeline automatically detects and processes .csv and .tsv files
```

### Spreadsheet Classification

The pipeline can classify CSV/Excel files as either **tabular** (structured data) or **report-like** (unstructured document) to enable appropriate downstream processing:

**Classification Heuristics:**
- **Row count**: Files with 10+ rows suggest tabular data
- **Column count**: 2-50 columns is typical for tabular data
- **Numeric ratio**: >30% numeric cells suggests tabular
- **Long text ratio**: Cells >200 chars suggest report-like content
- **Empty cell ratio**: Sparse data (>50% empty) suggests report-like

**CLI Usage:**
```bash
# Enable spreadsheet classification
python -m scripts.ingest --input-dir data/raw --output-dir data/processed --classify-spreadsheets

# Customize classification thresholds
python -m scripts.ingest --classify-spreadsheets --min-rows-tabular 5 --numeric-ratio 0.4

# Available options:
#   --classify-spreadsheets     Enable classification
#   --min-rows-tabular N        Minimum rows for tabular (default: 10)
#   --numeric-ratio N           Minimum numeric ratio (default: 0.3)
#   --long-text-threshold N     Character threshold for long text (default: 200)
#   --max-columns-tabular N     Maximum columns for tabular (default: 50)
```

**Classification Metadata:**
When enabled, the following metadata is added to spreadsheet documents:
- `spreadsheet_classification`: Either "tabular" or "report_like"
- `classification_confidence`: Score from 0.0 to 1.0
- `classification_reasons`: List of human-readable reasons
- `classification_metrics`: Computed metrics (row_count, numeric_ratio, etc.)

**Override Patterns:**
Classification can be overridden programmatically using filename patterns:
```python
config = SpreadsheetClassificationConfig(
    override_patterns=[
        {"pattern": "*_report.xlsx", "classification": "report_like"},
        {"pattern": "*_data.csv", "classification": "tabular"},
    ]
)
```

### Spreadsheet Flattening (Report-like)

When a spreadsheet is classified as `report_like`, it can be automatically converted to human-readable text format for better semantic chunking and retrieval:

**Layout Detection:**
The flattener detects three layout types:
- **table**: Regular grid with consistent columns, rendered as Markdown tables
- **form**: Sparse key-value pairs, rendered as prose
- **mixed**: Combination, uses heuristics to choose best format

**Flattening Strategies:**

*Markdown (for table layouts):*
```markdown
## Sheet: Financial Summary

| Category | Q1 | Q2 | Q3 |
|----------|----|----|-----|
| Revenue  | 100| 120| 140 |
```

*Prose (for form/sparse layouts):*
```
Sheet: Executive Summary

Title: Q4 2024 Report
Author: Finance Team

Key Metrics:
- Total Revenue: $1.2M
- Growth Rate: 15%
```

**Special Case Handling:**
- **Footer notes**: Detected and extracted to separate notes section
- **Header rows**: Automatically detected for proper table formatting
- **Empty cells**: Skipped in prose, shown as empty in Markdown tables
- **Large tables**: Truncated to configurable max rows (default: 100)

**Flattening Metadata:**
When flattening is applied, the following metadata is added:
- `flattening_applied`: Boolean indicating if flattening occurred
- `flattening_strategy`: Either "markdown" or "prose"
- `layout_type`: Detected layout ("table", "form", or "mixed")
- `original_row_count`: Total rows before flattening
- `original_col_count`: Total columns before flattening
- `sheets_flattened`: Per-sheet details (name, layout, strategy, row/col counts)

**Configuration:**
Flattening is enabled by default when classification is enabled. It can be configured programmatically:
```python
from ingestion import SpreadsheetFlatteningConfig, PipelineConfig

config = PipelineConfig(
    input_dir=Path("data/raw"),
    output_dir=Path("data/processed"),
    spreadsheet_classification_config=SpreadsheetClassificationConfig(),
    flatten_report_like_spreadsheets=True,  # default: True
    spreadsheet_flattening_config=SpreadsheetFlatteningConfig(
        empty_ratio_threshold_form=0.6,
        max_columns_for_form=4,
        min_rows_for_table=3,
        include_sheet_headers=True,
        max_table_rows_markdown=100,
    ),
)
```

### Text Normalization

The ingestion pipeline includes configurable text normalization applied after document parsing:

**Components:**
- `NormalizationConfig`: Dataclass with all configuration options
- `TextNormalizer`: Applies configured rules to document text
- `NormalizationResult`: Contains normalized text and metadata

**Normalization Order:**
1. Zero-width character removal
2. Special character normalization (smart quotes → ASCII)
3. Bullet character normalization
4. Page number removal
5. Boilerplate removal (confidential, copyright, etc.)
6. Header/footer removal (repeated lines)
7. Custom pattern/replacement application
8. Whitespace normalization

**CLI Usage:**
```bash
# Default normalization (all features enabled)
python -m scripts.ingest --input-dir data/raw --output-dir data/processed

# Minimal (whitespace and special chars only)
python -m scripts.ingest --normalize minimal

# Aggressive (stricter thresholds)
python -m scripts.ingest --normalize aggressive

# Disable normalization
python -m scripts.ingest --normalize none

# Custom YAML config
python -m scripts.ingest --normalize-config config/normalization.yaml

# Disable specific features
python -m scripts.ingest --no-remove-page-numbers --no-remove-boilerplate
```

**Normalization Presets:**
- `default`: All features enabled with balanced thresholds
- `minimal`: Only whitespace/special chars (preserves structure)
- `aggressive`: All features with stricter header/footer detection
- `none`: No normalization applied

### Batch Processing with Failure Tolerance

The ingestion pipeline continues processing on individual document failures and provides detailed failure reporting:

**Output Files:**
- `failures.json`: Structured failure data with error type, message, and full traceback
- `ingestion-report.json`: Run summary with timing, counts, and failure list

**CLI Usage:**
```bash
# Normal batch run (continues on failures)
python -m scripts.ingest --input-dir data/raw --output-dir data/processed

# Retry only previously failed documents
python -m scripts.ingest --retry-failed --output-dir data/processed

# Stop on first failure
python -m scripts.ingest --fail-fast --input-dir data/raw --output-dir data/processed
```

**FailureInfo Model:**
- `source_path`: Path to failed document
- `error_type`: Exception class name
- `error_message`: Error description
- `traceback`: Full stack trace for debugging
- `timestamp`: When the failure occurred

### Dataset Catalog & Summaries

The pipeline can generate LLM-based summaries for tabular datasets, enabling semantic discovery of data sources.

**Architecture:**
```
MetadataCatalog ──────► DatasetSummarizer ──────► SummaryIndexer
      │                        │                        │
 (table schema)          (LLM generate)          (Chroma upsert)
      │                        │                        │
      ▼                        ▼                        ▼
_ingestion_catalog       LLM API call            dataset-summaries
(DuckDB table)           + cost tracking         (Chroma collection)
```

**Generate Summaries:**
```bash
# Generate summaries for all tables without summaries
python -m scripts.generate_summaries --chroma-dir data/vectorstore --verbose

# Generate for specific tables only
python -m scripts.generate_summaries --tables sales_2024 inventory

# Dry run to estimate tokens/cost
python -m scripts.generate_summaries --dry-run

# Set token budget
python -m scripts.generate_summaries --max-tokens 10000

# List tables and summary status
python -m scripts.generate_summaries --list-tables
```

**Query Datasets:**
```bash
# Semantic search for datasets
python -m scripts.query_datasets --question "What sales data do we have?"

# List all indexed datasets
python -m scripts.query_datasets --list-all

# Get details for specific dataset
python -m scripts.query_datasets --describe sales_2024

# JSON output for programmatic use
python -m scripts.query_datasets -q "financial data" --json

# Show catalog and index statistics
python -m scripts.query_datasets --stats
```

**Summary Metadata:**
When summaries are generated, the following is stored:
- `summary`: LLM-generated natural-language description
- `column_descriptions`: Descriptions for key columns
- `summary_generated_at`: Timestamp of generation
- `summary_token_usage`: Token counts for cost tracking

**Cost Controls:**
- Default token budget: 50,000 tokens per run
- Budget enforcement stops processing when limit is reached
- Use `--dry-run` to estimate costs before generating

**Chroma Collection:**
Summaries are indexed in a separate `dataset-summaries` collection for semantic search.
- Document format: Summary text + column list
- Metadata includes: table_name, source_file, row_count, domain_labels

### REST API

The REST API provides a UI-agnostic interface to the RAG system. It is designed to work with any frontend framework or HTTP client.

**Architecture Principles:**
- **UI Independence**: The API has no dependencies on any UI framework
- **Standard REST**: JSON request/response format, standard HTTP methods
- **CORS Enabled**: Allows requests from any origin (configurable)
- **OpenAPI Documentation**: Auto-generated Swagger UI at `/docs`

**Start the Server:**
```bash
# Development mode with auto-reload
python -m uvicorn api.main:app --reload

# Production mode
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080
```

**Environment Variables:**
```bash
# Override default paths
PROCESSED_DIR=data/processed
VECTORSTORE_DIR=data/vectorstore
RAW_DIR=data/raw
COLLECTION_NAME=pilot-docs

# LLM provider (see generation/ docs)
LLM_PROVIDER=openai  # or: custom, anthropic, ollama
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info and version |
| GET | `/api/health` | Health check (vectorstore + LLM) |
| GET | `/api/documents` | List all ingested documents |
| POST | `/api/query` | Query RAG with a question |
| POST | `/api/ingest` | Upload and ingest a document |

**Query Endpoint:**
```bash
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What skills are in demand?", "k": 5, "search_mode": "hybrid"}'
```

Request parameters:
- `question` (required): The question to ask
- `k` (optional, default: 5): Number of chunks to retrieve
- `agent_id` (optional): Agent ID for custom prompts
- `search_mode` (optional, default: "vector"): Search mode - "vector", "lexical", or "hybrid"

Response:
```json
{
  "answer": "The most in-demand skills include...",
  "sources": [
    {
      "doc_id": "report-pdf",
      "filename": "report.pdf",
      "chunk_id": "report-pdf::chunk-0001",
      "snippet": "First 200 chars of chunk...",
      "page": 5
    }
  ],
  "metadata": {
    "retrieval_time_ms": 45.2,
    "generation_time_ms": 1523.8,
    "search_mode": "hybrid"
  }
}
```

**Ingest Endpoint:**
```bash
curl -X POST http://localhost:8080/api/ingest \
  -F "file=@document.pdf"
```

Response:
```json
{
  "doc_id": "document-pdf",
  "filename": "document.pdf",
  "status": "success",
  "chunks_created": 42,
  "message": "Document ingested and indexed successfully"
}
```

**Health Endpoint:**
```bash
curl http://localhost:8080/api/health
```

Response:
```json
{
  "status": "healthy",
  "vectorstore": {
    "healthy": true,
    "message": "Chroma healthy. Collection 'pilot-docs' has 590 documents.",
    "collection_count": 2,
    "document_count": 590
  },
  "llm_provider": {
    "healthy": true,
    "message": "OpenAI configured",
    "provider": "openai"
  },
  "bm25_index": {
    "healthy": true,
    "message": "BM25 index healthy. 590 chunks indexed.",
    "document_count": 590,
    "index_name": "pilot-docs"
  }
}
```

**Documents Endpoint:**
```bash
curl http://localhost:8080/api/documents
```

Response:
```json
{
  "documents": [
    {
      "doc_id": "report-pdf",
      "filename": "report.pdf",
      "file_type": "pdf",
      "chunks": 417,
      "ingested_at": "2026-01-07T16:29:52.977785Z"
    }
  ],
  "total": 4
}
```

**Error Responses:**
All errors follow a consistent format:
```json
{
  "detail": "Error message describing what went wrong",
  "error_type": "ExceptionClassName"
}
```

HTTP Status Codes:
- `400`: Validation error (invalid request)
- `404`: Resource not found
- `500`: Internal server error
- `503`: Service unavailable (RAG not initialized)

### SQL Agent

The SQL Agent allows natural language queries against tabular data stored in DuckDB. It translates questions into SQL, executes them with SELECT-only guardrails, and returns results with natural language summaries.

**Architecture:**
```
User Question → SQLChain.query()
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  SchemaExtractor          SQLGenerator
  (get table schemas)      (LLM → SQL)
        │                       │
        └───────────┬───────────┘
                    ▼
              QueryValidator
              (SELECT-only check)
                    │
                    ▼
              DuckDB Execute
                    │
                    ▼
              Answer Generator
              (LLM summarize results)
```

**Safety Guardrails:**
- **Read-only mode**: Database is always opened in read-only mode
- **SELECT-only**: Only SELECT and WITH (CTEs) queries are allowed
- **Blocked operations**: INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE, SET, GRANT, etc.
- **Excluded tables**: System tables (`_agents`, `_ingestion_catalog`) are blocked
- **No SQL comments**: `--` and `/* */` comments are rejected
- **No file operations**: `read_csv()`, `read_parquet()`, etc. are blocked
- **Auto-LIMIT**: Queries without LIMIT automatically get `LIMIT 100`

**CLI Usage:**
```bash
# Single question
python -m scripts.sql_chat -q "What tables are available?"

# Interactive mode
python -m scripts.sql_chat --interactive

# Show generated SQL
python -m scripts.sql_chat -q "Top 5 products by revenue" --show-sql

# JSON output
python -m scripts.sql_chat -q "Count by category" --json

# List available tables
python -m scripts.sql_chat --list-tables

# Custom database path
python -m scripts.sql_chat -q "List tables" --db-path data/vectorstore/catalog.duckdb
```

**API Usage:**
```bash
# Query with natural language
curl -X POST http://localhost:8080/api/sql-query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the top 5 products by revenue?", "max_rows": 100}'

# Show generated SQL in response
curl -X POST http://localhost:8080/api/sql-query \
  -H "Content-Type: application/json" \
  -d '{"question": "Count products by category", "show_sql": true}'

# List available tables
curl http://localhost:8080/api/sql-tables
```

**API Response:**
```json
{
  "answer": "The top 5 products by revenue are...",
  "data": [
    {"product": "Widget", "revenue": 10000},
    {"product": "Gadget", "revenue": 8000}
  ],
  "columns": ["product", "revenue"],
  "row_count": 5,
  "generated_sql": "SELECT product, SUM(revenue) as revenue FROM sales GROUP BY product ORDER BY revenue DESC LIMIT 5",
  "metadata": {
    "generation_time_ms": 450.2,
    "execution_time_ms": 12.5,
    "summarization_time_ms": 800.0,
    "tables_used": ["sales"]
  }
}
```

**Error Handling:**
| Exception | HTTP Status | Scenario |
|-----------|-------------|----------|
| `QueryValidationError` | 400 | Non-SELECT query, forbidden operations, excluded tables |
| `QueryGenerationError` | 400 | LLM cannot answer, invalid SQL syntax |
| `QueryExecutionError` | 500 | DuckDB execution error |
| `HTTPException(503)` | 503 | SQL agent not initialized |

**Health Check:**
The `/api/health` endpoint includes SQL Agent status:
```json
{
  "sql_agent": {
    "healthy": true,
    "message": "SQL Agent healthy. 5 tables, 12345 total rows.",
    "table_count": 5,
    "total_rows": 12345
  }
}
```

**Configuration:**
```python
from sql_agent import SQLAgentConfig, SQLChain

config = SQLAgentConfig(
    db_path=Path("data/vectorstore/catalog.duckdb"),
    read_only=True,                    # Always forced to True
    excluded_tables=["_agents", "_ingestion_catalog"],
    max_rows_preview=3,                # Sample rows in schema prompt
    max_result_rows=100,               # Max rows returned
    query_timeout=30,                  # Execution timeout (seconds)
    max_tokens=500,                    # LLM response tokens
    temperature=0.0,                   # Deterministic SQL generation
    auto_limit=True,                   # Auto-add LIMIT if missing
    default_limit=100,                 # Default LIMIT value
)

chain = SQLChain(llm_client, config)
result = chain.query("What tables are available?")
```

### Migration Notes

#### BM25 Lexical Search (PR #123)
- **New feature**: BM25 lexical index created alongside Chroma during indexing
- **Backward compatibility**: Existing Chroma indexes continue to work; BM25 is additive
- **To enable BM25**: Re-run the indexing pipeline to create the BM25 index:
  ```bash
  python -m scripts.index_chunks --processed-dir data/processed --chroma-dir data/vectorstore --verbose
  ```
- **API change**: New `search_mode` parameter in `/api/query` endpoint (defaults to "vector" for backward compatibility)
- **New dependency**: `rank-bm25>=0.2.2` added to requirements.txt

#### Chunk Metadata Indexing (PR #48)
- **New metadata fields**: `page`, `section`, `timestamp` added to indexed chunks
- **Backward compatibility**: Existing chunks will not have `page`/`section` metadata until documents are re-ingested
- **Recommendation**: Re-run ingestion and indexing pipelines to populate new metadata for all documents:
  ```bash
  python -m scripts.ingest --input-dir data/raw --output-dir data/processed --verbose
  python -m scripts.index_chunks --processed-dir data/processed --chroma-dir data/vectorstore --verbose
  ```
