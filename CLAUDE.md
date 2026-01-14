# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG MVP - A Python-based Retrieval Augmented Generation system for semantic document search. The pipeline ingests documents, chunks them, generates embeddings, and stores them in a Chroma vector database for retrieval.

## Commands

### Setup
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
    → [indexing]   → data/vectorstore/ (Chroma)
    → [query]      → Top-k semantic matches
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
- `embeddings.py`: `EmbeddingService` with retry logic, `EmbeddingConfig`, `EmbeddingError`
- `dataset.py`: Loads chunks from manifest/JSONL files
- `pipeline.py`: `ChromaIndexingPipeline` handles batch embedding and upsert with failure tolerance
- `summary_indexer.py`: `SummaryIndexer` for indexing dataset summaries in Chroma

**generation/** - LLM-powered response generation
- `api_client.py`: HMAC-authenticated LLM client with `LLMClient`, `LLMConfig`
- `rag_chain.py`: `RAGChain` combines retrieval and generation
- `dataset_summarizer.py`: `DatasetSummarizer` generates LLM summaries for tabular datasets
- `cost_tracker.py`: `CostTracker` for token usage tracking and budget controls

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

### Migration Notes

#### Chunk Metadata Indexing (PR #48)
- **New metadata fields**: `page`, `section`, `timestamp` added to indexed chunks
- **Backward compatibility**: Existing chunks will not have `page`/`section` metadata until documents are re-ingested
- **Recommendation**: Re-run ingestion and indexing pipelines to populate new metadata for all documents:
  ```bash
  python -m scripts.ingest --input-dir data/raw --output-dir data/processed --verbose
  python -m scripts.index_chunks --processed-dir data/processed --chroma-dir data/vectorstore --verbose
  ```
