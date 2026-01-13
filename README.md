# RAG MVP - "Chat With Our Documents"

This epic delivers a usable retrieval augmented generation (RAG) MVP that lets users chat over a limited pilot corpus. The scope covers ingestion, indexing, retrieval, a minimal UI, and an evaluation loop. Target delivery is **end of Week 8**.

## Feature Breakdown

### 1. Ingestion Pipeline v1 (Parse + Heuristic Chunking)
- Batch job parses the pilot corpus (PDF, DOCX, Excel, Markdown) into normalized text with metadata.
- Simple heuristic chunking (~300-500 tokens) emits chunks with `doc_id` plus source references.
- Failures log without aborting the run and reruns remain idempotent (no duplicate records).

#### Implementation snapshot
- Python ingestion package under `ingestion/` handles parsing, normalization, chunking, and persistence.
- Document loaders support PDF (`PyPDF2`), DOCX (`python-docx`), Excel (`openpyxl`), Markdown, and plain text. Each run computes a content hash so unchanged documents are skipped automatically.
- Heuristic chunker groups paragraphs into ~400-token windows with ~80-token overlap to preserve context continuity.
- Outputs are written to `data/processed/chunks/<doc_id>.jsonl` plus a `manifest.json` summarizing each document's metadata and hash, guaranteeing idempotent re-runs.

#### How to run it
1. `python -m venv .venv && .venv\Scripts\activate` (or activate via your shell of choice).
2. `pip install -r requirements.txt`.
3. Drop pilot documents under `data/raw/` (any subfolder structure is fine).
4. `python -m scripts.ingest --input-dir data/raw --output-dir data/processed` (you can tune `--chunk-size` / `--chunk-overlap`).
5. Inspect `data/processed/chunks/*.jsonl` and `data/processed/manifest.json` for chunk outputs and metadata snapshots.

| Attribute | Value |
| --- | --- |
| Estimated Effort | 6 dev-days |
| Confidence | High |
| Dependencies | E1 complete; pilot documents approved |
| Priority | High |

### 2. Chroma Setup & Vector Indexing
- Persistent Chroma instance with a defined collection schema.
- Embeddings generated for every chunk plus metadata (doc_id, source path or URI, section or page if known, timestamps, ACL placeholder fields).
- Re-indexing supported without duplicate entries; smoke-tested top-k retrieval returns relevant context.

#### Implementation snapshot
- `indexing/` package keeps the indexing pipeline modular: chunk dataset loader, SentenceTransformer embedding factory (`all-MiniLM-L6-v2` by default), and a Chroma-specific pipeline that handles per-document deletes plus batch upserts.
- Vector store persists under `data/vectorstore/` using a configurable collection name (default `pilot-docs`) so it survives container restarts.
- Chunk metadata stored with each vector includes source path, chunk/doc IDs, paragraph ranges, hashes, and timestamps pulled from the ingestion manifest, which makes retrieval/auditing straightforward.
- CLI `python -m scripts.index_chunks` orchestrates the flow end-to-end and accepts overrides for processed dir, Chroma dir, collection name, embedding model, batch size, and doc filters to support idempotent re-indexing.

#### How to run it
1. Ensure ingestion has produced `data/processed/manifest.json` and `data/processed/chunks/*.jsonl`.
2. `pip install -r requirements.txt` (installs Chroma + sentence-transformers).
3. `python -m scripts.index_chunks --processed-dir data/processed --chroma-dir data/vectorstore --collection-name pilot-docs --verbose`.
   - Re-index only certain docs via `--doc-ids microsoft-annual-report-2022-pdf`.
   - Swap embedding models via `--embedding-model sentence-transformers/all-mpnet-base-v2` if you need better recall.
4. Quick CLI retrieval test:
   ```
   python -m scripts.query_chunks --question "How did Microsoft describe FY22 results?" --k 2 --max-chars 300
   ```
   Add `--pretty` to dump pure JSON if you want to post-process programmatically.
5. Optional Python smoke test:

```python
python - <<'PY'
import chromadb
client = chromadb.PersistentClient(path="data/vectorstore")
collection = client.get_collection("pilot-docs")
results = collection.query(query_texts=["How did Microsoft describe FY22 results?"], n_results=2)
print(results["documents"][0][0][:500])
print(results["metadatas"][0][0]["relative_path"], results["ids"][0][0])
PY
```

| Attribute | Value |
| --- | --- |
| Estimated Effort | 5 dev-days |
| Confidence | High |
| Dependencies | Ingestion pipeline v1; embeddings working |
| Priority | High |

### 3. Simple RAG Chain + REST API
- LangChain-based RAG chain: Chroma retriever -> grounding prompt ("use only provided context or say you don't know") -> LLM.
- REST endpoint `POST /rag/query` accepts the user question (plus optional context) and returns answer, cited sources (doc_id + snippet or offset), and retrieval metadata.
- Integration tests cover representative pilot queries.

#### Implementation snapshot
- `generation/` package provides RAG chain functionality: HMAC-authenticated LLM client (`api_client.py`) and retrieval + generation logic (`rag_chain.py`).
- `RAGChain` retrieves top-k chunks from Chroma, formats them as context, and sends to the LLM API with a grounding prompt that instructs the model to answer only from provided context.
- CLI `python -m scripts.rag_chat` supports single questions, interactive chat mode, and JSON output.

#### Setup
1. Create a `.env` file in the project root with your LLM API credentials:
   ```
   API_KEY=your_api_key
   API_SECRET=your_api_secret
   BASE_URL=https://your-llm-api-endpoint
   ```

#### How to run it
1. Ensure indexing has completed (`data/vectorstore/` exists with indexed chunks).
2. `pip install -r requirements.txt` (includes `requests` and `python-dotenv`).
3. Single question:
   ```bash
   python -m scripts.rag_chat --question "What are the key job trends for 2025?" --k 5
   ```
4. Interactive chat mode:
   ```bash
   python -m scripts.rag_chat --interactive
   ```
5. Show retrieved sources with the answer:
   ```bash
   python -m scripts.rag_chat -q "What skills are most in demand?" --show-sources
   ```
6. JSON output for programmatic use:
   ```bash
   python -m scripts.rag_chat -q "Summarize the main findings" --json
   ```

#### CLI Options
| Option | Description | Default |
| --- | --- | --- |
| `--question`, `-q` | Question to ask (omit for interactive mode) | - |
| `--interactive`, `-i` | Run in interactive chat mode | False |
| `--k` | Number of chunks to retrieve | 5 |
| `--max-tokens` | Maximum tokens in LLM response | 1000 |
| `--temperature` | LLM temperature (0.0-1.0) | 0.7 |
| `--show-sources` | Display retrieved source chunks | False |
| `--json` | Output as JSON (single question only) | False |
| `--vectorstore-dir` | Chroma database directory | data/vectorstore |
| `--collection-name` | Chroma collection name | pilot-docs |

#### Testing the RAG chain
```bash
# Quick test with the demo PDF (WEF Future of Jobs Report 2025)
python -m scripts.rag_chat -q "What are the most in-demand skills according to the report?" --show-sources

# Verify retrieval is working
python -m scripts.query_chunks --question "skills demand 2025" --k 3 --pretty
```

| Attribute | Value |
| --- | --- |
| Estimated Effort | 7 dev-days |
| Confidence | Medium |
| Dependencies | Chroma indexing; LLM baseline; auth/config baseline |
| Priority | High |

### 4. Minimal UI + Evaluation Set
- Lightweight UI or client for internal stakeholders providing Q&A, citation display, and basic feedback (thumbs up/down + comment).
- 20-30 evaluation questions mapped to expected source docs; manual evaluation produces a short report on failure modes and priority fixes.
- Release notes document known limitations of the MVP.

| Attribute | Value |
| --- | --- |
| Estimated Effort | 4 dev-days |
| Confidence | Medium |
| Dependencies | REST API available; pilot users identified |
| Priority | Medium |

## Summary Table

| Feature | Effort | Confidence | Priority | Dependencies |
| --- | --- | --- | --- | --- |
| Ingestion Pipeline v1 | 6 dev-days | High | High | E1 complete; pilot documents available |
| Chroma Setup & Vector Indexing | 5 dev-days | High | High | Ingestion pipeline v1; embeddings working |
| Simple RAG Chain + REST API | 7 dev-days | Medium | High | Chroma indexing; LLM baseline; auth/config |
| Minimal UI + Evaluation Set | 4 dev-days | Medium | Medium | REST API available; pilot users identified |

**Total Estimated Effort:** 22 dev-days.
