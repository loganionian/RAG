"""Document ingestion endpoint."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from api.schemas import IngestResponse
from indexing.pipeline import ChromaIndexingPipeline, IndexingConfig
from ingestion.pipeline import IngestionPipeline, PipelineConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ingest"])

# Default paths (can be overridden via environment variables)
RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "data/processed"))
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "data/vectorstore"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pilot-docs")

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".xlsx", ".xls", ".csv", ".tsv", ".md", ".txt"}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile) -> IngestResponse:
    """Ingest a document into the RAG system.

    Accepts a file upload, saves it to the raw directory, runs the ingestion
    pipeline to create chunks, and indexes the chunks in the vectorstore.
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided",
        )

    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # Ensure raw directory exists
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    file_path = RAW_DIR / file.filename
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("Saved uploaded file: %s", file_path)
    except Exception as e:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {e}",
        ) from e
    finally:
        await file.close()

    # Run ingestion pipeline for this specific file
    try:
        ingestion_config = PipelineConfig(
            input_dir=RAW_DIR,
            output_dir=PROCESSED_DIR,
            fail_fast=True,
        )
        ingestion_pipeline = IngestionPipeline(ingestion_config)
        ingestion_result = ingestion_pipeline.run(document_paths=[file_path])

        if ingestion_result.failed > 0:
            # Get failure details
            failure_msg = "Unknown error"
            if ingestion_result.failures:
                failure_msg = ingestion_result.failures[0].error_message
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion failed: {failure_msg}",
            )

        if ingestion_result.processed == 0:
            return IngestResponse(
                doc_id="",
                filename=file.filename,
                status="skipped",
                chunks_created=0,
                message="Document was skipped (no changes detected or no content)",
            )

        # Get doc_id from manifest
        storage = ingestion_pipeline.storage
        manifest = storage._manifest

        # Find the doc_id for this file
        doc_id = None
        for did, entry in manifest.items():
            source_path = entry.get("source_path", "")
            if Path(source_path).resolve() == file_path.resolve():
                doc_id = did
                break

        if not doc_id:
            raise HTTPException(
                status_code=500,
                detail="Document was processed but not found in manifest",
            )

        chunks_created = ingestion_result.chunk_count

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ingestion pipeline failed")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {e}",
        ) from e

    # Run indexing pipeline for this specific document
    try:
        indexing_config = IndexingConfig(
            processed_dir=PROCESSED_DIR,
            chroma_dir=VECTORSTORE_DIR,
            collection_name=COLLECTION_NAME,
            doc_filter=[doc_id],
        )
        indexing_pipeline = ChromaIndexingPipeline(indexing_config)
        indexing_result = indexing_pipeline.run()

        if indexing_result.indexed_docs == 0:
            logger.warning("Document %s was not indexed", doc_id)

        logger.info(
            "Indexed document %s: %d chunks",
            doc_id,
            indexing_result.indexed_chunks,
        )

    except Exception as e:
        logger.exception("Indexing pipeline failed")
        # Document was ingested but not indexed - return partial success
        return IngestResponse(
            doc_id=doc_id,
            filename=file.filename,
            status="partial",
            chunks_created=chunks_created,
            message=f"Ingested but indexing failed: {e}",
        )

    return IngestResponse(
        doc_id=doc_id,
        filename=file.filename,
        status="success",
        chunks_created=chunks_created,
        message=f"Document ingested and indexed successfully",
    )
