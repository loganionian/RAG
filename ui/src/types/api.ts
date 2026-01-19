/**
 * TypeScript interfaces matching api/schemas.py
 */

/** Request model for RAG query endpoint */
export interface QueryRequest {
  question: string;
  k?: number;
}

/** Information about a retrieved source chunk */
export interface SourceInfo {
  doc_id: string;
  filename: string;
  chunk_id: string;
  snippet: string;
  page: number | null;
}

/** Metadata about query execution */
export interface QueryMetadata {
  retrieval_time_ms: number;
  generation_time_ms: number;
}

/** Response model for RAG query endpoint */
export interface QueryResponse {
  answer: string;
  sources: SourceInfo[];
  metadata: QueryMetadata;
}

/** Response model for document ingestion endpoint */
export interface IngestResponse {
  doc_id: string;
  filename: string;
  status: string;
  chunks_created: number;
  message: string | null;
}

/** Information about an ingested document */
export interface DocumentInfo {
  doc_id: string;
  filename: string;
  file_type: string;
  chunks: number;
  ingested_at: string;
}

/** Response model for documents list endpoint */
export interface DocumentsResponse {
  documents: DocumentInfo[];
  total: number;
}

/** Vectorstore health details */
export interface VectorstoreHealth {
  healthy: boolean;
  message: string;
  collection_count?: number;
  document_count?: number;
}

/** LLM provider health details */
export interface LLMProviderHealth {
  healthy: boolean;
  message: string;
  provider?: string;
}

/** Response model for health check endpoint */
export interface HealthResponse {
  status: string;
  vectorstore: VectorstoreHealth;
  llm_provider: LLMProviderHealth;
}

/** Standard error response format */
export interface ErrorResponse {
  detail: string;
  error_type: string;
}
