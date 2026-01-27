/**
 * TypeScript interfaces matching api/schemas.py
 */

/** Request model for RAG query endpoint */
export interface QueryRequest {
  question: string;
  k?: number;
  agent_id?: string;
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

/** Request model for creating an agent */
export interface AgentCreate {
  name: string;
  description?: string;
  system_prompt: string;
  temperature?: number | null;
  max_tokens?: number | null;
}

/** Request model for updating an agent */
export interface AgentUpdate {
  name?: string;
  description?: string;
  system_prompt?: string;
  temperature?: number | null;
  max_tokens?: number | null;
}

/** Response model for agent data */
export interface AgentResponse {
  id: string;
  name: string;
  description: string | null;
  system_prompt: string;
  temperature: number | null;
  max_tokens: number | null;
  is_default: boolean;
  created_at: string;
  updated_at: string;
}

/** Response model for agents list endpoint */
export interface AgentsListResponse {
  agents: AgentResponse[];
  total: number;
}

/** Response model for agent deletion */
export interface AgentDeleteResponse {
  deleted: boolean;
}
