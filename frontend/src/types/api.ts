export type Provider = 'huggingface' | 'openai' | 'anthropic' | 'grok'

export interface ProviderCredential {
  provider: Provider
  model_name: string
  key_hint: string
  expires_at: string
  created_at: string
  updated_at: string
}

export interface ProviderCredentialListResponse {
  credentials: ProviderCredential[]
}

export interface ProviderCredentialRequest {
  provider: Provider
  model_name: string
  api_key: string
  expires_in_hours: number
}

export interface DataUploadResponse {
  message: string
  total_files: number
  structured_files: string[]
  unstructured_files: string[]
}

export type IngestionState =
  | 'idle'
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'

export interface IngestionStatusResponse {
  status: IngestionState
  error: string | null
  started_at: string | null
  finished_at: string | null
}

export interface IngestionStartResponse {
  message: string
  status: 'accepted'
  status_url: string
}

export interface RetrievalOptions {
  semantic_search: boolean
  keyword_search: boolean
  query_rewriting: boolean
  reranking: boolean
  source_diversity: boolean
}

export interface RetrievedChunk {
  content: string
  relevance_score: number
  chunk_id: string | null
  source_file: string | null
  source_type: string | null
  section: string | null
  chunk_index: number | null
}

export interface LlmUsage {
  input_tokens: number
  cached_input_tokens: number
  cache_creation_input_tokens: number
  output_tokens: number
  total_tokens: number
  token_count_source: 'provider' | 'estimated'
  input_cost_per_million_usd: number | null
  cached_input_cost_per_million_usd: number | null
  cache_creation_cost_per_million_usd: number | null
  output_cost_per_million_usd: number | null
  estimated_cost_usd: number | null
  pricing_source: 'built_in' | 'unavailable'
  currency: 'USD'
}

export interface LlmUsageSummary {
  llm_calls: number
  input_tokens: number
  cached_input_tokens: number
  cache_creation_input_tokens: number
  output_tokens: number
  total_tokens: number
  priced_calls: number
  unpriced_calls: number
  estimated_cost_usd: number | null
  currency: 'USD'
}

export interface AnswerRequest extends RetrievalOptions {
  question: string
  hf: boolean
  openai: boolean
  anthropic: boolean
  grok: boolean
}

export interface AnswerResponse extends RetrievalOptions {
  question: string
  effective_query: string
  applied_strategies: string[]
  retrieval_used: boolean
  provider: string
  model: string
  answer: string
  usage: LlmUsage | null
  sources: RetrievedChunk[]
}

export interface ScoreAverages {
  precision_at_k: number | null
  recall_at_k: number | null
  reciprocal_rank: number | null
  average_precision_at_k: number | null
  ndcg_at_k: number | null
  source_precision_at_k: number | null
  source_recall_at_k: number | null
  correctness: number | null
  groundedness: number | null
  completeness: number | null
  relevance: number | null
  citation_accuracy: number | null
  citation_coverage: number | null
  citation_validity: number | null
  abstention_correctness: number | null
  lexical_similarity: number | null
  semantic_similarity: number | null
  overall: number | null
}

export interface AnswerScores {
  correctness: number
  groundedness: number
  completeness: number
  relevance: number
  citation_accuracy: number
  citation_coverage: number
  citation_validity: number
  abstention_correctness: number
  lexical_similarity: number
  semantic_similarity: number | null
  overall: number
}

export interface RetrievalScores {
  precision_at_k: number
  recall_at_k: number
  reciprocal_rank: number
  average_precision_at_k: number
  ndcg_at_k: number
  source_precision_at_k: number | null
  source_recall_at_k: number | null
}

export interface EvaluationResult {
  record_id: string
  question: string
  reference_answer: string
  required_facts: string[]
  review_status: string
  ground_truth_status: string
  answer_provider: Provider
  answer_model: string | null
  judge_provider: Provider
  judge_model: string | null
  answer: string | null
  usage: LlmUsageSummary
  retrieved_chunk_ids: string[]
  retrieved_sources: string[]
  retrieval_scores: RetrievalScores | null
  answer_scores: AnswerScores | null
  unsupported_claims: string[]
  missing_facts: string[]
  rationale: string | null
  passed: boolean | null
  timing: {
    retrieval_ms: number
    answer_generation_ms: number
    grading_ms: number
    total_ms: number
  }
  warnings: string[]
  error: string | null
}

export interface ProviderSummary {
  provider: Provider
  evaluated: number
  passed: number
  failed: number
  errors: number
  stale_ground_truth: number
  unreviewed_ground_truth: number
  pass_rate: number | null
  average_scores: ScoreAverages
  average_latency_ms: number | null
  usage: LlmUsageSummary
}

export interface EvaluationRun {
  run_id: string
  dataset_name: string
  dataset_sha256: string
  generated_at: string
  judge_provider: Provider
  usage: LlmUsageSummary
  summaries: ProviderSummary[]
  results: EvaluationResult[]
}

export interface UploadEvaluationDatasetResponse {
  dataset_name: string
  dataset_file: string
  dataset_sha256: string
  record_count: number
  review_required: boolean
}

export interface EvaluationJobAccepted {
  job_id: string
  job_type: 'dataset_generation' | 'evaluation_run'
  status: 'queued'
  status_url: string
}

export interface EvaluationJobStatus {
  job_id: string
  job_type: 'dataset_generation' | 'evaluation_run'
  status:
    | 'queued'
    | 'running'
    | 'completed'
    | 'failed'
    | 'deferred'
    | 'scheduled'
    | 'stopped'
    | 'canceled'
  result: Record<string, unknown> | null
  error: string | null
}
