import type { EvaluationResult } from '../types/api'

export function EvaluationErrorDetails({ result }: { result: EvaluationResult }) {
  if (!result.error) return null
  return (
    <section className="case-warning-block" role="alert">
      <strong>{(result.error_category || 'Evaluation error').replaceAll('_', ' ')}</strong>
      <p>{result.error}</p>
      <p>Stage: {result.failed_stage || 'Unavailable for this older result'}</p>
      <p>Provider: {result.error_provider || result.answer_provider} · Model: {result.error_model || result.answer_model || 'Unavailable'}</p>
      <p>Dataset: {result.dataset_id} · Record: {result.record_id}</p>
      <p>Run: {result.run_id || 'Unavailable'} · Job: {result.job_id || 'Synchronous run'}</p>
    </section>
  )
}
