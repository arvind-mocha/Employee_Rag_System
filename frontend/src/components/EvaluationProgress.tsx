import type { EvaluationJobStatus } from '../types/api'

export function EvaluationProgress({ job, onCancel, cancelPending }: {
  job: EvaluationJobStatus
  onCancel: () => void
  cancelPending: boolean
}) {
  const total = job.total_cases
  const completed = job.completed_cases
  const terminal = ['completed', 'failed', 'stopped', 'canceled'].includes(job.status)
  const percent = total ? (100 * completed / total).toFixed(1) : '0.0'
  const elapsed = Math.floor(job.elapsed_seconds ?? 0)
  return (
    <div className="evaluation-progress" role="status" aria-live="polite">
      <progress value={completed} max={Math.max(total, 1)} aria-label="Evaluation progress" />
      <p>Total: {total} · Completed: {completed} · Running: {job.running_cases ?? 0} · Successful: {job.successful_cases ?? 0} · Failed (errors): {job.failed_cases ?? 0} · Progress: {percent}% · Elapsed: {Math.floor(elapsed / 60)}m {elapsed % 60}s</p>
      <small>Successful means scored; {job.low_scoring_cases ?? 0} scored cases are below the pass threshold. Retries: {job.retry_count ?? 0}.</small>
      {!terminal && <button type="button" className="text-button" onClick={onCancel}
        disabled={cancelPending || job.cancellation_requested}>
        {job.cancellation_requested ? 'Canceling evaluation…' : 'Cancel evaluation'}
      </button>}
    </div>
  )
}
