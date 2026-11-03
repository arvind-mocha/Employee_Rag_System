import { EVALUATION_CONCURRENCY_OPTIONS, validEvaluationConcurrency } from '../lib/evaluationConcurrency'

export function ParallelEvaluationSettings({ value, onChange, disabled }: {
  value: number
  onChange: (value: number) => void
  disabled: boolean
}) {
  return (
    <div>
      <label htmlFor="evaluation-concurrency">Parallel Evaluation</label>
      <select id="evaluation-concurrency" className="select-field" value={value} disabled={disabled}
        aria-describedby="evaluation-concurrency-help" onChange={(event) => {
          const selected = Number(event.target.value)
          if (validEvaluationConcurrency(selected)) onChange(selected)
        }}>
        {EVALUATION_CONCURRENCY_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
      </select>
      <small id="evaluation-concurrency-help" className="field-help">
        Higher concurrency increases instantaneous API requests and may increase rate-limit errors. Provider limits can reduce effective parallelism.
      </small>
    </div>
  )
}
