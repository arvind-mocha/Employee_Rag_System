import type { EvaluationResult } from '../types/api'

export function evaluationCaseStatus(result: Pick<EvaluationResult, 'error' | 'passed'>): string {
  if (result.error) return 'Errored'
  if (result.passed === true) return 'Passed'
  if (result.passed === false) return 'Failed'
  return 'Unavailable'
}

export function evaluationErrorLabel(result: Pick<EvaluationResult, 'error_category'>): string {
  return (result.error_category || 'Evaluation error').replaceAll('_', ' ')
}
