export const EVALUATION_CONCURRENCY_OPTIONS = [
  { value: 1, label: '1 — Sequential' },
  { value: 2, label: '2 — Low concurrency' },
  { value: 3, label: '3 — Balanced' },
  { value: 4, label: '4 — Parallel (Default)' },
  { value: 6, label: '6 — High concurrency' },
  { value: 8, label: '8 — Maximum concurrency' },
] as const

export function validEvaluationConcurrency(value: number): boolean {
  return Number.isInteger(value) && EVALUATION_CONCURRENCY_OPTIONS.some((option) => option.value === value)
}
