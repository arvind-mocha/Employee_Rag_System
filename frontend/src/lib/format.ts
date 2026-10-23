import type { EvaluationResult, Provider } from '../types/api'

export const PROVIDERS: Array<{
  id: Provider
  name: string
  shortName: string
  description: string
}> = [
  {
    id: 'huggingface',
    name: 'Hugging Face',
    shortName: 'HF',
    description: 'Default · lowest cost',
  },
  { id: 'openai', name: 'OpenAI', shortName: 'OA', description: 'OpenAI model' },
  {
    id: 'anthropic',
    name: 'Anthropic',
    shortName: 'AN',
    description: 'Claude model',
  },
  { id: 'grok', name: 'Grok', shortName: 'GX', description: 'xAI model' },
]

export const PROVIDER_COLORS: Record<Provider, string> = {
  huggingface: '#21675c',
  openai: '#db7c41',
  anthropic: '#6d5fa8',
  grok: '#2975a8',
}

export const DEFAULT_RETRIEVAL = {
  semantic_search: true,
  keyword_search: true,
  query_rewriting: false,
  reranking: false,
  source_diversity: false,
}

export const CATEGORIES = [
  'Leave & Attendance',
  'Benefits & Compensation',
  'Policies & Compliance',
  'Roles & Employee Information',
  'Company, Culture & Careers',
] as const

export type EvaluationCategory = (typeof CATEGORIES)[number]

const CATEGORY_TERMS: Array<[EvaluationCategory, string[]]> = [
  [
    'Leave & Attendance',
    ['vacation', 'leave', 'holiday', 'attendance', 'sick', 'pto', 'time off', 'absence'],
  ],
  [
    'Benefits & Compensation',
    ['benefit', 'salary', 'compensation', 'bonus', 'insurance', 'pay', 'pension', 'retirement', 'reimbursement'],
  ],
  [
    'Policies & Compliance',
    ['policy', 'compliance', 'conduct', 'security', 'privacy', 'harassment', 'required', 'rule'],
  ],
  [
    'Roles & Employee Information',
    ['role', 'manager', 'employee', 'department', 'team', 'report', 'title', 'responsibility'],
  ],
]

export function classifyQuestion(question: string): EvaluationCategory {
  const normalized = question.toLowerCase()
  for (const [category, terms] of CATEGORY_TERMS) {
    if (terms.some((term) => normalized.includes(term))) return category
  }
  return 'Company, Culture & Careers'
}

export function providerName(provider: string): string {
  return PROVIDERS.find((item) => item.id === provider)?.name ?? provider
}

export function percent(value: number | null | undefined): string {
  return value == null ? '—' : `${Math.round(value * 100)}%`
}

export function score100(value: number | null | undefined): string {
  return value == null ? '—' : Math.round(value * 100).toString()
}

export function compactNumber(value: number): string {
  return Intl.NumberFormat('en', { notation: 'compact' }).format(value)
}

export function money(value: number | null | undefined): string {
  if (value == null) return 'Unpriced'
  if (value === 0) return '$0.00'
  if (value < 0.01) return '<$0.01'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(value)
}

export function milliseconds(value: number | null | undefined): string {
  if (value == null) return '—'
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(1)} s`
}

export function averageResultScore(
  results: EvaluationResult[],
  getter: (result: EvaluationResult) => number | null | undefined,
): number | null {
  const values = results.map(getter).filter((value): value is number => value != null)
  if (!values.length) return null
  return values.reduce((sum, value) => sum + value, 0) / values.length
}
