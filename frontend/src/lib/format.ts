import type {
  EvaluationCategory,
  EvaluationResult,
  Provider,
} from '../types/api'

export const PROVIDERS: Array<{
  id: Provider
  name: string
  shortName: string
  description: string
}> = [
  {
    id: 'gemini',
    name: 'Google Gemini',
    shortName: 'GM',
    description: 'Default · Flash-Lite',
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
  gemini: '#21675c',
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

export const CATEGORIES: Array<{
  id: EvaluationCategory
  label: string
  shortLabel: string
}> = [
  { id: 'direct_fact', label: 'Direct fact', shortLabel: 'Fact' },
  { id: 'temporal', label: 'Temporal', shortLabel: 'Time' },
  { id: 'spanning', label: 'Spanning', shortLabel: 'Multi-hop' },
  { id: 'relationship', label: 'Relationship', shortLabel: 'Relation' },
  { id: 'numerical', label: 'Numerical', shortLabel: 'Number' },
  { id: 'comparative', label: 'Comparative', shortLabel: 'Compare' },
  { id: 'holistic', label: 'Holistic', shortLabel: 'Holistic' },
  { id: 'procedural', label: 'Procedural', shortLabel: 'Process' },
  { id: 'unanswerable', label: 'Unanswerable', shortLabel: 'No answer' },
]

export function categoryLabel(
  category: EvaluationCategory | null | undefined,
): string {
  return CATEGORIES.find((item) => item.id === category)?.label ?? 'Direct fact'
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
