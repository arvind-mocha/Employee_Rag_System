import type { RetrievalOptions } from '../types/api'

interface StrategyControlsProps {
  value: RetrievalOptions
  onChange: (value: RetrievalOptions) => void
  compact?: boolean
}

const STRATEGIES: Array<{
  key: keyof RetrievalOptions
  title: string
  description: string
}> = [
  {
    key: 'semantic_search',
    title: 'Semantic',
    description: 'Meaning-based similarity',
  },
  {
    key: 'keyword_search',
    title: 'Keyword',
    description: 'Exact terms and phrases',
  },
  {
    key: 'query_rewriting',
    title: 'Rewrite query',
    description: 'Normalize before retrieval',
  },
  {
    key: 'reranking',
    title: 'Rerank',
    description: 'Reorder best candidates',
  },
  {
    key: 'source_diversity',
    title: 'Source diversity',
    description: 'Spread across documents',
  },
]

export function StrategyControls({
  value,
  onChange,
  compact = false,
}: StrategyControlsProps) {
  const toggle = (key: keyof RetrievalOptions) => {
    const next = { ...value, [key]: !value[key] }
    if (!next.semantic_search && !next.keyword_search) return
    onChange(next)
  }

  return (
    <div className={compact ? 'strategy-list compact' : 'strategy-list'}>
      {STRATEGIES.map((strategy) => (
        <button
          className="strategy-toggle"
          type="button"
          key={strategy.key}
          onClick={() => toggle(strategy.key)}
          aria-pressed={value[strategy.key]}
        >
          <span>
            <strong>{strategy.title}</strong>
            {!compact && <small>{strategy.description}</small>}
          </span>
          <i className={value[strategy.key] ? 'switch on' : 'switch'}>
            <b />
          </i>
        </button>
      ))}
    </div>
  )
}
