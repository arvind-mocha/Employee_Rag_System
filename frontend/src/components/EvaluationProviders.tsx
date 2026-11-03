import { Check, RefreshCw } from 'lucide-react'
import { Link } from 'react-router-dom'
import { PROVIDERS, PROVIDER_COLORS } from '../lib/format'
import { PROVIDER_STATUS_LABELS } from '../lib/providerConfiguration'
import type { Provider, ProviderConfigurationStatus } from '../types/api'

export function EvaluationProviderStatus({ configurations, loading, error, onRetry }: {
  configurations: ProviderConfigurationStatus[]
  loading: boolean
  error: boolean
  onRetry: () => void
}) {
  return <>
    {loading && <p role="status">Checking provider credentials…</p>}
    {error && <div className="error-banner" role="alert">
      <span>Configuration Unavailable. Provider status could not be retrieved.</span>
      <button type="button" className="text-button" onClick={onRetry}><RefreshCw size={14} /> Retry</button>
    </div>}
    <div className="evaluation-provider-status-grid">
      {PROVIDERS.map((provider) => {
        const configuration = configurations.find((item) => item.provider === provider.id)!
        return <article key={provider.id}>
          <strong>{provider.name}</strong>
          <span>Model: {configuration.model_name || 'Not configured'}</span>
          <span className={`credential-status ${loading ? 'checking' : configuration.status}`}>
            <i aria-hidden="true" />{loading ? 'Checking credentials…' : PROVIDER_STATUS_LABELS[configuration.status]}
          </span>
          {!loading && <p>{configuration.message}</p>}
          <Link to="/settings">Configure in Settings</Link>
        </article>
      })}
    </div>
    <p className="field-help">Configured means a saved, non-expired API key and model are available. It does not verify acceptance by the external provider.</p>
  </>
}

export function ProviderPicker({ selected, onChange, configurations, loading }: {
  selected: Provider[]
  onChange: (providers: Provider[]) => void
  configurations: ProviderConfigurationStatus[]
  loading: boolean
}) {
  return <div className="provider-check-grid">
    {PROVIDERS.map((provider) => {
      const active = selected.includes(provider.id)
      const configuration = configurations.find((item) => item.provider === provider.id)!
      const usable = !loading && configuration.status === 'configured'
      return <button type="button" key={provider.id} className={active ? 'selected' : ''}
        disabled={!usable && !active} aria-pressed={active}
        title={configuration.message}
        onClick={() => onChange(active
          ? selected.filter((item) => item !== provider.id)
          : [...selected, provider.id])}>
        <i style={{ background: PROVIDER_COLORS[provider.id] }}>{provider.shortName}</i>
        <span>{provider.name}<small>{configuration.model_name || 'No model configured'}</small>
          <small>{loading ? 'Checking…' : PROVIDER_STATUS_LABELS[configuration.status]}</small></span>
        <b>{active && <Check size={13} />}</b>
      </button>
    })}
  </div>
}
