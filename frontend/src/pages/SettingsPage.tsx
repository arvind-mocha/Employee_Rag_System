import { FormEvent, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  CheckCircle2,
  Clock3,
  Database,
  Eye,
  EyeOff,
  KeyRound,
  LockKeyhole,
  ShieldCheck,
  Trash2,
} from 'lucide-react'
import { api } from '../lib/api'
import { PROVIDERS, providerName } from '../lib/format'
import type { Provider } from '../types/api'

const EXPIRY_OPTIONS = [
  { hours: 1, label: '1 hour' },
  { hours: 8, label: '8 hours' },
  { hours: 24, label: '24 hours' },
  { hours: 168, label: '7 days' },
  { hours: 720, label: '30 days' },
  { hours: 2_160, label: '90 days' },
  { hours: 8_760, label: '1 year' },
]

const DEFAULT_MODEL_NAMES: Partial<Record<Provider, string>> = {
  gemini: 'gemini-3.5-flash-lite',
}

function dateTime(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

function timeRemaining(value: string): string {
  const milliseconds = new Date(value).getTime() - Date.now()
  if (milliseconds <= 0) return 'Expired'
  const hours = Math.ceil(milliseconds / 3_600_000)
  if (hours < 24) return `${hours}h remaining`
  const days = Math.ceil(hours / 24)
  return `${days}d remaining`
}

export function SettingsPage() {
  const queryClient = useQueryClient()
  const [provider, setProvider] = useState<Provider>('gemini')
  const [modelNames, setModelNames] = useState<Partial<Record<Provider, string>>>({})
  const [apiKey, setApiKey] = useState('')
  const [expiresInHours, setExpiresInHours] = useState(24)
  const [showKey, setShowKey] = useState(false)
  const [savedProvider, setSavedProvider] = useState<Provider | null>(null)

  const credentialsQuery = useQuery({
    queryKey: ['provider-credentials'],
    queryFn: api.listProviderCredentials,
  })

  const credentials = useMemo(
    () => credentialsQuery.data?.credentials ?? [],
    [credentialsQuery.data?.credentials],
  )
  const selectedCredential = useMemo(
    () => credentials.find((credential) => credential.provider === provider),
    [credentials, provider],
  )
  const modelName =
    modelNames[provider] ??
    selectedCredential?.model_name ??
    DEFAULT_MODEL_NAMES[provider] ??
    ''

  const saveMutation = useMutation({
    mutationFn: api.saveProviderCredential,
    onSuccess: async (credential) => {
      setApiKey('')
      setSavedProvider(credential.provider)
      await queryClient.invalidateQueries({ queryKey: ['provider-credentials'] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: api.deleteProviderCredential,
    onSuccess: async (_, deletedProvider) => {
      if (deletedProvider === provider) {
        setModelNames((current) => ({ ...current, [deletedProvider]: '' }))
        setApiKey('')
      }
      await queryClient.invalidateQueries({ queryKey: ['provider-credentials'] })
    },
  })

  const chooseProvider = (nextProvider: Provider) => {
    setProvider(nextProvider)
    setApiKey('')
    setSavedProvider(null)
  }

  const submit = (event: FormEvent) => {
    event.preventDefault()
    if (!modelName.trim() || apiKey.trim().length < 8) return
    setSavedProvider(null)
    saveMutation.mutate({
      provider,
      model_name: modelName.trim(),
      api_key: apiKey.trim(),
      expires_in_hours: expiresInHours,
    })
  }

  const remove = (targetProvider: Provider) => {
    if (
      window.confirm(
        `Delete the saved ${providerName(targetProvider)} credential from this application?`,
      )
    ) {
      deleteMutation.mutate(targetProvider)
    }
  }

  return (
    <div className="page settings-page">
      <div className="settings-hero">
        <div>
          <span className="eyebrow">
            <ShieldCheck size={14} /> Secure provider access
          </span>
          <h1>Model settings</h1>
          <p>
            Add your own provider credentials, choose the model, and decide when
            this installation must delete access automatically.
          </p>
        </div>
        <div className="settings-security-badge">
          <LockKeyhole size={18} />
          <span>
            <strong>Encrypted at rest</strong>
            <small>Keys are never returned to the browser</small>
          </span>
        </div>
      </div>

      {credentialsQuery.isError && (
        <div className="settings-message error">
          {credentialsQuery.error instanceof Error
            ? credentialsQuery.error.message
            : 'Provider settings could not be loaded.'}
        </div>
      )}

      <div className="settings-layout">
        <form className="settings-form-card" onSubmit={submit}>
          <div className="settings-card-heading">
            <div className="settings-step">01</div>
            <div>
              <h2>{selectedCredential ? 'Replace credential' : 'Add credential'}</h2>
              <p>Only the backend receives and decrypts provider keys.</p>
            </div>
          </div>

          <fieldset className="settings-fieldset">
            <legend>Provider</legend>
            <div className="settings-provider-grid">
              {PROVIDERS.map((item) => {
                const configured = credentials.some(
                  (credential) => credential.provider === item.id,
                )
                return (
                  <button
                    type="button"
                    key={item.id}
                    className={provider === item.id ? 'selected' : ''}
                    onClick={() => chooseProvider(item.id)}
                  >
                    <i>{item.shortName}</i>
                    <span>
                      <strong>{item.name}</strong>
                      <small>{configured ? 'Configured' : 'Not configured'}</small>
                    </span>
                    {configured && <CheckCircle2 size={15} />}
                  </button>
                )
              })}
            </div>
          </fieldset>

          <label className="settings-input-group">
            <span>Model name</span>
            <input
              value={modelName}
              onChange={(event) =>
                setModelNames((current) => ({
                  ...current,
                  [provider]: event.target.value,
                }))
              }
              placeholder="Enter the provider model ID"
              maxLength={255}
              autoComplete="off"
              required
            />
            <small>Use the exact model ID shown in your provider account.</small>
          </label>

          <label className="settings-input-group">
            <span>API key</span>
            <div className="secret-input">
              <KeyRound size={17} />
              <input
                type={showKey ? 'text' : 'password'}
                value={apiKey}
                onChange={(event) => setApiKey(event.target.value)}
                placeholder={
                  selectedCredential
                    ? `Replace key ending in ${selectedCredential.key_hint}`
                    : 'Paste the provider API key'
                }
                minLength={8}
                maxLength={4096}
                autoComplete="new-password"
                spellCheck={false}
                required
              />
              <button
                type="button"
                onClick={() => setShowKey((visible) => !visible)}
                aria-label={showKey ? 'Hide API key' : 'Show API key'}
              >
                {showKey ? <EyeOff size={17} /> : <Eye size={17} />}
              </button>
            </div>
            <small>The plaintext value is never sent back after this request.</small>
          </label>

          <label className="settings-input-group">
            <span>Delete automatically after</span>
            <div className="expiry-select">
              <Clock3 size={17} />
              <select
                value={expiresInHours}
                onChange={(event) => setExpiresInHours(Number(event.target.value))}
              >
                {EXPIRY_OPTIONS.map((option) => (
                  <option key={option.hours} value={option.hours}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <small>
              Expired credentials are blocked immediately and removed from PostgreSQL.
            </small>
          </label>

          {saveMutation.isError && (
            <div className="settings-message error">
              {saveMutation.error instanceof Error
                ? saveMutation.error.message
                : 'The credential could not be saved.'}
            </div>
          )}
          {savedProvider === provider && (
            <div className="settings-message success">
              <CheckCircle2 size={16} /> {providerName(provider)} was saved securely.
            </div>
          )}

          <button
            className="primary-button settings-save-button"
            type="submit"
            disabled={
              saveMutation.isPending ||
              !modelName.trim() ||
              apiKey.trim().length < 8
            }
          >
            {saveMutation.isPending ? <i className="spinner" /> : <LockKeyhole size={16} />}
            {selectedCredential ? 'Replace encrypted credential' : 'Save encrypted credential'}
          </button>
        </form>

        <div className="settings-side">
          <section className="credential-list-card">
            <div className="settings-card-heading compact">
              <div className="settings-step">02</div>
              <div>
                <h2>Active credentials</h2>
                <p>Only masked metadata is displayed.</p>
              </div>
            </div>

            {credentialsQuery.isLoading ? (
              <div className="credentials-empty">
                <i className="spinner dark" /> Loading credentials…
              </div>
            ) : credentials.length ? (
              <div className="credential-list">
                {credentials.map((credential) => (
                  <article key={credential.provider}>
                    <div className="credential-provider-icon">
                      {PROVIDERS.find((item) => item.id === credential.provider)?.shortName}
                    </div>
                    <div className="credential-details">
                      <div>
                        <strong>{providerName(credential.provider)}</strong>
                        <span className="active-pill">Active</span>
                      </div>
                      <code>{credential.model_name}</code>
                      <small>
                        Key ••••{credential.key_hint} · {timeRemaining(credential.expires_at)}
                      </small>
                      <small>Deletes {dateTime(credential.expires_at)}</small>
                    </div>
                    <button
                      className="credential-delete"
                      type="button"
                      onClick={() => remove(credential.provider)}
                      disabled={deleteMutation.isPending}
                      aria-label={`Delete ${providerName(credential.provider)} credential`}
                    >
                      <Trash2 size={16} />
                    </button>
                  </article>
                ))}
              </div>
            ) : (
              <div className="credentials-empty">
                <KeyRound size={24} />
                <strong>No provider credentials</strong>
                <span>Add one to start generating answers.</span>
              </div>
            )}

            {deleteMutation.isError && (
              <div className="settings-message error credential-error">
                {deleteMutation.error instanceof Error
                  ? deleteMutation.error.message
                  : 'The credential could not be deleted.'}
              </div>
            )}
          </section>

          <section className="settings-explainer">
            <Database size={18} />
            <div>
              <strong>What gets stored</strong>
              <p>
                PostgreSQL stores encrypted key bytes, model name, provider, key
                hint, and expiry. The encryption key stays outside the database.
              </p>
            </div>
          </section>
          <section className="settings-explainer warning">
            <ShieldCheck size={18} />
            <div>
              <strong>Deletion is local</strong>
              <p>
                Removing a key here stops this app from using it. Revoke the key in
                the provider dashboard if it may be compromised.
              </p>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
