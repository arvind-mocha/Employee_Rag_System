import type { Provider, ProviderConfigurationStatus } from '../types/api'

export const PROVIDER_STATUS_LABELS = {
  configured: 'Configured',
  not_configured: 'API Key Not Configured',
  expired: 'API Key Expired',
  unavailable: 'Configuration Unavailable',
}

export function providerConfiguration(
  provider: Provider,
  configurations: ProviderConfigurationStatus[],
  unavailable = false,
  now = Date.now(),
): ProviderConfigurationStatus {
  const saved = configurations.find((item) => item.provider === provider)
  if (unavailable || !saved) return {
    provider, model_name: null, expires_at: null, status: 'unavailable',
    message: 'Configuration unavailable. Check Settings and retry.',
  }
  if (saved.status === 'configured') {
    const expiry = Date.parse(saved.expires_at ?? '')
    if (!Number.isFinite(expiry)) return {
      ...saved, status: 'unavailable', message: 'Configuration unavailable. Check Settings and retry.',
    }
    if (expiry <= now) return {
      ...saved, status: 'expired', message: 'API key expired. Update your credentials in Settings.',
    }
    if (!saved.model_name?.trim()) return {
      ...saved, status: 'not_configured', message: 'Model not configured. Configure a model for this provider in Settings.',
    }
  }
  return saved
}

export function evaluationConfigurationError(
  datasetIds: string[],
  answerProviders: Provider[],
  judgeProvider: Provider | '',
  configurations: ProviderConfigurationStatus[],
): string | null {
  if (!datasetIds.length) return 'Select at least one valid JSONL dataset.'
  if (!answerProviders.length) return 'Select at least one configured answer provider.'
  if (!judgeProvider) return 'Select a configured judge provider.'
  const names: Record<Provider, string> = {
    gemini: 'Google Gemini', openai: 'OpenAI', anthropic: 'Anthropic', grok: 'Grok (xAI)',
  }
  for (const provider of new Set([...answerProviders, judgeProvider])) {
    const configuration = providerConfiguration(provider, configurations)
    if (configuration.status !== 'configured') {
      return `Cannot start evaluation. ${names[provider]}: ${configuration.message}`
    }
  }
  return null
}
