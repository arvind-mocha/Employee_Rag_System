import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { PROVIDERS } from '../lib/format'
import { providerConfiguration } from '../lib/providerConfiguration'

export function useProviderConfiguration() {
  // Settings mutations invalidate this same key. Focus and polling pick up
  // changes in another tab; the next expiry shortens the polling interval.
  const query = useQuery({
    queryKey: ['provider-credentials'],
    queryFn: api.listProviderCredentials,
    retry: false,
    refetchOnWindowFocus: true,
    refetchOnMount: 'always',
    refetchInterval: (state) => {
      const expiries = (state.state.data?.providers ?? [])
        .filter((item) => item.status === 'configured' && item.expires_at)
        .map((item) => Date.parse(item.expires_at!) - Date.now())
        .filter((remaining) => remaining > 0)
      return Math.max(250, Math.min(15_000, ...expiries.map((remaining) => remaining + 10)))
    },
  })
  const configurations = PROVIDERS.map((provider) => providerConfiguration(
    provider.id, query.data?.providers ?? [], query.isError,
  ))
  return { ...query, configurations }
}
