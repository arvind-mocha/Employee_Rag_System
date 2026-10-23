import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'

export const INGESTION_STATUS_QUERY_KEY = ['ingestion-status'] as const

export function useIngestionStatus() {
  return useQuery({
    queryKey: INGESTION_STATUS_QUERY_KEY,
    queryFn: api.getIngestionStatus,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === 'pending' || status === 'running' ? 1_000 : false
    },
  })
}
