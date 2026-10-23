import type {
  AnswerRequest,
  AnswerResponse,
  EvaluationJobAccepted,
  EvaluationJobStatus,
  Provider,
  RetrievalOptions,
  UploadEvaluationDatasetResponse,
} from '../types/api'

const API_ROOT = import.meta.env.VITE_API_ROOT ?? '/api/v1'

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const hasMultipartBody = init?.body instanceof FormData
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    headers: hasMultipartBody
      ? init?.headers
      : {
          'Content-Type': 'application/json',
          ...init?.headers,
        },
  })

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`
    try {
      const body = (await response.json()) as { detail?: string | unknown[] }
      if (typeof body.detail === 'string') message = body.detail
      else if (Array.isArray(body.detail)) {
        message = body.detail
          .map((item) =>
            typeof item === 'object' && item && 'msg' in item
              ? String(item.msg)
              : String(item),
          )
          .join(', ')
      }
    } catch {
      // Keep the status-based fallback when a proxy or server returns HTML.
    }
    throw new ApiError(message, response.status)
  }

  return response.json() as Promise<T>
}

export const api = {
  answer: (payload: AnswerRequest) =>
    request<AnswerResponse>('/answers/generate', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  generateDataset: (payload: {
    dataset_name: string
    questions: string[]
    reference_provider: Provider
    overwrite: boolean
  } & RetrievalOptions) =>
    request<EvaluationJobAccepted>('/evaluations/datasets/generate/async', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  uploadDataset: (payload: {
    dataset_name: string
    dataset_file: File
    overwrite: boolean
  }) => {
    const form = new FormData()
    form.append('dataset_name', payload.dataset_name)
    form.append('dataset_file', payload.dataset_file)
    form.append('overwrite', String(payload.overwrite))
    return request<UploadEvaluationDatasetResponse>(
      '/evaluations/datasets/upload',
      {
        method: 'POST',
        body: form,
      },
    )
  },

  runEvaluation: (payload: {
    dataset_name: string
    answer_providers: Provider[]
    judge_provider: Provider
  } & RetrievalOptions) =>
    request<EvaluationJobAccepted>('/evaluations/run/async', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getJob: (statusUrl: string) => {
    const path = statusUrl.startsWith('/api/v1')
      ? statusUrl.slice('/api/v1'.length)
      : statusUrl
    return request<EvaluationJobStatus>(path)
  },
}
