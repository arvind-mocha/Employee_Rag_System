import type {
  AnswerRequest,
  AnswerResponse,
  DataUploadResponse,
  DeleteEvaluationDatasetResponse,
  DatasetUploadResponse,
  EvaluationDataset,
  EvaluationDatasetPreview,
  EvaluationJobAccepted,
  EvaluationJobStatus,
  IngestionStartResponse,
  IngestionStatusResponse,
  Provider,
  ProviderCredential,
  ProviderCredentialListResponse,
  ProviderCredentialRequest,
  RetrievalOptions,
  VectorDatabaseEraseResponse,
} from '../types/api'

const API_ROOT = '/api/v1'

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

  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export const api = {
  listProviderCredentials: () =>
    request<ProviderCredentialListResponse>('/settings/providers'),

  saveProviderCredential: (payload: ProviderCredentialRequest) =>
    request<ProviderCredential>('/settings/providers', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  deleteProviderCredential: (provider: Provider) =>
    request<void>(`/settings/providers/${provider}`, {
      method: 'DELETE',
    }),

  answer: (payload: AnswerRequest) =>
    request<AnswerResponse>('/answers/generate', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  uploadData: (zipFile: File) => {
    const form = new FormData()
    form.append('zip_file', zipFile)
    return request<DataUploadResponse>('/data/upload', {
      method: 'POST',
      body: form,
    })
  },

  startIngestion: () =>
    request<IngestionStartResponse>('/ingestion/start', {
      method: 'POST',
    }),

  eraseVectorDatabase: () =>
    request<VectorDatabaseEraseResponse>('/data/vector-database', {
      method: 'DELETE',
    }),

  getIngestionStatus: () =>
    request<IngestionStatusResponse>('/ingestion/status'),

  listDatasets: () =>
    request<{ datasets: EvaluationDataset[] }>('/evaluations/datasets'),

  previewDataset: (datasetId: string, offset = 0) =>
    request<EvaluationDatasetPreview>(
      `/evaluations/datasets/${encodeURIComponent(datasetId)}?offset=${offset}&limit=20`,
    ),

  uploadDatasets: (files: File[], onProgress: (percent: number) => void) =>
    new Promise<DatasetUploadResponse>((resolve, reject) => {
      const form = new FormData()
      files.forEach((file) => form.append('files', file))
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${API_ROOT}/evaluations/datasets/upload`)
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) onProgress(Math.round(event.loaded / event.total * 100))
      }
      xhr.onerror = () => reject(new ApiError('Upload failed. Check your connection and retry.', 0))
      xhr.onload = () => {
        let body
        try { body = JSON.parse(xhr.responseText) }
        catch { reject(new ApiError(`Upload failed (${xhr.status})`, xhr.status)); return }
        if (xhr.status < 200 || xhr.status >= 300) {
          reject(new ApiError(typeof body.detail === 'string' ? body.detail : 'Upload failed.', xhr.status))
        } else {
          resolve(body as DatasetUploadResponse)
        }
      }
      xhr.send(form)
    }),

  deleteDataset: (datasetName: string) =>
    request<DeleteEvaluationDatasetResponse>(
      `/evaluations/datasets/${encodeURIComponent(datasetName)}`,
      { method: 'DELETE' },
    ),

  runEvaluation: (payload: {
    dataset_ids: string[]
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
