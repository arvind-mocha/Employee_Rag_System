import { FormEvent, useRef, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  DatabaseZap,
  FileArchive,
  FolderPlus,
  FileSpreadsheet,
  FileText,
  LoaderCircle,
  Play,
  RefreshCw,
  Trash2,
  UploadCloud,
} from 'lucide-react'
import {
  INGESTION_STATUS_QUERY_KEY,
  useIngestionStatus,
} from '../hooks/useIngestionStatus'
import { api } from '../lib/api'
import type {
  DataUploadResponse,
  IngestionState,
  IngestionStatusResponse,
} from '../types/api'

const STATUS_COPY: Record<
  IngestionState,
  { title: string; description: string }
> = {
  idle: {
    title: 'Waiting for ingestion',
    description: 'Upload your source files, then build the knowledge index.',
  },
  pending: {
    title: 'Ingestion queued',
    description: 'The worker is preparing to process your documents.',
  },
  running: {
    title: 'Building the knowledge index',
    description: 'Documents are being cleaned, chunked, embedded, and indexed.',
  },
  completed: {
    title: 'Knowledge base ready',
    description: 'Ingestion completed successfully. Chat is now available.',
  },
  failed: {
    title: 'Ingestion failed',
    description: 'Review the error below, correct the issue, and try again.',
  },
}

function readableTime(value: string | null): string | null {
  if (!value) return null
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

function UploadedFiles({ upload }: { upload: DataUploadResponse }) {
  const renamedFiles = Object.entries(upload.renamed_files)

  return (
    <div className="data-upload-result">
      <div className="data-result-heading">
        <CheckCircle2 size={18} />
        <span>
          <strong>{upload.total_files} files uploaded</strong>
          <small>Supported files are stored locally and ready for ingestion.</small>
        </span>
      </div>
      <div className="data-file-groups">
        <div>
          <span>
            <FileText size={14} /> Documents
          </span>
          <strong>{upload.unstructured_files.length}</strong>
          <small>{upload.unstructured_files.join(', ') || 'None'}</small>
        </div>
        <div>
          <span>
            <FileSpreadsheet size={14} /> Structured
          </span>
          <strong>{upload.structured_files.length}</strong>
          <small>{upload.structured_files.join(', ') || 'None'}</small>
        </div>
      </div>
      {(renamedFiles.length > 0 || upload.ignored_files.length > 0) && (
        <div className="data-upload-decisions">
          {renamedFiles.length > 0 && (
            <div>
              <span>
                <RefreshCw size={13} /> Renamed duplicates
              </span>
              <ul>
                {renamedFiles.map(([original, stored]) => (
                  <li key={`${original}-${stored}`}>
                    <code>{original}</code> → <code>{stored}</code>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {upload.ignored_files.length > 0 && (
            <div className="ignored">
              <span>
                <AlertCircle size={13} /> Ignored unsupported files
              </span>
              <ul>
                {upload.ignored_files.map((filename) => (
                  <li key={filename}>
                    <code>{filename}</code>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export function DataPage() {
  const queryClient = useQueryClient()
  const fileInput = useRef<HTMLInputElement>(null)
  const [zipFile, setZipFile] = useState<File | null>(null)
  const [confirmErase, setConfirmErase] = useState(false)
  const statusQuery = useIngestionStatus()
  const status = statusQuery.data?.status ?? 'idle'
  const active = status === 'pending' || status === 'running'

  const uploadMutation = useMutation({
    mutationFn: api.uploadData,
    onSuccess: () => {
      setZipFile(null)
      if (fileInput.current) fileInput.current.value = ''
      const idle: IngestionStatusResponse = {
        status: 'idle',
        error: null,
        started_at: null,
        finished_at: null,
      }
      queryClient.setQueryData(INGESTION_STATUS_QUERY_KEY, idle)
    },
  })

  const startMutation = useMutation({
    mutationFn: api.startIngestion,
    onSuccess: async () => {
      const pending: IngestionStatusResponse = {
        status: 'pending',
        error: null,
        started_at: null,
        finished_at: null,
      }
      queryClient.setQueryData(INGESTION_STATUS_QUERY_KEY, pending)
      await queryClient.invalidateQueries({
        queryKey: INGESTION_STATUS_QUERY_KEY,
      })
    },
  })

  const eraseMutation = useMutation({
    mutationFn: api.eraseVectorDatabase,
    onSuccess: async () => {
      setConfirmErase(false)
      uploadMutation.reset()
      const idle: IngestionStatusResponse = {
        status: 'idle',
        error: null,
        started_at: null,
        finished_at: null,
      }
      queryClient.setQueryData(INGESTION_STATUS_QUERY_KEY, idle)
      await queryClient.invalidateQueries({
        queryKey: INGESTION_STATUS_QUERY_KEY,
      })
    },
  })

  const upload = (event: FormEvent) => {
    event.preventDefault()
    if (zipFile) uploadMutation.mutate(zipFile)
  }

  const chooseFile = (file: File | undefined) => {
    uploadMutation.reset()
    eraseMutation.reset()
    setConfirmErase(false)
    setZipFile(file ?? null)
  }

  const chooseAdditionalDocuments = () => {
    uploadMutation.reset()
    eraseMutation.reset()
    setConfirmErase(false)
    fileInput.current?.click()
  }

  const statusCopy = STATUS_COPY[status]
  const finishedAt = readableTime(statusQuery.data?.finished_at ?? null)

  return (
    <div className="page data-page">
      <div className="data-hero">
        <div>
          <span className="eyebrow">
            <DatabaseZap size={14} /> Knowledge setup
          </span>
          <h1>Prepare your company knowledge</h1>
          <p>
            Upload a ZIP of supported employee documents, add more ZIPs later,
            then start ingestion. Existing filenames are never overwritten.
          </p>
        </div>
        <div className={`data-readiness-badge ${status}`}>
          {active ? (
            <LoaderCircle className="spin" size={20} />
          ) : status === 'completed' ? (
            <CheckCircle2 size={20} />
          ) : (
            <DatabaseZap size={20} />
          )}
          <span>
            <strong>{statusCopy.title}</strong>
            <small>{finishedAt ? `Last completed ${finishedAt}` : statusCopy.description}</small>
          </span>
        </div>
      </div>

      {statusQuery.isError && (
        <div className="data-alert error">
          <AlertCircle size={18} />
          <span>
            <strong>Ingestion status is unavailable</strong>
            <small>
              {statusQuery.error instanceof Error
                ? statusQuery.error.message
                : 'The backend could not be reached.'}
            </small>
          </span>
          <button type="button" onClick={() => statusQuery.refetch()}>
            <RefreshCw size={14} /> Retry
          </button>
        </div>
      )}

      <section className="data-management-bar">
        <div>
          <strong>Manage knowledge data</strong>
          <small>
            Add new files or erase indexed vectors while keeping uploaded source files.
          </small>
        </div>
        <div className="data-management-actions">
          <button
            className="secondary-button"
            type="button"
            onClick={chooseAdditionalDocuments}
            disabled={active || uploadMutation.isPending || eraseMutation.isPending}
          >
            <FolderPlus size={16} /> Add more documents
          </button>
          <button
            className="data-danger-button"
            type="button"
            onClick={() => {
              eraseMutation.reset()
              setConfirmErase(true)
            }}
            disabled={active || eraseMutation.isPending}
          >
            <Trash2 size={16} /> Erase vector database
          </button>
        </div>
      </section>

      {confirmErase && (
        <div className="data-erase-confirmation" role="alert">
          <AlertCircle size={19} />
          <span>
            <strong>Erase every indexed chunk?</strong>
            <small>
              Chat will be locked until ingestion runs again. Uploaded source files
              and provider credentials will not be deleted.
            </small>
          </span>
          <div>
            <button
              className="secondary-button"
              type="button"
              onClick={() => setConfirmErase(false)}
              disabled={eraseMutation.isPending}
            >
              Cancel
            </button>
            <button
              className="data-danger-button solid"
              type="button"
              onClick={() => eraseMutation.mutate()}
              disabled={eraseMutation.isPending}
            >
              {eraseMutation.isPending ? (
                <LoaderCircle className="spin" size={16} />
              ) : (
                <Trash2 size={16} />
              )}
              {eraseMutation.isPending ? 'Erasing vectors' : 'Yes, erase vectors'}
            </button>
          </div>
        </div>
      )}

      {eraseMutation.isError && (
        <div className="data-alert error" role="alert">
          <AlertCircle size={18} />
          <span>
            <strong>Vector database could not be erased</strong>
            <small>
              {eraseMutation.error instanceof Error
                ? eraseMutation.error.message
                : 'The erase operation failed.'}
            </small>
          </span>
        </div>
      )}

      {eraseMutation.isSuccess && (
        <div className="data-alert success" role="status">
          <CheckCircle2 size={18} />
          <span>
            <strong>Vector database erased</strong>
            <small>{eraseMutation.data.message}</small>
          </span>
        </div>
      )}

      <div className="data-steps">
        <form className="data-step-card" onSubmit={upload}>
          <div className="data-step-heading">
            <span>01</span>
            <div>
              <h2>{status === 'completed' ? 'Add more documents' : 'Upload source files'}</h2>
              <p>
                Use a ZIP containing documents or structured data. Unsupported
                files are ignored, and duplicate names receive a numeric suffix.
              </p>
            </div>
          </div>

          <div className="data-quality-guidance" role="note">
            <div>
              <FileText size={17} />
              <span>
                <strong>Good answers start with good source data</strong>
                <small>
                  RAG quality depends directly on the accuracy, completeness,
                  and clarity of the files you provide.
                </small>
              </span>
            </div>
            <ul>
              <li>Provide current, complete information with clear headings.</li>
              <li>Use machine-readable text, not scanned images of documents.</li>
              <li>Supported files: PDF, DOCX, TXT, Markdown, CSV, JSON, and XLSX.</li>
              <li>Remove outdated, conflicting, empty, or irrelevant content.</li>
            </ul>
          </div>

          <button
            className={zipFile ? 'data-dropzone selected' : 'data-dropzone'}
            type="button"
            onClick={() => fileInput.current?.click()}
            disabled={active || uploadMutation.isPending || eraseMutation.isPending}
          >
            <FileArchive size={28} />
            <strong>{zipFile ? zipFile.name : 'Choose a ZIP file'}</strong>
            <small>
              {zipFile
                ? `${(zipFile.size / 1_048_576).toFixed(2)} MB selected`
                : 'PDF, DOCX, TXT, Markdown, CSV, JSON, and XLSX · maximum 50 MB'}
            </small>
          </button>
          <input
            ref={fileInput}
            className="sr-only"
            type="file"
            accept=".zip,application/zip"
            onChange={(event) => chooseFile(event.target.files?.[0])}
            disabled={active || uploadMutation.isPending || eraseMutation.isPending}
          />

          {uploadMutation.isError && (
            <div className="data-inline-error">
              <AlertCircle size={15} />
              {uploadMutation.error instanceof Error
                ? uploadMutation.error.message
                : 'The ZIP could not be uploaded.'}
            </div>
          )}

          {uploadMutation.data && <UploadedFiles upload={uploadMutation.data} />}

          <button
            className="primary-button data-action-button"
            type="submit"
            disabled={
              !zipFile ||
              active ||
              uploadMutation.isPending ||
              eraseMutation.isPending
            }
          >
            {uploadMutation.isPending ? (
              <i className="spinner" />
            ) : (
              <UploadCloud size={17} />
            )}
            {uploadMutation.isPending
              ? 'Uploading ZIP'
              : status === 'completed'
                ? 'Add documents'
                : 'Upload ZIP'}
          </button>
        </form>

        <section className="data-step-card">
          <div className="data-step-heading">
            <span>02</span>
            <div>
              <h2>Build the index</h2>
              <p>Start the ingestion pipeline and follow its live status.</p>
            </div>
          </div>

          <div className={`ingestion-state-panel ${status}`}>
            <div className="ingestion-state-icon">
              {active ? (
                <LoaderCircle className="spin" size={24} />
              ) : status === 'completed' ? (
                <CheckCircle2 size={24} />
              ) : status === 'failed' ? (
                <AlertCircle size={24} />
              ) : (
                <DatabaseZap size={24} />
              )}
            </div>
            <div>
              <span>Current status · {status}</span>
              <strong>{statusCopy.title}</strong>
              <p>{statusCopy.description}</p>
            </div>
          </div>

          {status === 'failed' && statusQuery.data?.error && (
            <div className="data-inline-error">
              <AlertCircle size={15} /> {statusQuery.data.error}
            </div>
          )}
          {startMutation.isError && (
            <div className="data-inline-error">
              <AlertCircle size={15} />
              {startMutation.error instanceof Error
                ? startMutation.error.message
                : 'Ingestion could not be started.'}
            </div>
          )}

          {status === 'completed' ? (
            <Link className="primary-button data-action-button" to="/">
              Open chat <ArrowRight size={17} />
            </Link>
          ) : (
            <button
              className="primary-button data-action-button"
              type="button"
              onClick={() => startMutation.mutate()}
              disabled={
                active ||
                statusQuery.isLoading ||
                statusQuery.isError ||
                startMutation.isPending ||
                eraseMutation.isPending ||
                confirmErase
              }
            >
              {active || startMutation.isPending ? (
                <LoaderCircle className="spin" size={17} />
              ) : (
                <Play size={17} />
              )}
              {active ? 'Ingestion in progress' : 'Start ingestion'}
            </button>
          )}
        </section>
      </div>
    </div>
  )
}
