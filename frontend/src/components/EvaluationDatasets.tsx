import { useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Eye, Trash2, Upload, X } from 'lucide-react'
import { api } from '../lib/api'
import type { EvaluationDataset } from '../types/api'

export function EvaluationDatasets({ datasets, selected, onChange, busy }: {
  datasets: EvaluationDataset[]
  selected: string[]
  onChange: (ids: string[]) => void
  busy: boolean
}) {
  const queryClient = useQueryClient()
  const input = useRef<HTMLInputElement>(null)
  const [progress, setProgress] = useState(0)
  const [dragging, setDragging] = useState(false)
  const [previewId, setPreviewId] = useState<string | null>(null)
  const [offset, setOffset] = useState(0)
  const [deleteTarget, setDeleteTarget] = useState<EvaluationDataset | null>(null)
  const preview = useQuery({
    queryKey: ['evaluation-dataset-preview', previewId, offset],
    queryFn: () => api.previewDataset(previewId!, offset),
    enabled: Boolean(previewId),
  })
  const upload = useMutation({
    mutationFn: (files: File[]) => {
      setProgress(0)
      return api.uploadDatasets(files, setProgress)
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['evaluation-datasets'] }),
  })
  const remove = useMutation({
    mutationFn: api.deleteDataset,
    onSuccess: (_, id) => {
      onChange(selected.filter((value) => value !== id))
      setDeleteTarget(null)
      if (previewId === id) setPreviewId(null)
      return queryClient.invalidateQueries({ queryKey: ['evaluation-datasets'] })
    },
  })
  const uploadFiles = (files: File[]) => {
    if (files.length && !upload.isPending) upload.mutate(files)
  }

  return (
    <div className="dataset-library">
      <div
        className={`dataset-dropzone${dragging ? ' dragging' : ''}`}
        onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault()
          setDragging(false)
          uploadFiles(Array.from(event.dataTransfer.files))
        }}
      >
        <Upload size={24} />
        <strong>Drop your JSONL evaluation files here</strong>
        <span>Each record needs a question and reference_answer. Files are stored independently.</span>
        <input ref={input} type="file" multiple accept=".jsonl" aria-label="Select JSONL datasets"
          disabled={upload.isPending}
          onChange={(event) => {
            uploadFiles(Array.from(event.target.files ?? []))
            event.target.value = ''
          }} />
        <button type="button" className="secondary-button" disabled={upload.isPending}
          onClick={() => input.current?.click()}>
          <Upload size={16} /> {upload.isPending ? 'Uploading…' : 'Upload datasets'}
        </button>
      </div>
      {upload.isPending && <div className="dataset-upload-progress" role="status">
        <progress value={progress} max={100} aria-label="Dataset upload progress" />
        <span>{progress}% uploaded{progress === 100 ? ' · validating files…' : ''}</span>
      </div>}
      {upload.isError && <p className="error-banner" role="alert">{upload.error.message}</p>}
      {upload.data && !upload.isPending && <div role="status" className="dataset-upload-feedback">
        <p>{upload.data.datasets.length} file(s) uploaded successfully.</p>
        {upload.data.errors.map((error, index) => <p role="alert" key={index}>{error.filename}: {error.error}</p>)}
      </div>}
      <div className="dataset-library-controls">
        <span>{datasets.length} datasets · {selected.length} selected</span>
        <button type="button" className="text-button" onClick={() => onChange(
          datasets.filter((dataset) => dataset.validation_status === 'valid').map((dataset) => dataset.dataset_id),
        )}>Select All</button>
        <button type="button" className="text-button" onClick={() => onChange([])}>Clear Selection</button>
      </div>
      {!datasets.length && <p className="field-help">Upload a dataset to get started.</p>}
      <div className="dataset-library-list">
        {datasets.map((dataset) => <article className="dataset-library-row" key={dataset.dataset_id}>
          <label>
            <input type="checkbox" checked={selected.includes(dataset.dataset_id)}
              disabled={dataset.validation_status !== 'valid'}
              onChange={(event) => onChange(event.target.checked
                ? [...selected, dataset.dataset_id]
                : selected.filter((id) => id !== dataset.dataset_id))} />
            <span>
              <strong>{dataset.filename}</strong>
              <small>{dataset.record_count} records · {(dataset.size_bytes / 1024).toFixed(1)} KB · Uploaded {new Date(dataset.uploaded_at).toLocaleString()}</small>
              <small>{dataset.validation_status === 'valid' ? 'Validated' : `Invalid: ${dataset.validation_error}`} · ID {dataset.dataset_id}</small>
            </span>
          </label>
          <div className="dataset-row-actions">
            <button type="button" className="text-button" onClick={() => { setPreviewId(dataset.dataset_id); setOffset(0) }}>
              <Eye size={16} /> Preview
            </button>
            <button type="button" className="text-button" disabled={busy || remove.isPending}
              aria-label={`Delete ${dataset.filename}`} onClick={() => { remove.reset(); setDeleteTarget(dataset) }}>
              <Trash2 size={16} /> Delete
            </button>
          </div>
        </article>)}
      </div>
      {deleteTarget && <div className="dataset-delete-confirmation" role="alert">
        <span>Delete {deleteTarget.filename}? This removes this file from the library.</span>
        <button type="button" className="secondary-button" disabled={remove.isPending} onClick={() => setDeleteTarget(null)}>Cancel</button>
        <button type="button" className="secondary-button" disabled={busy || remove.isPending}
          onClick={() => remove.mutate(deleteTarget.dataset_id)}>{remove.isPending ? 'Deleting…' : 'Delete dataset'}</button>
      </div>}
      {remove.isError && <p className="error-banner" role="alert">{remove.error.message}</p>}
      {remove.isSuccess && <p role="status">Dataset deleted.</p>}
      {previewId && <div className="case-detail-backdrop" onClick={(event) => {
        if (event.target === event.currentTarget) setPreviewId(null)
      }}>
        <aside className="case-detail-panel" role="dialog" aria-modal="true" aria-labelledby="dataset-preview-title">
          <div className="case-detail-header">
            <h3 id="dataset-preview-title">{preview.data?.filename ?? 'Dataset preview'}</h3>
            <button type="button" aria-label="Close preview" onClick={() => setPreviewId(null)}><X size={18} /></button>
          </div>
          <div className="case-detail-body">
            {preview.isLoading && <p>Loading records…</p>}
            {preview.isError && <p role="alert">{preview.error.message}</p>}
            {preview.data?.validation_error && <p role="alert">{preview.data.validation_error}</p>}
            {preview.data?.records.map((record) => <section className="case-copy-block" key={record.id}>
              <span>{record.id} · {record.category} · {record.answerable ? 'Answerable' : 'Unanswerable'}</span>
              <h4>{record.question}</h4>
              <p>{record.reference_answer}</p>
              {record.keywords.length > 0 && <small>Expected keywords: {record.keywords.join(', ')}</small>}
            </section>)}
            {preview.data && <div className="dataset-library-controls">
              <button type="button" className="secondary-button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 20))}>Previous</button>
              <span>{offset + 1}–{Math.min(offset + 20, preview.data.record_count)} of {preview.data.record_count}</span>
              <button type="button" className="secondary-button" disabled={offset + 20 >= preview.data.record_count} onClick={() => setOffset(offset + 20)}>Next</button>
            </div>}
          </div>
        </aside>
      </div>}
    </div>
  )
}
