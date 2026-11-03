import { useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Copy, Download, Eye, X } from 'lucide-react'
import { api } from '../lib/api'

export function EvaluationReference() {
  const [open, setOpen] = useState(false)
  const [copyStatus, setCopyStatus] = useState('')
  const rawExample = useRef<HTMLTextAreaElement>(null)
  const schema = useQuery({ queryKey: ['evaluation-schema'], queryFn: api.getEvaluationSchema })
  const example = useQuery({
    queryKey: ['evaluation-example'], queryFn: api.getEvaluationExample, enabled: open,
  })
  const copy = async () => {
    if (!example.data) return
    try {
      await navigator.clipboard.writeText(example.data)
      setCopyStatus('Example JSONL copied.')
    } catch {
      rawExample.current?.select()
      setCopyStatus('Select the JSONL below and copy it with your keyboard.')
    }
  }
  return <div className="evaluation-reference">
    <h3>Example JSONL Template</h3>
    <p>Evaluation datasets must use JSONL format, with one JSON object per line. Each record represents an evaluation test case containing a question and its expected reference information.</p>
    <p>The five examples use fictional Cedar Lane Labs policies. Replace them with your own reference data before interpreting evaluation scores.</p>
    <div className="dataset-library-controls">
      <button type="button" className="secondary-button" onClick={() => setOpen((value) => !value)}>
        {open ? <X size={16} /> : <Eye size={16} />}{open ? 'Close Example' : 'View Example'}
      </button>
      <a className="secondary-button example-download" href="/api/v1/evaluations/reference/example" download="evaluation_dataset.example.jsonl">
        <Download size={16} /> Download Example JSONL
      </a>
    </div>
    {open && <div className="example-preview">
      {example.isLoading && <p role="status">Loading example…</p>}
      {example.isError && <p className="error-banner" role="alert">{example.error.message}
        <button type="button" className="text-button" onClick={() => example.refetch()}>Retry</button>
      </p>}
      {example.data && <>
        <pre aria-label="Formatted read-only example">{example.data.trim().split('\n').map((line) => JSON.stringify(JSON.parse(line), null, 2)).join('\n\n')}</pre>
        <button type="button" className="text-button" onClick={copy}><Copy size={15} /> Copy JSONL</button>
        <span role="status">{copyStatus}</span>
        <label className="field-label" htmlFor="example-jsonl">Copyable JSONL source</label>
        <textarea id="example-jsonl" ref={rawExample} value={example.data} readOnly rows={6} wrap="off" />
      </>}
    </div>}
    <details className="evaluation-schema-reference">
      <summary>View schema documentation</summary>
      <p>Required: <code>question</code> and <code>reference_answer</code>, both nonblank strings. Questions may contain up to 2,000 characters.</p>
      <p>Optional: <code>id</code> (unique string per file; defaults to line-N), <code>category</code> (string), <code>answerable</code> (boolean, default true), <code>expected_keywords</code> or <code>keywords</code> (string arrays), <code>required_facts</code> (string array), <code>reference_evidence</code> or <code>relevant_evidence</code> (object array), and <code>metadata</code> (object).</p>
      <p>Each evidence object requires a nonempty <code>content</code> string. Source names, positions, and content hashes are optional; document IDs, chunk IDs, fingerprints, and model provenance are not required.</p>
      <p>Unanswerable records must set <code>answerable:false</code>, provide a reference abstention, and leave required facts and relevant evidence empty. They use category <code>unanswerable</code>. Answerable records cannot use that category.</p>
      <p>Missing reference evidence makes retrieval metrics unavailable. Source metrics also need source filenames. No ground-truth evidence is invented.</p>
      <p>Use UTF-8. Blank lines are ignored. Malformed JSON, duplicate record IDs, contradictory answerability, and mismatched evidence hashes are rejected with a line number.</p>
      {schema.isLoading && <p role="status">Loading current limits and categories…</p>}
      {schema.isError && <p role="alert">Schema limits could not be retrieved. <button type="button" className="text-button" onClick={() => schema.refetch()}>Retry</button></p>}
      {schema.data && <>
        <p>Current limits: {(schema.data.maximum_file_size_bytes / 1024 / 1024).toFixed(1)} MiB ({schema.data.maximum_file_size_bytes.toLocaleString()} bytes) per file; {schema.data.maximum_records} records per file; {schema.data.maximum_upload_files} files per upload.</p>
        <p>Supported categories: {schema.data.categories.join(', ')}. The default category is direct_fact.</p>
        <a href="/api/v1/evaluations/reference/schema" target="_blank" rel="noreferrer">View complete record JSON Schema</a>
      </>}
    </details>
  </div>
}
