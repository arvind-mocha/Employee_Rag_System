import { FormEvent, useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  AlertTriangle,
  BarChart3,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleDollarSign,
  Database,
  FilePlus2,
  Gauge,
  Info,
  Layers3,
  Play,
  Plus,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Target,
  UploadCloud,
  X,
  XCircle,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { StrategyControls } from '../components/StrategyControls'
import { api } from '../lib/api'
import {
  averageResultScore,
  CATEGORIES,
  classifyQuestion,
  compactNumber,
  DEFAULT_RETRIEVAL,
  milliseconds,
  money,
  percent,
  PROVIDER_COLORS,
  PROVIDERS,
  providerName,
  score100,
} from '../lib/format'
import type {
  EvaluationJobAccepted,
  EvaluationRun,
  Provider,
  RetrievalOptions,
} from '../types/api'

const DATASET_EXAMPLE = `How many vacation days do employees receive?
What health and dental benefits are available?
What is the remote work policy?
Who approves employee expense reports?
What are the company's core values?`

const TERMINAL_JOB_STATES = new Set([
  'completed',
  'failed',
  'stopped',
  'canceled',
])

const EVALUATION_JOB_STORAGE_KEY = 'employee-rag-active-evaluation-job'
const DATASET_JOB_STORAGE_KEY = 'employee-rag-active-dataset-job'
const DATASET_NAME_STORAGE_KEY = 'employee-rag-evaluation-dataset-name'
const STORED_JOB_MAX_AGE_MS = 25 * 60 * 60 * 1_000

interface StoredEvaluationJob {
  job: EvaluationJobAccepted
  savedAt: number
}

function isEvaluationJob(value: unknown): value is EvaluationJobAccepted {
  if (!value || typeof value !== 'object') return false
  const job = value as Record<string, unknown>
  return (
    typeof job.job_id === 'string' &&
    (job.job_type === 'dataset_generation' || job.job_type === 'evaluation_run') &&
    job.status === 'queued' &&
    typeof job.status_url === 'string' &&
    job.status_url.startsWith('/api/v1/evaluations/jobs/')
  )
}

function readStoredJob(key: string): EvaluationJobAccepted | null {
  try {
    const parsed = JSON.parse(
      window.localStorage.getItem(key) ?? 'null',
    ) as Partial<StoredEvaluationJob> | null
    if (
      !parsed ||
      !isEvaluationJob(parsed.job) ||
      typeof parsed.savedAt !== 'number' ||
      Date.now() - parsed.savedAt > STORED_JOB_MAX_AGE_MS
    ) {
      window.localStorage.removeItem(key)
      return null
    }
    return parsed.job
  } catch {
    window.localStorage.removeItem(key)
    return null
  }
}

function storeJob(key: string, job: EvaluationJobAccepted | null) {
  if (!job) {
    window.localStorage.removeItem(key)
    return
  }
  const stored: StoredEvaluationJob = { job, savedAt: Date.now() }
  window.localStorage.setItem(key, JSON.stringify(stored))
}

function readStoredDatasetName(): string {
  const value = window.localStorage.getItem(DATASET_NAME_STORAGE_KEY) ?? ''
  return /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(value)
    ? value
    : 'employee-policies'
}

function storeDatasetName(value: string) {
  window.localStorage.setItem(DATASET_NAME_STORAGE_KEY, value)
}

type DatasetBuilderMode = 'upload' | 'generate'

function useEvaluationJob(job: EvaluationJobAccepted | null) {
  return useQuery({
    queryKey: ['evaluation-job', job?.job_id],
    queryFn: () => api.getJob(job!.status_url),
    enabled: Boolean(job),
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status && TERMINAL_JOB_STATES.has(status) ? false : 1500
    },
  })
}

function ProviderPicker({
  selected,
  onChange,
}: {
  selected: Provider[]
  onChange: (providers: Provider[]) => void
}) {
  const toggle = (provider: Provider) => {
    if (selected.includes(provider)) {
      if (selected.length === 1) return
      onChange(selected.filter((item) => item !== provider))
    } else {
      onChange([...selected, provider])
    }
  }

  return (
    <div className="provider-check-grid">
      {PROVIDERS.map((provider) => {
        const active = selected.includes(provider.id)
        return (
          <button
            type="button"
            key={provider.id}
            className={active ? 'selected' : ''}
            onClick={() => toggle(provider.id)}
            aria-pressed={active}
          >
            <i style={{ background: PROVIDER_COLORS[provider.id] }}>
              {provider.shortName}
            </i>
            <span>{provider.name}</span>
            <b>{active && <Check size={13} />}</b>
          </button>
        )
      })}
    </div>
  )
}

function JobProgress({
  label,
  status,
  error,
}: {
  label: string
  status: string
  error?: string | null
}) {
  const failed = ['failed', 'stopped', 'canceled'].includes(status)
  const complete = status === 'completed'
  return (
    <div className={failed ? 'job-progress failed' : 'job-progress'}>
      <div className="job-icon">
        {failed ? (
          <XCircle size={18} />
        ) : complete ? (
          <CheckCircle2 size={18} />
        ) : (
          <RefreshCw className="spin" size={18} />
        )}
      </div>
      <div>
        <strong>{label}</strong>
        <span>{error ?? (complete ? 'Completed successfully' : `${status}…`)}</span>
        {!failed && !complete && (
          <small>Safe to leave this page—the background worker will continue.</small>
        )}
      </div>
    </div>
  )
}

function MetricCard({
  icon,
  label,
  value,
  detail,
  accent,
}: {
  icon: React.ReactNode
  label: string
  value: string
  detail: string
  accent?: boolean
}) {
  return (
    <div className={accent ? 'metric-card accent' : 'metric-card'}>
      <div className="metric-icon">{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  )
}

function EvaluationDashboard({ run }: { run: EvaluationRun }) {
  const chartData = useMemo(
    () =>
      CATEGORIES.map((category) => {
        const row: Record<string, string | number | null> = {
          category,
          shortCategory: category.replace(' & ', ' / '),
        }
        for (const summary of run.summaries) {
          const matching = run.results.filter(
            (result) =>
              result.answer_provider === summary.provider &&
              classifyQuestion(result.question) === category &&
              result.answer_scores,
          )
          const average = averageResultScore(
            matching,
            (result) => result.answer_scores?.overall,
          )
          row[summary.provider] = average == null ? null : Math.round(average * 100)
        }
        return row
      }),
    [run],
  )

  const overall = averageResultScore(run.results, (result) => result.answer_scores?.overall)
  const groundedness = averageResultScore(
    run.results,
    (result) => result.answer_scores?.groundedness,
  )
  const retrievalRecall = averageResultScore(
    run.results,
    (result) => result.retrieval_scores?.recall_at_k,
  )
  const passed = run.results.filter((result) => result.passed).length
  const evaluated = run.results.filter((result) => result.passed != null).length

  return (
    <div className="evaluation-dashboard">
      <div className="dashboard-title-row">
        <div>
          <span className="eyebrow">Latest completed run</span>
          <h2>Evaluation results</h2>
          <p>
            {run.dataset_name} · {run.results.length} model answers · judged by{' '}
            {providerName(run.judge_provider)}
          </p>
        </div>
        <div className="run-meta">
          <span>Run ID</span>
          <code>{run.run_id}</code>
        </div>
      </div>

      <div className="evaluation-overview">
        <div className="metric-column">
          <MetricCard
            icon={<Gauge size={18} />}
            label="Overall score"
            value={score100(overall)}
            detail="Average across all answers"
            accent
          />
          <MetricCard
            icon={<ShieldCheck size={18} />}
            label="Groundedness"
            value={score100(groundedness)}
            detail="Claims supported by evidence"
          />
          <MetricCard
            icon={<Target size={18} />}
            label="Retrieval recall"
            value={score100(retrievalRecall)}
            detail="Relevant evidence recovered"
          />
          <MetricCard
            icon={<CheckCircle2 size={18} />}
            label="Pass rate"
            value={evaluated ? `${Math.round((passed / evaluated) * 100)}%` : '—'}
            detail={`${passed} of ${evaluated} graded answers`}
          />
          <MetricCard
            icon={<CircleDollarSign size={18} />}
            label="Estimated cost"
            value={money(run.usage.estimated_cost_usd)}
            detail={`${compactNumber(run.usage.total_tokens)} total tokens`}
          />
        </div>

        <section className="chart-card">
          <div className="chart-heading">
            <div>
              <span>Category performance</span>
              <h3>Overall answer score by model</h3>
            </div>
            <div className="score-scale">Score / 100</div>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 12, right: 8, left: -14, bottom: 28 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e8e2" />
                <XAxis
                  dataKey="shortCategory"
                  tick={{ fill: '#69716c', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  interval={0}
                  angle={-12}
                  textAnchor="end"
                  height={62}
                />
                <YAxis
                  domain={[0, 100]}
                  ticks={[0, 25, 50, 75, 100]}
                  tick={{ fill: '#818984', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  cursor={{ fill: '#f1f2ed' }}
                  contentStyle={{
                    border: '1px solid #dedfd8',
                    borderRadius: 10,
                    boxShadow: '0 10px 30px rgba(31, 40, 35, .08)',
                  }}
                  formatter={(value, name) => [
                    value == null ? 'No results' : `${value}/100`,
                    providerName(String(name)),
                  ]}
                />
                <Legend
                  iconType="circle"
                  formatter={(value) => providerName(value)}
                  wrapperStyle={{ fontSize: 12 }}
                />
                {run.summaries.map((summary) => (
                  <Bar
                    key={summary.provider}
                    dataKey={summary.provider}
                    fill={PROVIDER_COLORS[summary.provider]}
                    radius={[4, 4, 0, 0]}
                    maxBarSize={30}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-note">
            <Info size={14} /> Questions are mapped to the five employee categories
            for display; evaluation scores come directly from the API.
          </div>
        </section>
      </div>

      <section className="provider-summary-section">
        <div className="section-title">
          <div>
            <span>Model comparison</span>
            <h3>Provider scorecard</h3>
          </div>
        </div>
        <div className="provider-summary-grid">
          {run.summaries.map((summary) => (
            <article key={summary.provider}>
              <div className="provider-summary-name">
                <i style={{ background: PROVIDER_COLORS[summary.provider] }} />
                <strong>{providerName(summary.provider)}</strong>
                <span>{summary.evaluated} evaluated</span>
              </div>
              <div>
                <span>Overall</span>
                <b>{score100(summary.average_scores.overall)}</b>
              </div>
              <div>
                <span>Correctness</span>
                <b>{percent(summary.average_scores.correctness)}</b>
              </div>
              <div>
                <span>Pass rate</span>
                <b>{percent(summary.pass_rate)}</b>
              </div>
              <div>
                <span>Latency</span>
                <b>{milliseconds(summary.average_latency_ms)}</b>
              </div>
              <div>
                <span>Cost</span>
                <b>{money(summary.usage.estimated_cost_usd)}</b>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="results-table-section">
        <div className="section-title">
          <div>
            <span>Question-level detail</span>
            <h3>Evaluation cases</h3>
          </div>
          <span className="row-count">{run.results.length} results</span>
        </div>
        <div className="table-scroll">
          <table className="results-table">
            <thead>
              <tr>
                <th>Question</th>
                <th>Category</th>
                <th>Model</th>
                <th>Overall</th>
                <th>Correct</th>
                <th>Grounded</th>
                <th>Issues</th>
                <th>Latency</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {run.results.map((result) => (
                <tr key={`${result.record_id}-${result.answer_provider}`}>
                  <td>
                    <strong>{result.question}</strong>
                    <small>{result.record_id}</small>
                  </td>
                  <td>
                    <span className="category-pill">{classifyQuestion(result.question)}</span>
                  </td>
                  <td>{providerName(result.answer_provider)}</td>
                  <td className="score-cell">{score100(result.answer_scores?.overall)}</td>
                  <td>{percent(result.answer_scores?.correctness)}</td>
                  <td>{percent(result.answer_scores?.groundedness)}</td>
                  <td>
                    <span className={result.unsupported_claims.length ? 'issue-count warn' : 'issue-count'}>
                      {result.unsupported_claims.length} unsupported
                    </span>
                    <span className={result.missing_facts.length ? 'issue-count warn' : 'issue-count'}>
                      {result.missing_facts.length} missing
                    </span>
                  </td>
                  <td>{milliseconds(result.timing.total_ms)}</td>
                  <td>
                    {result.error ? (
                      <span className="result-badge error">Error</span>
                    ) : result.passed ? (
                      <span className="result-badge pass">Pass</span>
                    ) : (
                      <span className="result-badge fail">Fail</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}

export function EvaluationPage() {
  const [datasetName, setDatasetName] = useState(readStoredDatasetName)
  const [answerProviders, setAnswerProviders] = useState<Provider[]>([
    'huggingface',
    'openai',
  ])
  const [judgeProvider, setJudgeProvider] = useState<Provider>('openai')
  const [retrieval, setRetrieval] =
    useState<RetrievalOptions>(DEFAULT_RETRIEVAL)
  const [evaluationJob, setEvaluationJob] =
    useState<EvaluationJobAccepted | null>(() =>
      readStoredJob(EVALUATION_JOB_STORAGE_KEY),
    )
  const [showDatasetBuilder, setShowDatasetBuilder] = useState(false)
  const [datasetBuilderMode, setDatasetBuilderMode] =
    useState<DatasetBuilderMode>('upload')
  const [datasetQuestions, setDatasetQuestions] = useState(DATASET_EXAMPLE)
  const [referenceProvider, setReferenceProvider] =
    useState<Provider>('huggingface')
  const [overwrite, setOverwrite] = useState(false)
  const [datasetJob, setDatasetJob] = useState<EvaluationJobAccepted | null>(
    () => readStoredJob(DATASET_JOB_STORAGE_KEY),
  )
  const [datasetFile, setDatasetFile] = useState<File | null>(null)

  const evaluationMutation = useMutation({
    mutationFn: api.runEvaluation,
    onSuccess: (job) => {
      storeJob(EVALUATION_JOB_STORAGE_KEY, job)
      setEvaluationJob(job)
    },
  })
  const evaluationStatus = useEvaluationJob(evaluationJob)

  const datasetMutation = useMutation({
    mutationFn: api.generateDataset,
    onSuccess: (job) => {
      storeJob(DATASET_JOB_STORAGE_KEY, job)
      setDatasetJob(job)
    },
  })
  const datasetStatus = useEvaluationJob(datasetJob)

  const uploadMutation = useMutation({
    mutationFn: api.uploadDataset,
    onSuccess: (result) => {
      storeDatasetName(result.dataset_name)
      setDatasetName(result.dataset_name)
    },
  })

  const completedRun =
    evaluationStatus.data?.status === 'completed' && evaluationStatus.data.result
      ? (evaluationStatus.data.result as unknown as EvaluationRun)
      : null

  const runEvaluation = (event: FormEvent) => {
    event.preventDefault()
    if (!datasetName.trim() || !answerProviders.length) return
    storeDatasetName(datasetName.trim())
    storeJob(EVALUATION_JOB_STORAGE_KEY, null)
    setEvaluationJob(null)
    evaluationMutation.mutate({
      dataset_name: datasetName.trim(),
      answer_providers: answerProviders,
      judge_provider: judgeProvider,
      ...retrieval,
    })
  }

  const generateDataset = (event: FormEvent) => {
    event.preventDefault()
    const questions = datasetQuestions
      .split('\n')
      .map((question) => question.trim())
      .filter(Boolean)
    if (!datasetName.trim() || !questions.length) return
    storeDatasetName(datasetName.trim())
    storeJob(DATASET_JOB_STORAGE_KEY, null)
    setDatasetJob(null)
    uploadMutation.reset()
    datasetMutation.mutate({
      dataset_name: datasetName.trim(),
      questions,
      reference_provider: referenceProvider,
      overwrite,
      ...retrieval,
    })
  }

  const uploadDataset = (event: FormEvent) => {
    event.preventDefault()
    if (!datasetName.trim() || !datasetFile) return
    storeDatasetName(datasetName.trim())
    storeJob(DATASET_JOB_STORAGE_KEY, null)
    setDatasetJob(null)
    datasetMutation.reset()
    uploadMutation.mutate({
      dataset_name: datasetName.trim(),
      dataset_file: datasetFile,
      overwrite,
    })
  }

  const selectDatasetFile = (file: File | null) => {
    setDatasetFile(file)
    uploadMutation.reset()
    if (!file) return
    const filename = file.name.replace(/\.jsonl$/i, '')
    if (/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(filename)) {
      setDatasetName(filename)
    }
  }

  const activeEvaluation = evaluationMutation.isPending || Boolean(
    evaluationStatus.data && !TERMINAL_JOB_STATES.has(evaluationStatus.data.status),
  )
  const activeDataset = datasetMutation.isPending || Boolean(
    datasetStatus.data && !TERMINAL_JOB_STATES.has(datasetStatus.data.status),
  ) || uploadMutation.isPending

  return (
    <div className="page evaluation-page">
      <div className="evaluation-hero">
        <div>
          <span className="eyebrow">
            <BarChart3 size={14} /> On-demand quality lab
          </span>
          <h1>Evaluate answer quality</h1>
          <p>
            Compare answer models against a ground-truth dataset. Nothing runs
            automatically—you choose the dataset, models, judge, and timing.
          </p>
        </div>
        <div className="evaluation-assurance">
          <ShieldCheck size={19} />
          <div>
            <strong>Manual by design</strong>
            <span>Evaluation only starts when you click Run</span>
          </div>
        </div>
      </div>

      <form className="evaluation-config" onSubmit={runEvaluation}>
        <section className="config-primary">
          <div className="section-number">01</div>
          <div className="config-content">
            <div className="config-heading">
              <div>
                <span>Ground truth</span>
                <h2>Select an evaluation dataset</h2>
              </div>
              <button
                type="button"
                className="text-button"
                onClick={() => setShowDatasetBuilder((value) => !value)}
              >
                {showDatasetBuilder ? <X size={15} /> : <Plus size={15} />}
                {showDatasetBuilder ? 'Close builder' : 'Add dataset'}
              </button>
            </div>
            <label className="field-label" htmlFor="dataset-name">
              Dataset name
            </label>
            <div className="dataset-field">
              <Database size={17} />
              <input
                id="dataset-name"
                value={datasetName}
                onChange={(event) => setDatasetName(event.target.value)}
                placeholder="employee-policies"
                pattern="[A-Za-z0-9][A-Za-z0-9_-]*"
                maxLength={64}
                required
              />
              <span className="file-extension">.jsonl</span>
            </div>
            <small className="field-help">
              Enter the stored filename without the .jsonl extension.
            </small>
            <div className="dataset-quality-warning" role="note">
              <AlertTriangle size={15} />
              <span>
                <strong>Dataset quality matters.</strong> If the ground truth is
                incorrect, invalid, or not representative, the evaluation results
                will be unreliable and may not be useful.
              </span>
            </div>
          </div>
        </section>

        <section className="config-primary">
          <div className="section-number">02</div>
          <div className="config-content">
            <div className="config-heading">
              <div>
                <span>Models under test</span>
                <h2>Choose answer providers</h2>
              </div>
              <span className="selection-count">{answerProviders.length} selected</span>
            </div>
            <ProviderPicker selected={answerProviders} onChange={setAnswerProviders} />
          </div>
        </section>

        <section className="config-primary">
          <div className="section-number">03</div>
          <div className="config-content config-split">
            <div>
              <div className="config-heading">
                <div>
                  <span>Independent grader</span>
                  <h2>Choose the judge model</h2>
                </div>
              </div>
              <label className="select-field">
                <Sparkles size={17} />
                <select
                  value={judgeProvider}
                  onChange={(event) => setJudgeProvider(event.target.value as Provider)}
                >
                  {PROVIDERS.map((provider) => (
                    <option value={provider.id} key={provider.id}>
                      {provider.name}
                    </option>
                  ))}
                </select>
                <ChevronDown size={16} />
              </label>
              <small className="field-help">
                The judge scores correctness, groundedness, completeness, relevance,
                and citations.
              </small>
            </div>
            <div className="retrieval-compact-panel">
              <div className="config-heading">
                <div>
                  <span>Evidence search</span>
                  <h2>Retrieval controls</h2>
                </div>
              </div>
              <StrategyControls value={retrieval} onChange={setRetrieval} compact />
            </div>
          </div>
        </section>

        <div className="run-bar">
          <div>
            <Layers3 size={18} />
            <span>
              <strong>{datasetName || 'No dataset selected'}</strong>
              {answerProviders.length} answer model{answerProviders.length === 1 ? '' : 's'} ·{' '}
              {providerName(judgeProvider)} judge
            </span>
          </div>
          <button
            className="primary-button"
            type="submit"
            disabled={activeEvaluation || !datasetName.trim()}
          >
            {activeEvaluation ? <i className="spinner" /> : <Play size={17} />}
            {activeEvaluation ? 'Evaluation running' : 'Run evaluation'}
          </button>
        </div>
      </form>

      {showDatasetBuilder && (
        <section className="dataset-builder">
          <div className="dataset-builder-heading">
            <div className="builder-icon">
              {datasetBuilderMode === 'upload' ? (
                <UploadCloud size={20} />
              ) : (
                <FilePlus2 size={20} />
              )}
            </div>
            <div>
              <span>Evaluation dataset</span>
              <h2>Upload JSONL or generate ground truth</h2>
              <p>
                Provide a compatible JSONL dataset of your own, or generate one
                from the currently indexed documents.
              </p>
            </div>
          </div>
          <div className="dataset-mode-tabs" role="tablist" aria-label="Dataset source">
            <button
              type="button"
              role="tab"
              aria-selected={datasetBuilderMode === 'upload'}
              className={datasetBuilderMode === 'upload' ? 'active' : ''}
              onClick={() => setDatasetBuilderMode('upload')}
            >
              <UploadCloud size={15} /> Upload JSONL
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={datasetBuilderMode === 'generate'}
              className={datasetBuilderMode === 'generate' ? 'active' : ''}
              onClick={() => setDatasetBuilderMode('generate')}
            >
              <Sparkles size={15} /> Generate dataset
            </button>
          </div>

          {datasetBuilderMode === 'upload' ? (
            <form className="builder-grid" onSubmit={uploadDataset}>
              <div className="questions-field">
                <span>JSONL dataset</span>
                <label className={datasetFile ? 'upload-dropzone selected' : 'upload-dropzone'}>
                  <UploadCloud size={25} />
                  <strong>{datasetFile?.name ?? 'Choose a .jsonl file'}</strong>
                  <small>
                    {datasetFile
                      ? `${Math.max(1, Math.ceil(datasetFile.size / 1024))} KB selected`
                      : 'Up to 10 MB and 100 evaluation records'}
                  </small>
                  <input
                    type="file"
                    accept=".jsonl,application/x-ndjson"
                    onChange={(event) => selectDatasetFile(event.target.files?.[0] ?? null)}
                    required
                  />
                </label>
                <small>
                  Records must match the generated ground-truth JSONL schema,
                  including reference answers, facts, and evidence hashes.
                </small>
              </div>
              <div className="builder-options">
                <label>
                  <span>Save as</span>
                  <div className="builder-filename">
                    <input
                      value={datasetName}
                      onChange={(event) => setDatasetName(event.target.value)}
                      pattern="[A-Za-z0-9][A-Za-z0-9_-]*"
                      maxLength={64}
                      required
                    />
                    <span>.jsonl</span>
                  </div>
                </label>
                <button
                  className={overwrite ? 'checkbox-row checked' : 'checkbox-row'}
                  type="button"
                  onClick={() => setOverwrite((value) => !value)}
                >
                  <i>{overwrite && <Check size={13} />}</i>
                  <span>
                    <strong>Overwrite existing dataset</strong>
                    <small>Replace a JSONL file with the same name</small>
                  </span>
                </button>
                <div className="warning-note">
                  <AlertTriangle size={16} />
                  <span>
                    Uploaded data is validated for structure, not factual quality.
                    Review it carefully before relying on evaluation scores.
                  </span>
                </div>
                <button
                  className="secondary-button"
                  type="submit"
                  disabled={activeDataset || !datasetName.trim() || !datasetFile}
                >
                  {uploadMutation.isPending ? (
                    <i className="spinner dark" />
                  ) : (
                    <UploadCloud size={16} />
                  )}
                  {uploadMutation.isPending ? 'Uploading dataset' : 'Upload dataset'}
                </button>
              </div>
            </form>
          ) : (
            <form className="builder-grid" onSubmit={generateDataset}>
              <label className="questions-field">
                <span>Questions</span>
                <textarea
                  value={datasetQuestions}
                  onChange={(event) => setDatasetQuestions(event.target.value)}
                  rows={7}
                  required
                />
                <small>
                  {datasetQuestions.split('\n').filter((line) => line.trim()).length} questions
                </small>
              </label>
              <div className="builder-options">
                <label>
                  <span>Save as</span>
                  <div className="builder-filename">
                    <input
                      value={datasetName}
                      onChange={(event) => setDatasetName(event.target.value)}
                      pattern="[A-Za-z0-9][A-Za-z0-9_-]*"
                      maxLength={64}
                      required
                    />
                    <span>.jsonl</span>
                  </div>
                </label>
                <label>
                  <span>Reference model</span>
                  <select
                    value={referenceProvider}
                    onChange={(event) => setReferenceProvider(event.target.value as Provider)}
                  >
                    {PROVIDERS.map((provider) => (
                      <option value={provider.id} key={provider.id}>
                        {provider.name}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  className={overwrite ? 'checkbox-row checked' : 'checkbox-row'}
                  type="button"
                  onClick={() => setOverwrite((value) => !value)}
                >
                  <i>{overwrite && <Check size={13} />}</i>
                  <span>
                    <strong>Overwrite existing dataset</strong>
                    <small>Replace a JSONL file with the same name</small>
                  </span>
                </button>
                <div className="warning-note">
                  <AlertTriangle size={16} />
                  <span>
                    Generated references require human review. Incorrect or
                    unrepresentative ground truth makes results unreliable.
                  </span>
                </div>
                <button
                  className="secondary-button"
                  type="submit"
                  disabled={activeDataset || !datasetName.trim()}
                >
                  {activeDataset ? <i className="spinner dark" /> : <FilePlus2 size={16} />}
                  {activeDataset ? 'Generating dataset' : 'Generate ground truth'}
                </button>
              </div>
            </form>
          )}
        </section>
      )}

      {uploadMutation.isError && (
        <div className="error-banner" role="alert">
          <strong>Dataset upload failed.</strong>
          <span>{uploadMutation.error.message}</span>
        </div>
      )}
      {uploadMutation.data && (
        <JobProgress
          label={`${uploadMutation.data.record_count}-record dataset uploaded`}
          status="completed"
        />
      )}

      {datasetMutation.isError && (
        <div className="error-banner" role="alert">
          <strong>Dataset generation could not start.</strong>
          <span>{datasetMutation.error.message}</span>
        </div>
      )}
      {datasetStatus.data && (
        <JobProgress
          label="Ground-truth dataset"
          status={datasetStatus.data.status}
          error={datasetStatus.data.error}
        />
      )}

      {evaluationMutation.isError && (
        <div className="error-banner" role="alert">
          <strong>Evaluation could not start.</strong>
          <span>{evaluationMutation.error.message}</span>
        </div>
      )}
      {evaluationStatus.data && evaluationStatus.data.status !== 'completed' && (
        <JobProgress
          label="On-demand evaluation"
          status={evaluationStatus.data.status}
          error={evaluationStatus.data.error}
        />
      )}

      {completedRun ? (
        <EvaluationDashboard run={completedRun} />
      ) : (
        !activeEvaluation && (
          <div className="evaluation-empty-state">
            <div className="empty-visual">
              <BarChart3 size={27} />
              <i />
              <i />
              <i />
              <i />
            </div>
            <h2>Your results will appear here</h2>
            <p>
              Run an evaluation to compare model quality across five employee
              question categories, inspect every score, and review cost and latency.
            </p>
            <div>
              {CATEGORIES.map((category) => (
                <span key={category}>{category}</span>
              ))}
            </div>
          </div>
        )
      )}
    </div>
  )
}
