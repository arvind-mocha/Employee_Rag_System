import { FormEvent, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  AlertTriangle,
  BarChart3,
  Check,
  CheckCircle2,
  ChevronRight,
  ChevronDown,
  CircleDollarSign,
  Database,
  FilePlus2,
  Gauge,
  Info,
  Layers3,
  Play,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Target,
  Trash2,
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
  categoryLabel,
  CATEGORIES,
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
  EvaluationResult,
  EvaluationRun,
  Provider,
  RetrievalOptions,
} from '../types/api'

const TERMINAL_JOB_STATES = new Set([
  'completed',
  'failed',
  'stopped',
  'canceled',
])

const EVALUATION_JOB_STORAGE_KEY = 'employee-rag-active-evaluation-job-v2'
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
  const storedValue = window.localStorage.getItem(DATASET_NAME_STORAGE_KEY)
  if (storedValue === null) return 'employee-policies'
  const value = storedValue
  return /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(value)
    ? value
    : ''
}

function storeDatasetName(value: string) {
  window.localStorage.setItem(DATASET_NAME_STORAGE_KEY, value)
}

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

const PASS_THRESHOLD = 0.8

function scoreIssueLabel(label: string, value: number | null | undefined) {
  if (value == null || value >= PASS_THRESHOLD) return null
  return `${label} was ${Math.round(value * 100)}%, below the 80% pass threshold.`
}

function answerProblems(result: EvaluationResult) {
  const problems = [
    scoreIssueLabel('Completeness', result.answer_scores?.completeness),
    scoreIssueLabel('Relevance', result.answer_scores?.relevance),
  ].filter((problem): problem is string => Boolean(problem))

  if (result.error) problems.unshift(result.error)
  if (!result.answer && !result.error) problems.push('The model did not return an answer.')
  return problems
}

function CaseScore({
  label,
  value,
}: {
  label: string
  value: number | null | undefined
}) {
  const belowThreshold = value != null && value < PASS_THRESHOLD
  return (
    <div className={belowThreshold ? 'case-score below-threshold' : 'case-score'}>
      <span>{label}</span>
      <strong>{score100(value)}</strong>
      <div className="case-score-track" aria-hidden="true">
        <i style={{ width: `${Math.max(0, Math.min(100, (value ?? 0) * 100))}%` }} />
      </div>
    </div>
  )
}

function EvaluationDashboard({ run }: { run: EvaluationRun }) {
  const [selectedCase, setSelectedCase] = useState<EvaluationResult | null>(null)
  const chartData = useMemo(
    () =>
      CATEGORIES.filter((category) =>
        run.results.some(
          (result) => (result.category ?? 'direct_fact') === category.id,
        ),
      ).map((category) => {
        const row: Record<string, string | number | null> = {
          category: category.label,
          shortCategory: category.shortLabel,
        }
        for (const summary of run.summaries) {
          const matching = run.results.filter(
            (result) =>
              result.answer_provider === summary.provider &&
              (result.category ?? 'direct_fact') === category.id &&
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
  const completeness = averageResultScore(
    run.results,
    (result) => result.answer_scores?.completeness,
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
            detail="Relevance and completeness"
            accent
          />
          <MetricCard
            icon={<CheckCircle2 size={18} />}
            label="Completeness"
            value={score100(completeness)}
            detail="Required information included"
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
            <Info size={14} /> The overall score gives equal weight to relevance
            and completeness.
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
                <span>Relevance</span>
                <b>{percent(summary.average_scores.relevance)}</b>
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
            <p>Select a case to inspect its answer and understand what needs improvement.</p>
          </div>
          <span className="row-count">{run.results.length} cases</span>
        </div>
        <div className="evaluation-case-list">
          <div className="evaluation-case-head" aria-hidden="true">
            <span>Question</span>
            <span>Model</span>
            <span>Overall</span>
            <span>Issues</span>
            <span>Latency</span>
          </div>
          {run.results.map((result) => {
            const issueCount = result.missing_facts.length
            return (
              <button
                type="button"
                className="evaluation-case-row"
                key={`${result.record_id}-${result.answer_provider}`}
                onClick={() => setSelectedCase(result)}
                aria-label={`View answer for: ${result.question}`}
              >
                <span className="evaluation-case-question">
                  <strong>{result.question}</strong>
                  <small>
                    <span className="category-pill">{categoryLabel(result.category)}</span>
                    {result.record_id}
                  </small>
                </span>
                <span className="evaluation-case-model">
                  <i style={{ background: PROVIDER_COLORS[result.answer_provider] }} />
                  {providerName(result.answer_provider)}
                </span>
                <strong className="evaluation-case-score">
                  {score100(result.answer_scores?.overall)}
                </strong>
                <span className={issueCount || result.error ? 'evaluation-case-issues has-issues' : 'evaluation-case-issues'}>
                  {result.error
                    ? 'Evaluation error'
                    : issueCount
                      ? `${issueCount} ${issueCount === 1 ? 'issue' : 'issues'}`
                      : 'No flagged issues'}
                </span>
                <span className="evaluation-case-latency">
                  {milliseconds(result.timing.total_ms)}
                </span>
                <ChevronRight className="evaluation-case-arrow" size={17} />
              </button>
            )
          })}
        </div>
      </section>

      {selectedCase && (
        <div
          className="case-detail-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.currentTarget === event.target) setSelectedCase(null)
          }}
        >
          <aside
            className="case-detail-panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="case-detail-title"
          >
            <div className="case-detail-header">
              <div>
                <span className="eyebrow">Evaluation case</span>
                <h3 id="case-detail-title">{selectedCase.question}</h3>
                <p>
                  {providerName(selectedCase.answer_provider)} · {categoryLabel(selectedCase.category)} ·{' '}
                  {milliseconds(selectedCase.timing.total_ms)}
                </p>
              </div>
              <button type="button" onClick={() => setSelectedCase(null)} aria-label="Close case details">
                <X size={18} />
              </button>
            </div>

            <div className="case-detail-body">
              <section className="case-answer-block">
                <span>Generated answer</span>
                <p>{selectedCase.answer || 'No answer was returned by the model.'}</p>
              </section>

              <section className="case-diagnosis">
                <div className="case-detail-section-title">
                  <AlertTriangle size={17} />
                  <div>
                    <span>Diagnosis</span>
                    <h4>What is wrong with this answer?</h4>
                  </div>
                </div>
                {answerProblems(selectedCase).length ? (
                  <ul className="case-problem-list">
                    {answerProblems(selectedCase).map((problem) => (
                      <li key={problem}>{problem}</li>
                    ))}
                  </ul>
                ) : (
                  <div className="case-no-problems">
                    <CheckCircle2 size={17} />
                    No threshold failures were found for this answer.
                  </div>
                )}

                <div className="case-score-grid">
                  <CaseScore label="Overall" value={selectedCase.answer_scores?.overall} />
                  <CaseScore label="Completeness" value={selectedCase.answer_scores?.completeness} />
                  <CaseScore label="Relevance" value={selectedCase.answer_scores?.relevance} />
                </div>
                <p className="case-threshold-note">Completeness and relevance must both be at least 80% to pass.</p>
              </section>

              {selectedCase.missing_facts.length > 0 && (
                <div className="case-issue-columns single">
                  <section>
                    <span>Missing facts</span>
                    <ul>{selectedCase.missing_facts.map((fact) => <li key={fact}>{fact}</li>)}</ul>
                  </section>
                </div>
              )}

              {selectedCase.rationale && (
                <section className="case-copy-block">
                  <span>Judge explanation</span>
                  <p>{selectedCase.rationale}</p>
                </section>
              )}

              {selectedCase.warnings.length > 0 && (
                <section className="case-warning-block">
                  <span>Warnings</span>
                  <ul>{selectedCase.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
                </section>
              )}
            </div>
          </aside>
        </div>
      )}
    </div>
  )
}

export function EvaluationPage() {
  const datasetBuilderRef = useRef<HTMLElement>(null)
  const [datasetName, setDatasetName] = useState(readStoredDatasetName)
  const [answerProviders, setAnswerProviders] = useState<Provider[]>([
    'gemini',
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
  const [questionsPerCategory, setQuestionsPerCategory] = useState(1)
  const [referenceProvider, setReferenceProvider] =
    useState<Provider>('gemini')
  const [overwrite, setOverwrite] = useState(false)
  const [datasetJob, setDatasetJob] = useState<EvaluationJobAccepted | null>(
    () => readStoredJob(DATASET_JOB_STORAGE_KEY),
  )
  const [datasetDeleteTarget, setDatasetDeleteTarget] = useState<string | null>(
    null,
  )

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

  const deleteDatasetMutation = useMutation({
    mutationFn: api.deleteDataset,
    onSuccess: (_, deletedName) => {
      setDatasetDeleteTarget(null)
      if (datasetName.trim() === deletedName) {
        setDatasetName('')
        storeDatasetName('')
      }
      storeJob(DATASET_JOB_STORAGE_KEY, null)
      setDatasetJob(null)
      datasetMutation.reset()
      storeJob(EVALUATION_JOB_STORAGE_KEY, null)
      setEvaluationJob(null)
      evaluationMutation.reset()
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
    if (!datasetName.trim()) return
    storeDatasetName(datasetName.trim())
    storeJob(DATASET_JOB_STORAGE_KEY, null)
    setDatasetJob(null)
    datasetMutation.mutate({
      dataset_name: datasetName.trim(),
      questions_per_category: questionsPerCategory,
      reference_provider: referenceProvider,
      overwrite,
      ...retrieval,
    })
  }

  const toggleDatasetBuilder = () => {
    if (showDatasetBuilder) {
      setShowDatasetBuilder(false)
      return
    }
    setShowDatasetBuilder(true)
    window.requestAnimationFrame(() => {
      datasetBuilderRef.current?.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      })
    })
  }

  const activeEvaluation = evaluationMutation.isPending || Boolean(
    evaluationStatus.data && !TERMINAL_JOB_STATES.has(evaluationStatus.data.status),
  )
  const activeDataset = datasetMutation.isPending || Boolean(
    datasetStatus.data && !TERMINAL_JOB_STATES.has(datasetStatus.data.status),
  )

  return (
    <div className="page evaluation-page">
      <div className="evaluation-hero">
        <div>
          <span className="eyebrow">
            <BarChart3 size={14} /> On-demand quality lab
          </span>
          <h1>Evaluate answer quality</h1>
          <p>
            Check whether model answers are relevant and complete. Nothing runs
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
                <span>Evaluation data</span>
                <h2>Select an evaluation dataset</h2>
              </div>
              <button
                type="button"
                className="text-button"
                onClick={toggleDatasetBuilder}
              >
                {showDatasetBuilder ? <X size={15} /> : <Sparkles size={15} />}
                {showDatasetBuilder ? 'Close generator' : 'Configure generator'}
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
                onChange={(event) => {
                  setDatasetName(event.target.value)
                  setDatasetDeleteTarget(null)
                  deleteDatasetMutation.reset()
                }}
                placeholder="employee-policies"
                pattern="[A-Za-z0-9][A-Za-z0-9_-]*"
                maxLength={64}
                required
              />
              <button
                className="dataset-delete-button"
                type="button"
                aria-label="Delete selected evaluation dataset"
                title="Delete selected dataset"
                onClick={() => {
                  deleteDatasetMutation.reset()
                  setDatasetDeleteTarget(datasetName.trim())
                }}
                disabled={
                  !datasetName.trim() ||
                  activeDataset ||
                  activeEvaluation ||
                  deleteDatasetMutation.isPending
                }
              >
                <Trash2 size={15} />
              </button>
            </div>
            <small className="field-help">
              Enter the name of a generated dataset, or generate a new one.
            </small>
            {datasetDeleteTarget && (
              <div className="dataset-delete-confirmation" role="alert">
                <AlertTriangle size={17} />
                <span>
                  <strong>Delete {datasetDeleteTarget}.jsonl?</strong>
                  <small>
                    This permanently removes the generated evaluation dataset.
                    Ingested source documents are not affected.
                  </small>
                </span>
                <div>
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => setDatasetDeleteTarget(null)}
                    disabled={deleteDatasetMutation.isPending}
                  >
                    Cancel
                  </button>
                  <button
                    className="dataset-delete-confirm-button"
                    type="button"
                    onClick={() =>
                      deleteDatasetMutation.mutate(datasetDeleteTarget)
                    }
                    disabled={deleteDatasetMutation.isPending}
                  >
                    {deleteDatasetMutation.isPending ? (
                      <i className="spinner" />
                    ) : (
                      <Trash2 size={15} />
                    )}
                    {deleteDatasetMutation.isPending
                      ? 'Deleting dataset'
                      : 'Delete dataset'}
                  </button>
                </div>
              </div>
            )}
            {deleteDatasetMutation.isError && (
              <div className="dataset-delete-notice error" role="alert">
                <XCircle size={16} />
                <span>
                  <strong>Dataset could not be deleted.</strong>
                  <small>{deleteDatasetMutation.error.message}</small>
                </span>
              </div>
            )}
            {deleteDatasetMutation.isSuccess && (
              <div className="dataset-delete-notice success" role="status">
                <CheckCircle2 size={16} />
                <span>
                  <strong>Dataset deleted.</strong>
                  <small>{deleteDatasetMutation.data.message}</small>
                </span>
              </div>
            )}
            <div className="dataset-quality-warning" role="note">
              <AlertTriangle size={15} />
              <span>
                <strong>Dataset quality matters.</strong> Questions must be clear
                and representative for the evaluation results to be useful.
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
                The judge checks whether each answer is relevant to the question
                and includes all information needed from the retrieved documents.
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
            disabled={
              activeEvaluation ||
              !datasetName.trim() ||
              Boolean(datasetDeleteTarget) ||
              deleteDatasetMutation.isPending
            }
          >
            {activeEvaluation ? <i className="spinner" /> : <Play size={17} />}
            {activeEvaluation ? 'Evaluation running' : 'Run evaluation'}
          </button>
        </div>
      </form>

      {showDatasetBuilder && (
        <section className="dataset-builder" ref={datasetBuilderRef}>
          <div className="dataset-builder-heading">
            <div className="builder-icon">
              <FilePlus2 size={20} />
            </div>
            <div>
              <span>Evaluation dataset</span>
              <h2>Generate evaluation cases</h2>
              <p>
                The app reads the ingested corpus and creates a balanced,
                categorized evaluation dataset on demand.
              </p>
            </div>
          </div>
          <form className="builder-grid" onSubmit={generateDataset}>
            <div className="questions-field">
              <span>Balanced category coverage</span>
              <div className="generated-category-grid">
                {CATEGORIES.map((category) => (
                  <span key={category.id}>{category.label}</span>
                ))}
              </div>
              <label className="question-count-field">
                <span>Questions per category</span>
                <input
                  type="number"
                  min={1}
                  max={5}
                  value={questionsPerCategory}
                  onChange={(event) => {
                    const value = event.target.valueAsNumber
                    if (Number.isInteger(value) && value >= 1 && value <= 5) {
                      setQuestionsPerCategory(value)
                    }
                  }}
                />
              </label>
              <small>
                {questionsPerCategory * CATEGORIES.length} questions will be
                generated from stored ingestion chunks. No generation runs until
                you select Generate dataset.
              </small>
            </div>
            <div className="builder-options">
              <label>
                <span>Dataset name</span>
                <div className="builder-filename">
                  <input
                    value={datasetName}
                    onChange={(event) => {
                      setDatasetName(event.target.value)
                      setDatasetDeleteTarget(null)
                      deleteDatasetMutation.reset()
                    }}
                    pattern="[A-Za-z0-9][A-Za-z0-9_-]*"
                    maxLength={64}
                    required
                  />
                </div>
              </label>
              <label>
                <span>Dataset generator model</span>
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
                  <small>Replace a generated dataset with the same name</small>
                </span>
              </button>
              <div className="warning-note">
                <AlertTriangle size={16} />
                <span>
                  Generated questions and categories require human review.
                  Unclear cases can make evaluation results unreliable.
                </span>
              </div>
              <button
                className="secondary-button"
                type="submit"
                disabled={
                  activeDataset ||
                  !datasetName.trim() ||
                  Boolean(datasetDeleteTarget) ||
                  deleteDatasetMutation.isPending
                }
              >
                {activeDataset ? <i className="spinner dark" /> : <FilePlus2 size={16} />}
                {activeDataset ? 'Generating dataset' : 'Generate dataset'}
              </button>
            </div>
          </form>
        </section>
      )}

      {datasetMutation.isError && (
        <div className="error-banner" role="alert">
          <strong>Dataset generation could not start.</strong>
          <span>{datasetMutation.error.message}</span>
        </div>
      )}
      {datasetStatus.data && (
        <JobProgress
          label="Evaluation dataset"
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
              Run an evaluation to compare model quality across retrieval and
              reasoning categories, inspect every score, and review cost and latency.
            </p>
            <div>
              {CATEGORIES.map((category) => (
                <span key={category.id}>{category.label}</span>
              ))}
            </div>
          </div>
        )
      )}
    </div>
  )
}
