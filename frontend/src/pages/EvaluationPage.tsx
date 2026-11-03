import { EvaluationProgress } from '../components/EvaluationProgress'
import { ParallelEvaluationSettings } from '../components/ParallelEvaluationSettings'
import { validEvaluationConcurrency } from '../lib/evaluationConcurrency'
import { EvaluationErrorDetails } from '../components/EvaluationErrorDetails'
import { evaluationCaseStatus, evaluationErrorLabel } from '../lib/evaluationStatus'
import { FormEvent, useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  ChevronDown,
  CircleDollarSign,
  Gauge,
  Info,
  Layers3,
  Play,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Target,
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
import { Link } from 'react-router-dom'
import { EvaluationProviderStatus, ProviderPicker } from '../components/EvaluationProviders'
import { EvaluationReference } from '../components/EvaluationReference'
import { useProviderConfiguration } from '../hooks/useProviderConfiguration'
import { evaluationConfigurationError, providerConfiguration, PROVIDER_STATUS_LABELS } from '../lib/providerConfiguration'
import { EvaluationDatasets } from '../components/EvaluationDatasets'
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
  EvaluationBatchRun,
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
const DATASET_SELECTION_KEY = 'employee-rag-evaluation-selection'
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
    job.job_type === 'evaluation_run' &&
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

function readStoredSelection(): string[] {
  try {
    const value: unknown = JSON.parse(window.localStorage.getItem(DATASET_SELECTION_KEY) ?? '[]')
    return Array.isArray(value) && value.every((id) => typeof id === 'string') ? value : []
  } catch { return [] }
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
      <strong>{value == null ? 'Unavailable' : score100(value)}</strong>
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
      {run.performance && <p className="field-help">
        Elapsed: {milliseconds(run.performance.elapsed_seconds * 1000)} · Maximum concurrent cases: {run.performance.maximum_concurrent_cases} · Retries: {run.performance.retry_count}
      </p>}
      <div className="dashboard-title-row">
        <div>
          <span className="eyebrow">Latest evaluation run</span>
          <h2>Evaluation results</h2>
          <p>
            {run.dataset_name} · {run.results.length} model answers · judged by{' '}
            {providerName(run.judge_provider)} · {run.status === 'partial' ? 'Completed with evaluation errors' : run.status ?? 'completed'}
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
            detail={`${passed} passed · ${evaluated - passed} failed · ${run.results.filter((result) => result.error).length} errored`}
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

      {run.error && <p role="alert" className="error-banner">{run.error}</p>}
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
                key={`${result.dataset_id}-${result.record_id}-${result.answer_provider}`}
                onClick={() => setSelectedCase(result)}
                aria-label={`View answer for: ${result.question}`}
              >
                <span className="evaluation-case-question">
                  <strong>{result.question}</strong>
                  <small>
                    <span className="category-pill">{categoryLabel(result.category)}</span>
                    {result.dataset_name} · {result.dataset_id?.slice(0, 8)} · {result.record_id} · {evaluationCaseStatus(result)}
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
                    ? evaluationErrorLabel(result)
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
              <EvaluationErrorDetails result={selectedCase} />
              <section className="case-answer-block">
                <span>Generated answer</span>
                <p>{selectedCase.answer || 'No answer was returned by the model.'}</p>
              </section>

              <section className="case-copy-block">
                <span>Reference answer · {selectedCase.dataset_name} · {selectedCase.dataset_id}</span>
                <p>{selectedCase.reference_answer}</p>
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
                  <CaseScore label="Correctness" value={selectedCase.answer_scores?.correctness} />
                  <CaseScore label="Groundedness" value={selectedCase.answer_scores?.groundedness} />
                  <CaseScore label="Citation validity" value={selectedCase.answer_scores?.citation_validity} />
                  <CaseScore label="Semantic similarity" value={selectedCase.answer_scores?.semantic_similarity} />
                  <CaseScore label="Lexical similarity" value={selectedCase.answer_scores?.lexical_similarity} />
                  <CaseScore label="Abstention" value={selectedCase.answer_scores?.abstention} />
                  <CaseScore label="Retrieval precision" value={selectedCase.retrieval_scores?.precision_at_k} />
                  <CaseScore label="Retrieval recall" value={selectedCase.retrieval_scores?.recall_at_k} />
                  <CaseScore label="MRR" value={selectedCase.retrieval_scores?.reciprocal_rank} />
                  <CaseScore label="MAP" value={selectedCase.retrieval_scores?.average_precision_at_k} />
                  <CaseScore label="NDCG" value={selectedCase.retrieval_scores?.ndcg_at_k} />
                  <CaseScore label="Source precision" value={selectedCase.retrieval_scores?.source_precision_at_k} />
                  <CaseScore label="Source recall" value={selectedCase.retrieval_scores?.source_recall_at_k} />
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

              {selectedCase.unsupported_claims.length > 0 && (
                <section className="case-warning-block">
                  <span>Unsupported claims</span>
                  <ul>{selectedCase.unsupported_claims.map((claim) => <li key={claim}>{claim}</li>)}</ul>
                </section>
              )}

              <section className="case-copy-block">
                <span>Retrieval and usage</span>
                <p>{selectedCase.retrieved_sources.join(', ') || 'No sources retrieved'}</p>
                <p>{compactNumber(selectedCase.usage.total_tokens)} tokens · {money(selectedCase.usage.estimated_cost_usd)} · Retrieval {milliseconds(selectedCase.timing.retrieval_ms)} · Answer {milliseconds(selectedCase.timing.answer_generation_ms)} · Grading {milliseconds(selectedCase.timing.grading_ms)}</p>
              </section>

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
  const [selection, setSelection] = useState(readStoredSelection)
  const [answerProviders, setAnswerProviders] = useState<Provider[]>(() => {
    try {
      const saved = JSON.parse(window.localStorage.getItem('employee-rag-answer-providers') ?? '[]')
      return Array.isArray(saved) ? saved.filter((id): id is Provider => PROVIDERS.some((provider) => provider.id === id)) : []
    } catch { return [] }
  })
  const [judgeProvider, setJudgeProvider] = useState<Provider | ''>(() => {
    const saved = window.localStorage.getItem('employee-rag-judge-provider')
    return PROVIDERS.some((provider) => provider.id === saved) ? saved as Provider : ''
  })
  const credentials = useProviderConfiguration()
  const [credentialValidationError, setCredentialValidationError] = useState<string | null>(null)
  const [checkingBeforeRun, setCheckingBeforeRun] = useState(false)
  const changeAnswerProviders = (providers: Provider[]) => {
    setAnswerProviders(providers)
    setCredentialValidationError(null)
    window.localStorage.setItem('employee-rag-answer-providers', JSON.stringify(providers))
  }
  const changeJudgeProvider = (provider: Provider | '') => {
    setJudgeProvider(provider)
    setCredentialValidationError(null)
    window.localStorage.setItem('employee-rag-judge-provider', provider)
  }
  const [concurrency, setConcurrency] = useState(() => {
    const saved = Number(window.localStorage.getItem('employee-rag-evaluation-concurrency') ?? 4)
    return validEvaluationConcurrency(saved) ? saved : 4
  })
  const [retrieval, setRetrieval] = useState<RetrievalOptions>(DEFAULT_RETRIEVAL)
  const [evaluationJob, setEvaluationJob] = useState<EvaluationJobAccepted | null>(
    () => readStoredJob(EVALUATION_JOB_STORAGE_KEY),
  )
  const [resultScope, setResultScope] = useState('aggregate')
  const library = useQuery({ queryKey: ['evaluation-datasets'], queryFn: api.listDatasets })
  const datasets = library.data?.datasets ?? []
  const selected = datasets.filter((dataset) =>
    selection.includes(dataset.dataset_id) && dataset.validation_status === 'valid',
  ).map((dataset) => dataset.dataset_id)
  const changeSelection = (ids: string[]) => {
    setSelection(ids)
    window.localStorage.setItem(DATASET_SELECTION_KEY, JSON.stringify(ids))
  }
  const evaluationMutation = useMutation({
    mutationFn: api.runEvaluation,
    onSuccess: (job) => {
      storeJob(EVALUATION_JOB_STORAGE_KEY, job)
      setEvaluationJob(job)
      setResultScope('aggregate')
    },
  })
  const evaluationStatus = useEvaluationJob(evaluationJob)
  const cancellationMutation = useMutation({
    mutationFn: api.cancelEvaluation,
    onSuccess: () => evaluationStatus.refetch(),
  })
  const completedBatch = ['completed', 'failed', 'stopped', 'canceled'].includes(evaluationStatus.data?.status ?? '') && evaluationStatus.data?.result
    ? evaluationStatus.data.result as unknown as EvaluationBatchRun : null
  const visibleRun = completedBatch
    ? resultScope === 'aggregate' ? completedBatch.aggregate
      : completedBatch.datasets.find((run) => run.dataset_id === resultScope) ?? completedBatch.aggregate
    : null
  const activeEvaluation = evaluationMutation.isPending || Boolean(evaluationJob && (
    !evaluationStatus.data || !TERMINAL_JOB_STATES.has(evaluationStatus.data.status)
  ) && !evaluationStatus.isError)
  const configurationError = evaluationConfigurationError(
    selected, answerProviders, judgeProvider, credentials.configurations,
  )
  const runEvaluation = async (event: FormEvent) => {
    event.preventDefault()
    if (activeEvaluation || checkingBeforeRun) return
    setCheckingBeforeRun(true)
    setCredentialValidationError(null)
    try {
      // Always re-read credentials immediately before submission. Keep all
      // selections intact when a key expires or the status request fails.
      const latest = await credentials.refetch()
      const configurations = PROVIDERS.map((provider) => providerConfiguration(
        provider.id, latest.data?.providers ?? [], latest.isError,
      ))
      const error = evaluationConfigurationError(selected, answerProviders, judgeProvider, configurations)
      if (error || !judgeProvider) {
        setCredentialValidationError(error)
        return
      }
      if (!validEvaluationConcurrency(concurrency)) {
        setCredentialValidationError('Choose a valid parallel evaluation setting.')
        return
      }
      evaluationMutation.mutate({
        dataset_ids: selected, answer_providers: answerProviders,
        judge_provider: judgeProvider, concurrency, ...retrieval,
      })
    } finally {
      setCheckingBeforeRun(false)
    }
  }

  return (
    <div className="page evaluation-page">
      <div className="evaluation-hero">
        <div>
          <span className="eyebrow"><BarChart3 size={14} /> On-demand quality lab</span>
          <h1>Evaluate answer quality</h1>
          <p>Upload your JSONL datasets, select the files to evaluate, and compare answer models.</p>
        </div>
        <div className="evaluation-assurance">
          <ShieldCheck size={19} /><div><strong>Your evaluation data</strong><span>Only selected files run when you click Run Evaluation</span></div>
        </div>
      </div>
      <form className="evaluation-config" onSubmit={runEvaluation}>
        <section className="config-primary">
          <div className="section-number">01</div>
          <div className="config-content">
            <div className="config-heading"><div><span>Evaluation Datasets</span><h2>Dataset library</h2></div></div>
            {library.isLoading && <p role="status">Loading datasets…</p>}
            {library.isError && <p role="alert" className="error-banner">{library.error.message}</p>}
            <EvaluationDatasets datasets={datasets} selected={selected} onChange={changeSelection} busy={activeEvaluation} />
          </div>
        </section>
        <section className="config-primary">
          <div className="section-number">02</div>
          <div className="config-content">
            <div className="config-heading"><div><span>JSONL Developer Reference</span><h2>Create your evaluation files</h2></div></div>
            <EvaluationReference />
          </div>
        </section>
        <section className="config-primary">
          <div className="section-number">03</div>
          <div className="config-content">
            <div className="config-heading"><div><span>LLM Provider Configuration</span><h2>API keys and models</h2></div></div>
            <EvaluationProviderStatus configurations={credentials.configurations} loading={credentials.isLoading}
              error={credentials.isError} onRetry={() => credentials.refetch()} />
          </div>
        </section>
        <section className="config-primary">
          <div className="section-number">04</div>
          <div className="config-content">
            <div className="config-heading"><div><span>Evaluation Configuration</span><h2>Choose answer providers</h2></div>
              <span className="selection-count">{answerProviders.length} selected</span></div>
            <ProviderPicker selected={answerProviders} onChange={changeAnswerProviders}
              configurations={credentials.configurations} loading={credentials.isLoading} />
            <ParallelEvaluationSettings value={concurrency} disabled={activeEvaluation || checkingBeforeRun} onChange={(value) => {
              setConcurrency(value)
              window.localStorage.setItem('employee-rag-evaluation-concurrency', String(value))
            }} />
          </div>
        </section>
        <section className="config-primary">
          <div className="section-number" aria-hidden="true" />
          <div className="config-content config-split">
            <div>
              <div className="config-heading"><div><span>Independent grader</span><h2>Choose the judge model</h2></div></div>
              <label className="select-field">
                <Sparkles size={17} />
                <select aria-label="Judge provider" value={judgeProvider} onChange={(event) => changeJudgeProvider(event.target.value as Provider | '')}>
                  <option value="">Choose a configured judge</option>
                  {PROVIDERS.map((provider) => {
                    const configuration = credentials.configurations.find((item) => item.provider === provider.id)!
                    return <option value={provider.id} key={provider.id} disabled={credentials.isLoading || configuration.status !== 'configured'}>
                      {provider.name} · {configuration.model_name || 'No model'} · {PROVIDER_STATUS_LABELS[configuration.status]}
                    </option>
                  })}
                </select><ChevronDown size={16} />
              </label>
              <small className="field-help">The judge grades answers against your references and retrieved evidence.</small>
            </div>
            <div className="retrieval-compact-panel">
              <div className="config-heading"><div><span>Evidence search</span><h2>Retrieval controls</h2></div></div>
              <StrategyControls value={retrieval} onChange={setRetrieval} compact />
            </div>
          </div>
        </section>
        <div className="run-bar">
          <div><Layers3 size={18} /><span><strong>{selected.length} dataset(s) selected</strong>
            {answerProviders.length} answer models · {judgeProvider ? providerName(judgeProvider) : 'No'} judge</span></div>
          <button className="primary-button" type="submit" disabled={activeEvaluation || checkingBeforeRun || credentials.isLoading || Boolean(configurationError)}>
            {activeEvaluation ? <i className="spinner" /> : <Play size={17} />}
            {activeEvaluation ? 'Evaluation running' : checkingBeforeRun ? 'Checking credentials…' : 'Run Evaluation'}
          </button>
        </div>
        {(credentialValidationError || configurationError) && <div className="evaluation-configuration-notice" role={credentialValidationError ? 'alert' : 'status'}>
          <span>{credentialValidationError || configurationError}</span> <Link to="/settings">Configure in Settings</Link>
        </div>}
      </form>
      {evaluationMutation.isError && <div className="error-banner" role="alert"><strong>Evaluation could not start.</strong><span>{evaluationMutation.error.message}</span></div>}
      {evaluationStatus.isError && <div className="error-banner" role="alert">{evaluationStatus.error.message}
        <button type="button" className="text-button" onClick={() => evaluationStatus.refetch()}>Retry status</button>
      </div>}
      {evaluationStatus.data && evaluationStatus.data.status !== 'completed' && <>
        <JobProgress label="Selected dataset evaluation" status={evaluationStatus.data.status} error={evaluationStatus.data.error} />
      </>}
      {cancellationMutation.isError && <p className="error-banner" role="alert">{cancellationMutation.error.message}</p>}
      {evaluationStatus.data && <EvaluationProgress job={evaluationStatus.data}
        onCancel={() => evaluationJob && cancellationMutation.mutate(evaluationJob.job_id)} cancelPending={cancellationMutation.isPending} />}

      {completedBatch && visibleRun ? <>
        <div className="evaluation-result-scope">
          <label htmlFor="result-dataset">Results for</label>
          <select id="result-dataset" value={resultScope} onChange={(event) => setResultScope(event.target.value)}>
            <option value="aggregate">Aggregated summary · {completedBatch.datasets.length} datasets</option>
            {completedBatch.datasets.map((run) => <option key={run.dataset_id} value={run.dataset_id ?? ''}>{run.dataset_name} · {run.dataset_id}</option>)}
          </select>
        </div>
        <EvaluationDashboard key={`${visibleRun.run_id}:${visibleRun.dataset_id ?? "aggregate"}`} run={visibleRun} />
      </> : !activeEvaluation && <div className="evaluation-empty-state">
        <BarChart3 size={27} /><h2>Your results will appear here</h2>
        <p>Select datasets and run an evaluation to inspect scores, individual answers, cost, and latency.</p>
      </div>}
    </div>
  )
}
