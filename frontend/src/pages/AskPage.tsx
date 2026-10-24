import { FormEvent, useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  ArrowRight,
  BookOpenCheck,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock3,
  Coins,
  Copy,
  FileText,
  KeyRound,
  Search,
  Send,
  Settings2,
  Sparkles,
  TerminalSquare,
} from 'lucide-react'
import { StrategyControls } from '../components/StrategyControls'
import { api } from '../lib/api'
import {
  compactNumber,
  DEFAULT_RETRIEVAL,
  milliseconds,
  money,
  PROVIDERS,
  providerName,
} from '../lib/format'
import type {
  AnswerRequest,
  AnswerResponse,
  Provider,
  RetrievalOptions,
} from '../types/api'

type ResultTab = 'answer' | 'request'

const EXAMPLE_QUESTIONS = [
  'How many vacation days do employees receive?',
  'What health benefits are available?',
  'What is the remote work policy?',
]

function buildRequest(
  question: string,
  provider: Provider,
  retrieval: RetrievalOptions,
): AnswerRequest {
  return {
    question: question.trim(),
    ...retrieval,
    hf: provider === 'huggingface',
    openai: provider === 'openai',
    anthropic: provider === 'anthropic',
    grok: provider === 'grok',
  }
}

function AnswerPanel({ response }: { response: AnswerResponse }) {
  const [copied, setCopied] = useState(false)

  const copyAnswer = async () => {
    await navigator.clipboard.writeText(response.answer)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1600)
  }

  return (
    <section className="answer-panel">
      <div className="answer-heading">
        <div className="answer-avatar">
          <Sparkles size={18} />
        </div>
        <div>
          <span>{response.retrieval_used ? 'Grounded answer' : 'Direct answer'}</span>
          <small>
            {providerName(response.provider)} ·{' '}
            {response.retrieval_used
              ? `${response.sources.length} sources`
              : 'knowledge search skipped'}
          </small>
        </div>
        <button className="icon-button" type="button" onClick={copyAnswer}>
          {copied ? <CheckCircle2 size={17} /> : <Copy size={17} />}
          <span className="sr-only">Copy answer</span>
        </button>
      </div>
      <div className="answer-copy">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{response.answer}</ReactMarkdown>
      </div>
      {response.sources.length > 0 && (
        <div className="answer-source-strip">
          <BookOpenCheck size={16} />
          <span>Answer generated from retrieved employee documents</span>
          <span className="source-count">{response.sources.length}</span>
        </div>
      )}
    </section>
  )
}

function ChunkPanel({ response }: { response: AnswerResponse }) {
  const [expandedChunk, setExpandedChunk] = useState<number | null>(null)

  if (!response.sources.length) {
    return (
      <section className="developer-chunks">
        <div className="developer-chunks-heading">
          <div>
            <span>Retrieved evidence</span>
            <small>
              {response.retrieval_used
                ? 'The knowledge search returned no relevant chunks.'
                : 'The model answered without using the knowledge-search tool.'}
            </small>
          </div>
        </div>
      </section>
    )
  }

  return (
    <section className="developer-chunks">
      <div className="developer-chunks-heading">
        <div>
          <span>Retrieved evidence</span>
          <small>Open a chunk to inspect its content and retrieval metadata.</small>
        </div>
        <b>{response.sources.length} chunks</b>
      </div>
      <div className="chunk-list">
        {response.sources.map((chunk, index) => {
          const expanded = expandedChunk === index
          const panelId = `retrieved-chunk-${index + 1}`
          const relevance = Math.max(
            0,
            Math.min(100, chunk.relevance_score * 100),
          )

          return (
            <article
              className={expanded ? 'chunk-card expanded' : 'chunk-card'}
              key={`${chunk.chunk_id ?? 'chunk'}-${index}`}
            >
              <button
                className="chunk-summary"
                type="button"
                aria-expanded={expanded}
                aria-controls={panelId}
                onClick={() => setExpandedChunk(expanded ? null : index)}
              >
                <span className="chunk-rank">Chunk {index + 1}</span>
                <span className="chunk-source">
                  <FileText size={14} />
                  {chunk.source_file ?? 'Unknown source'}
                </span>
                <span className="chunk-summary-score">
                  <span>Relevance</span>
                  <i aria-hidden="true">
                    <b style={{ width: `${relevance}%` }} />
                  </i>
                  <strong>{Math.round(relevance)}%</strong>
                </span>
                <ChevronDown className="chunk-chevron" size={17} />
              </button>
              {expanded && (
                <div className="chunk-details" id={panelId} role="region">
                  <div className="chunk-information-grid">
                    <div>
                      <span>Retrieval rank</span>
                      <strong>{index + 1}</strong>
                    </div>
                    <div>
                      <span>Relevance score</span>
                      <strong>{chunk.relevance_score.toFixed(4)}</strong>
                    </div>
                    <div>
                      <span>Source file</span>
                      <strong>{chunk.source_file ?? 'Not available'}</strong>
                    </div>
                    <div>
                      <span>Source type</span>
                      <strong>{chunk.source_type ?? 'Not available'}</strong>
                    </div>
                    <div>
                      <span>Section</span>
                      <strong>{chunk.section ?? 'Not available'}</strong>
                    </div>
                    <div>
                      <span>Stored chunk index</span>
                      <strong>{chunk.chunk_index ?? 'Not available'}</strong>
                    </div>
                  </div>
                  <div className="chunk-identifier">
                    <span>Chunk ID</span>
                    <code>{chunk.chunk_id ?? 'Not available'}</code>
                  </div>
                  <div className="chunk-content">
                    <span>Full chunk content</span>
                    <p>{chunk.content}</p>
                  </div>
                </div>
              )}
            </article>
          )
        })}
      </div>
    </section>
  )
}

function RequestPanel({
  request,
  response,
  latency,
}: {
  request: AnswerRequest
  response: AnswerResponse
  latency: number
}) {
  const details = [
    [
      'Effective query',
      response.retrieval_used ? response.effective_query : 'Not used',
    ],
    ['Knowledge search', response.retrieval_used ? 'Used' : 'Skipped'],
    ['Provider', providerName(response.provider)],
    ['Model', response.model],
    ['Latency', milliseconds(latency)],
    ['Input tokens', response.usage ? compactNumber(response.usage.input_tokens) : '—'],
    ['Output tokens', response.usage ? compactNumber(response.usage.output_tokens) : '—'],
    ['Approx. cost', money(response.usage?.estimated_cost_usd)],
    ['Token source', response.usage?.token_count_source ?? '—'],
  ]

  return (
    <div className="request-panel">
      <div className="detail-grid">
        {details.map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
      <div className="strategy-summary">
        <span>Applied strategies</span>
        <div>
          {response.applied_strategies.map((strategy) => (
            <i key={strategy}>{strategy.replaceAll('_', ' ')}</i>
          ))}
          {!response.applied_strategies.length && <i>none — tool not used</i>}
        </div>
      </div>
      <details className="request-json">
        <summary>
          <TerminalSquare size={16} /> Request payload
        </summary>
        <pre>{JSON.stringify(request, null, 2)}</pre>
      </details>
    </div>
  )
}

export function AskPage({ developerMode }: { developerMode: boolean }) {
  const [question, setQuestion] = useState('')
  const [providerOverride, setProviderOverride] = useState<Provider | null>(null)
  const [retrieval, setRetrieval] =
    useState<RetrievalOptions>(DEFAULT_RETRIEVAL)
  const [resultTab, setResultTab] = useState<ResultTab>('answer')
  const [lastRequest, setLastRequest] = useState<AnswerRequest | null>(null)
  const [latency, setLatency] = useState(0)

  const credentialsQuery = useQuery({
    queryKey: ['provider-credentials'],
    queryFn: api.listProviderCredentials,
  })
  const configuredProviders = useMemo(
    () =>
      new Set(
        credentialsQuery.data?.credentials.map((credential) => credential.provider) ?? [],
      ),
    [credentialsQuery.data?.credentials],
  )
  const defaultConfiguredProvider = PROVIDERS.find((item) =>
    configuredProviders.has(item.id),
  )?.id
  const provider =
    providerOverride && configuredProviders.has(providerOverride)
      ? providerOverride
      : defaultConfiguredProvider ?? 'huggingface'

  const answerMutation = useMutation({
    mutationFn: async (request: AnswerRequest) => {
      const startedAt = performance.now()
      const response = await api.answer(request)
      return { response, latency: performance.now() - startedAt }
    },
    onSuccess: (result) => {
      setLatency(result.latency)
      setResultTab('answer')
    },
  })

  const response = answerMutation.data?.response
  const hasConfiguredProvider =
    !credentialsQuery.isSuccess || configuredProviders.size > 0
  const canSubmit =
    question.trim().length > 0 &&
    !answerMutation.isPending &&
    hasConfiguredProvider

  const providerMeta = useMemo(
    () => PROVIDERS.find((item) => item.id === provider)!,
    [provider],
  )

  const submit = (event: FormEvent) => {
    event.preventDefault()
    if (!canSubmit) return
    const request = buildRequest(question, provider, retrieval)
    setLastRequest(request)
    answerMutation.mutate(request)
  }

  const visibleResultTab = developerMode ? resultTab : 'answer'

  return (
    <div className="page ask-page">
      <div className="ask-intro">
        <div>
          <span className="eyebrow">
            <i className="status-dot" /> Employee knowledge assistant
          </span>
          <h1>What can we help you find?</h1>
          <p>
            Ask about workplace policies, benefits, time off, and employee
            information. Answers are grounded in your organization’s documents.
          </p>
        </div>
        {developerMode && (
          <div className="dev-mode-badge">
            <Settings2 size={15} /> Developer controls active
          </div>
        )}
      </div>

      <div className={developerMode ? 'ask-layout developer' : 'ask-layout'}>
        <div className="ask-main">
          {credentialsQuery.isSuccess && configuredProviders.size === 0 && (
            <div className="provider-required-banner">
              <KeyRound size={18} />
              <span>
                <strong>Connect a model provider to ask questions</strong>
                <small>Add an API key, model name, and automatic expiry time.</small>
              </span>
              <Link to="/settings">Open settings</Link>
            </div>
          )}
          <form className="question-box" onSubmit={submit}>
            <Search size={20} />
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask a question about your workplace…"
              rows={2}
              maxLength={1000}
              aria-label="Employee knowledge question"
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  event.currentTarget.form?.requestSubmit()
                }
              }}
            />
            <button className="send-button" type="submit" disabled={!canSubmit}>
              {answerMutation.isPending ? <i className="spinner" /> : <Send size={18} />}
              <span>Ask</span>
            </button>
          </form>

          {!response && !answerMutation.isPending && !answerMutation.isError && (
            <div className="suggestions">
              <span>Try asking</span>
              <div>
                {EXAMPLE_QUESTIONS.map((example) => (
                  <button type="button" key={example} onClick={() => setQuestion(example)}>
                    {example}
                    <ArrowRight size={14} />
                  </button>
                ))}
              </div>
            </div>
          )}

          {answerMutation.isPending && (
            <div className="loading-answer">
              <div className="loading-orbit">
                <Sparkles size={20} />
              </div>
              <div>
                <strong>Searching employee knowledge</strong>
                <span>Retrieving evidence and preparing a grounded answer…</span>
              </div>
            </div>
          )}

          {answerMutation.isError && (
            <div className="error-banner" role="alert">
              <strong>We couldn’t generate an answer.</strong>
              <span>{answerMutation.error.message}</span>
              <button type="button" onClick={() => answerMutation.reset()}>
                Dismiss
              </button>
            </div>
          )}

          {response && lastRequest && (
            <div className="results-card">
              {developerMode && (
                <div className="result-tabs" role="tablist">
                  <button
                    type="button"
                    className={resultTab === 'answer' ? 'active' : ''}
                    onClick={() => setResultTab('answer')}
                  >
                    Answer
                  </button>
                  <button
                    type="button"
                    className={resultTab === 'request' ? 'active' : ''}
                    onClick={() => setResultTab('request')}
                  >
                    Request details
                  </button>
                </div>
              )}
              {visibleResultTab === 'answer' && (
                <>
                  <AnswerPanel response={response} />
                  {developerMode && <ChunkPanel response={response} />}
                </>
              )}
              {visibleResultTab === 'request' && (
                <RequestPanel request={lastRequest} response={response} latency={latency} />
              )}
            </div>
          )}
        </div>

        {developerMode && (
          <aside className="developer-sidebar">
            <section className="control-card">
              <div className="control-heading">
                <div>
                  <span>Answer model</span>
                  <small>Choose one provider</small>
                </div>
                <i className="cheap-badge">HF default</i>
              </div>
              <div className="provider-radio-list">
                {PROVIDERS.map((item) => {
                  const configured = configuredProviders.has(item.id)
                  return (
                    <button
                      type="button"
                      key={item.id}
                      className={provider === item.id ? 'selected' : ''}
                      onClick={() => setProviderOverride(item.id)}
                      disabled={credentialsQuery.isSuccess && !configured}
                    >
                      <i>{item.shortName}</i>
                      <span>
                        <strong>{item.name}</strong>
                        <small>
                          {configured ? item.description : 'Not configured'}
                        </small>
                      </span>
                      <b />
                    </button>
                  )
                })}
              </div>
            </section>

            <section className="control-card">
              <div className="control-heading">
                <div>
                  <span>Retrieval strategy</span>
                  <small>Configure the evidence search</small>
                </div>
              </div>
              <StrategyControls value={retrieval} onChange={setRetrieval} />
            </section>

            <div className="sidebar-note">
              <Clock3 size={15} />
              <span>
                Requests use <strong>{providerMeta.name}</strong>. Cost and token
                details appear after completion.
              </span>
              <ChevronRight size={14} />
            </div>
            <div className="sidebar-note subtle">
              <Coins size={15} />
              <span>Evaluation only runs when started from the Evaluation page.</span>
            </div>
          </aside>
        )}
      </div>
    </div>
  )
}
