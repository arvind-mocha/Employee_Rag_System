import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import ts from 'typescript'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

async function load(relative) {
  const source = readFileSync(new URL(relative, import.meta.url), 'utf8')
  let compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.ES2022, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX },
  }).outputText
  compiled = compiled.replaceAll('"react/jsx-runtime"', JSON.stringify(import.meta.resolve('react/jsx-runtime')))
  return import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)
}
const { evaluationCaseStatus, evaluationErrorLabel } = await load('../src/lib/evaluationStatus.ts')
const { EvaluationErrorDetails } = await load('../src/components/EvaluationErrorDetails.tsx')

test('case statuses distinguish errors, low scores, success and unavailable scores', () => {
  assert.equal(evaluationCaseStatus({ error: 'Failed to authenticate', passed: null }), 'Errored')
  assert.equal(evaluationCaseStatus({ error: null, passed: false }), 'Failed')
  assert.equal(evaluationCaseStatus({ error: null, passed: true }), 'Passed')
  assert.equal(evaluationCaseStatus({ error: null, passed: null }), 'Unavailable')
})
test('case row labels display the error category and support older results', () => {
  assert.equal(evaluationErrorLabel({ error_category: 'provider_authentication' }), 'provider authentication')
  assert.equal(evaluationErrorLabel({}), 'Evaluation error')
})
test('expandable details render category, message, stage and correlation context safely', () => {
  const markup = renderToStaticMarkup(React.createElement(EvaluationErrorDetails, { result: {
    error: 'OpenAI rejected its configured API key. Replace it in Settings. <script>',
    error_category: 'provider_authentication', failed_stage: 'answer_generation',
    error_provider: 'openai', error_model: 'selected-model', dataset_id: 'dataset-1',
    record_id: 'record-2', run_id: 'run-3', job_id: 'job-4',
  } }))
  for (const text of ['provider authentication', 'Replace it in Settings.', 'answer_generation',
    'openai', 'selected-model', 'dataset-1', 'record-2', 'run-3', 'job-4']) assert.ok(markup.includes(text))
  assert.ok(markup.includes('role="alert"'))
  assert.ok(!markup.includes('<script>'))
})
test('successful and low-scoring cases do not render an error panel', () => {
  for (const passed of [true, false]) {
    assert.equal(renderToStaticMarkup(React.createElement(EvaluationErrorDetails, { result: { error: null, passed } })), '')
  }
})

const { EvaluationProgress } = await load('../src/components/EvaluationProgress.tsx')
const { validEvaluationConcurrency, EVALUATION_CONCURRENCY_OPTIONS } = await load('../src/lib/evaluationConcurrency.ts')

test('parallel evaluation exposes the required choices and rejects invalid UI values', () => {
  assert.deepEqual(EVALUATION_CONCURRENCY_OPTIONS.map((option) => option.value), [1, 2, 3, 4, 6, 8])
  assert.ok(EVALUATION_CONCURRENCY_OPTIONS.find((option) => option.value === 4).label.includes('Default'))
  for (const value of [1, 2, 3, 4, 6, 8]) assert.equal(validEvaluationConcurrency(value), true)
  for (const value of [0, 5, 7, 9, 2.5, NaN, '4', null]) assert.equal(validEvaluationConcurrency(value), false)
})
test('live progress renders counters, elapsed time and cancellation without classifying low scores as errors', () => {
  const markup = renderToStaticMarkup(React.createElement(EvaluationProgress, {
    job: { status: 'running', total_cases: 36, completed_cases: 20, running_cases: 4,
      successful_cases: 19, failed_cases: 1, low_scoring_cases: 3, elapsed_seconds: 123, retry_count: 2 },
    onCancel: () => {}, cancelPending: false,
  }))
  for (const text of ['Total: 36', 'Completed: 20', 'Running: 4', 'Successful: 19', 'Failed (errors): 1',
    '55.6%', '2m 3s', '3 scored cases', 'Retries: 2', 'Cancel evaluation']) assert.ok(markup.includes(text))
})
test('terminal progress disables cancellation and preserves incomplete progress after cancellation', () => {
  const markup = renderToStaticMarkup(React.createElement(EvaluationProgress, {
    job: { status: 'canceled', total_cases: 36, completed_cases: 4, running_cases: 0 },
    onCancel: () => {}, cancelPending: false,
  }))
  assert.ok(markup.includes('11.1%'))
  assert.ok(!markup.includes('Cancel evaluation'))
})
