import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import ts from 'typescript'

// Use the project's TypeScript compiler; no test-runtime dependency is needed.
const source = readFileSync(new URL('../src/lib/providerConfiguration.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ES2022, target: ts.ScriptTarget.ES2022 } }).outputText
const { providerConfiguration, evaluationConfigurationError } = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`
)
const configured = (provider = 'openai') => ({
  provider, model_name: 'example-model', expires_at: new Date(Date.now() + 3_600_000).toISOString(),
  status: 'configured', message: 'Saved API key and model configured.',
})

test('a saved unexpired key and model are configured', () => {
  assert.equal(providerConfiguration('openai', [configured()]).status, 'configured')
})
test('missing providers and failed status requests are unavailable', () => {
  assert.equal(providerConfiguration('openai', []).status, 'unavailable')
  assert.equal(providerConfiguration('openai', [configured()], true).status, 'unavailable')
})
test('expired cached credentials are rejected even before the next request', () => {
  const record = configured()
  assert.equal(providerConfiguration('openai', [record], false, Date.parse(record.expires_at)).status, 'expired')
})
test('a missing model never enables a provider', () => {
  assert.equal(providerConfiguration('openai', [{ ...configured(), model_name: ' ' }]).status, 'not_configured')
})
test('missing or invalid expiry is unavailable', () => {
  assert.equal(providerConfiguration('openai', [{ ...configured(), expires_at: null }]).status, 'unavailable')
})
test('evaluation requires datasets, answer providers and a judge', () => {
  assert.match(evaluationConfigurationError([], [], '', []), /dataset/i)
  assert.match(evaluationConfigurationError(['dataset'], [], '', []), /answer provider/i)
  assert.match(evaluationConfigurationError(['dataset'], ['openai'], '', []), /judge/i)
})
test('configured answer and judge providers allow a run', () => {
  assert.equal(evaluationConfigurationError(['dataset'], ['openai'], 'anthropic', [configured(), configured('anthropic')]), null)
})
test('unconfigured answers and judges are identified without changing selections', () => {
  const answers = ['openai', 'grok']
  assert.match(evaluationConfigurationError(['dataset'], answers, 'anthropic', [configured(), configured('anthropic')]), /Grok/)
  assert.deepEqual(answers, ['openai', 'grok'])
  assert.match(evaluationConfigurationError(['dataset'], ['openai'], 'anthropic', [configured()]), /Anthropic/)
})
test('deletion and replacement change provider usability', () => {
  const deleted = { ...configured(), status: 'not_configured', message: 'API key not configured. Configure this provider in Settings before running evaluations.' }
  assert.match(evaluationConfigurationError(['dataset'], ['openai'], 'openai', [deleted]), /not configured/)
  assert.equal(evaluationConfigurationError(['dataset'], ['openai'], 'openai', [configured()]), null)
})
