import assert from 'node:assert/strict'
import test from 'node:test'
import { createWranglerPreviewFailureDetector } from './wrangler-recovery'

const failure = `✘ [ERROR] A request to the Cloudflare API (/accounts/example/workers/scripts/portfolio-dev/edge-preview) failed.

  An unknown error has occurred. If this error persists, please file a report in workers-sdk (https://github.com/cloudflare/workers-sdk/issues) or reach out to your account team. [code: 10013]
`

test('recognizes the preview failure at every possible stream boundary once per child', () => {
  const colored = `\x1b[31m${failure}\x1b[0m`
  for (let split = 0; split <= colored.length; split++) {
    const detect = createWranglerPreviewFailureDetector()
    const results = [detect(colored.slice(0, split)), detect(colored.slice(split)), detect(failure)]
    assert.equal(results.filter(Boolean).length, 1, `split ${split}`)
    assert.equal(detect(failure), false)
  }
})

test('ignores other endpoints, codes, and errors between a request and its code', () => {
  for (const output of [
    failure.replace('/edge-preview', '/versions'),
    failure.replace('10013', '10000'),
    failure.replace('An unknown error', '[ERROR] Another request failed. An unknown error'),
    'application output: [code: 10013]',
  ]) {
    assert.equal(createWranglerPreviewFailureDetector()(output), false)
  }
})

test('retains bounded recent output and gives each replacement child a fresh detector', () => {
  const detect = createWranglerPreviewFailureDetector()
  assert.equal(detect('x'.repeat(20000)), false)
  assert.equal(detect(failure), true)
  assert.equal(detect(failure), false)
  assert.equal(createWranglerPreviewFailureDetector()(failure), true)
})
