import assert from 'node:assert/strict'
import test from 'node:test'
import { renderToString } from 'preact-render-to-string'
import type { ArenaReaderArtifact } from '../../util/arena-reader'
import { CodeContent, highlightArenaSource } from './code'

test('source files receive syntax tokens without interpreting code as HTML', () => {
  const source = 'def greet():\n\t# café\n\treturn "<script>alert(1)</script> & text"\n\n'
  const result = highlightArenaSource(source, 'model_runner.py')
  assert.equal(result.language, 'py')
  assert.match(result.html, /hljs-keyword/)
  assert.match(result.html, /hljs-string/)
  assert.match(result.html, /&lt;script&gt;/)
  assert.doesNotMatch(result.html, /<script>/)
  assert.equal((result.html.match(/\n/g) ?? []).length, 4)
  assert.match(result.html, /\t/)
})

test('unknown languages and large files remain complete escaped text', () => {
  for (const [fileName, source] of [
    ['file.unknown', '<img src=x onerror=alert(1)>\n'],
    ['large.py', '# line\n'.repeat(30_000) + 'last_line = "<end>"\n'],
    ['empty.txt', ''],
    ['constructor', '<plain text>'],
    ['__proto__', '<plain text>'],
  ]) {
    const result = highlightArenaSource(source, fileName)
    assert.doesNotMatch(result.html, /<img|hljs-/)
    assert.equal(
      result.html.replaceAll('&lt;', '<').replaceAll('&gt;', '>').replaceAll('&quot;', '"'),
      source,
    )
  }
})

test('the code view uses a focusable pre and preserves a source artifact as code', () => {
  const artifact: Extract<ArenaReaderArtifact, { kind: 'code' }> = {
    schemaVersion: 1,
    articleId: 'article-v1-fixture',
    snapshotId: 'snapshot-fixture',
    title: 'Model runner',
    sourceUrl: 'https://github.com/owner/repo/blob/main/file.py',
    finalUrl: 'https://raw.githubusercontent.com/owner/repo/main/file.py',
    capturedAt: 0,
    profileVersion: 'github-source-1',
    fingerprint: '',
    resources: [],
    kind: 'code',
    fileName: 'file.py',
    code: 'import torch\n\nclass ModelRunner:\n    pass\n',
  }
  const html = renderToString(<CodeContent artifact={artifact} contentRef={{ current: null }} />)
  assert.match(html, /<pre tabindex="0" aria-label="Source code: file.py">/)
  assert.match(html, /<code data-language="py">/)
  assert.match(html, /hljs-keyword/)
  assert.match(html, /ModelRunner/)
  assert.match(html, /<figcaption>file.py<\/figcaption>/)
})
