import assert from 'node:assert/strict'
import test from 'node:test'
import { renderToString } from 'preact-render-to-string'
import type { ArenaReaderArtifact } from '../../util/arena-reader'
import { unescapeHTML } from '../../util/escape'
import { highlightSourceCode } from '../../util/highlight-source'
import { CodeContent } from './code'

const lineText = (html: string) =>
  html.split('\n').map(line => unescapeHTML(line.replace(/<[^>]*>/g, '')))

test('source files receive syntax roles without interpreting code as HTML', () => {
  const source = 'def greet():\n\t# café\n\treturn "<script>alert(1)</script> & text"\n\n'
  const result = highlightSourceCode(source, 'model_runner.py')
  assert.equal(result.language, 'python')
  assert.match(result.html, /data-syntax-role="keyword"/)
  assert.match(result.html, /&lt;script&gt;/)
  assert.doesNotMatch(result.html, /<script>/)
  // The final newline ends the file; the blank line before it keeps a spacer.
  assert.deepEqual(lineText(result.html), [
    'def greet():',
    '\t# café',
    '\treturn "<script>alert(1)</script> & text"',
    ' ',
  ])
})

test('unknown languages and large files remain complete escaped text', () => {
  for (const [fileName, source] of [
    ['file.unknown', '<img src=x onerror=alert(1)>\n'],
    ['large.py', '# line\n'.repeat(30_000) + 'last_line = "<end>"\n'],
    ['empty.txt', ''],
    ['constructor', '<plain text>'],
    ['__proto__', '<plain text>'],
  ]) {
    const result = highlightSourceCode(source, fileName)
    assert.doesNotMatch(result.html, /<img|data-syntax-role="(?!normal)/)
    const expected = source.replace(/\n$/, '').split('\n')
    assert.deepEqual(
      lineText(result.html),
      expected.map(line => line || ' '),
    )
  }
})

test('the code view uses a focusable pre and shows the source before highlighting loads', () => {
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
    code: 'import torch\n\nclass <ModelRunner>:\n    pass\n',
  }
  const html = renderToString(<CodeContent artifact={artifact} contentRef={{ current: null }} />)
  assert.match(html, /<pre tabindex="0" aria-label="Source code: file.py">/)
  assert.match(html, /class &lt;ModelRunner/)
  assert.doesNotMatch(html, /<ModelRunner>/)
  assert.match(html, /<figcaption>file.py<\/figcaption>/)
})
