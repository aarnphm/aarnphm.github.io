import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { createServer } from 'node:http'
import { test } from 'node:test'
import type { CodeQuery, CodeSearchManifest, CodeSourceShard, CodeUnit } from './code-search-types'
import { parseCodeQuery } from './code-query'
import { CodeSearchEngine } from './code-search-engine'
import { literalRanges } from './code-search-match'

const query: CodeQuery = { mode: 'literal', pattern: 'foo', caseSensitive: true }
const hash = (text: string) => createHash('sha256').update(text).digest('hex')

test('code queries preserve code syntax and only consume operators outside quotes', () => {
  assert.equal(parseCodeQuery('ordinary note'), null)
  assert.equal(parseCodeQuery('filter:images foo'), null)
  assert.deepEqual(parseCodeQuery('filter:py path:"courses/a b" "#include filter:images"'), {
    mode: 'literal',
    pattern: '#include filter:images',
    language: 'python',
    path: 'courses/a b',
    caseSensitive: true,
  })
  assert.equal(parseCodeQuery('filter:code "  a\n b  "')?.pattern, '  a\n b  ')
  assert.equal(
    parseCodeQuery('filter:py ast:def f():\n    return "filter:json"')?.pattern,
    'def f():\n    return "filter:json"',
  )
  assert.equal(parseCodeQuery('filter:code #include <x>')?.pattern, '#include <x>')
  assert.equal(parseCodeQuery('filter:py \n  foo\n  bar')?.pattern, '\n  foo\n  bar')
  assert.equal(parseCodeQuery('filter:code print("filter:json")')?.pattern, 'print("filter:json")')
  assert.equal(parseCodeQuery('filter:json')?.language, 'json')
  assert.equal(parseCodeQuery('filter:code \n  foo  \n  bar  ')?.pattern, '\n  foo  \n  bar  ')
  assert.equal(parseCodeQuery('filter:tsx foo')?.language, 'tsx')
  assert.equal(parseCodeQuery('filter: py foo')?.language, 'python')
  assert.equal(parseCodeQuery('filter:code "a" + "b"')?.pattern, '"a" + "b"')
  assert.equal(parseCodeQuery('filter:code `text filter:json value`')?.language, undefined)
})

test('literal ranges use original UTF-16 offsets and CRLF line positions', () => {
  const text = '😀İ\r\nfoo foo\r\nFOO'
  const ranges = literalRanges(text, 'foo', false)
  assert.deepEqual(
    ranges.map(r => [r.start.offset, r.end.offset, r.start.line, r.start.column]),
    [
      [5, 8, 2, 1],
      [9, 12, 2, 5],
      [14, 17, 3, 1],
    ],
  )
  assert.equal(literalRanges('aaaa', 'aa', true).length, 3)
  assert.equal(literalRanges(text, '', true).length, 0)
  assert.equal(literalRanges(text, 'foo', true).length, 2)
})

test('real HTTP retrieval validates shards, scopes listings, recovers generations and cancels', async () => {
  let generation = 'one'
  let broken = false
  let prune = false
  const requests: string[] = []
  const sources = [{ id: hash('foo foo\r\n😀foo'), text: 'foo foo\r\n😀foo' }]
  const units: CodeUnit[] = ['a', 'b'].map((owner, i) => ({
    id: owner,
    ownerSlug: owner,
    title: owner,
    path: `courses/${owner}.py`,
    kind: 'block',
    language: 'python',
    originalLanguage: 'py',
    sourceId: sources[0].id,
    anchor: `code-${i}`,
    startLine: 10,
    endLine: 11,
  }))
  units.push({ ...units[0], id: 'a2', path: 'courses/a-two.py' })
  const body = (gen: string, language: string) =>
    JSON.stringify({
      kind: 'quartz-code-sources-v1',
      generation: gen,
      language,
      sources,
    } satisfies CodeSourceShard)
  const manifest = (): CodeSearchManifest => ({
    kind: 'quartz-code-search-v1',
    version: 1,
    generation,
    units: [...units, { ...units[0], id: 'json', language: 'json', path: 'data.json' }],
    shards: ['python', 'json'].map(language => ({
      language,
      path: `${generation}/${language}.json`,
      byteLength: Buffer.byteLength(body(generation, language)),
      sha256: hash(body(generation, language)),
      sourceIds: [sources[0].id],
    })),
    coverage: { owners: 2, units: 4, sources: 1 },
  })
  const server = createServer((req, res) => {
    const url = req.url ?? ''
    requests.push(url)
    if (url.endsWith('manifest.json')) {
      res.end(JSON.stringify(manifest()))
      return
    }
    if (prune && url.includes('/one/')) {
      generation = 'two'
      res.writeHead(404)
      res.end()
      return
    }
    const language = url.includes('python') ? 'python' : 'json'
    const payload = body(url.includes('/one/') ? 'one' : generation, language)
    res.end(broken ? payload.replace('foo', 'bar') : payload)
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  assert.ok(address && typeof address !== 'string')
  const engine = new CodeSearchEngine(
    `http://127.0.0.1:${address.port}/static/code-search/manifest.json`,
  )
  try {
    const progress: number[] = []
    const result = await engine.search(query, new AbortController().signal, p =>
      progress.push(p.scannedUnits),
    )
    assert.equal(result.matches.length, 3)
    assert.equal(result.totalMatches, 9)
    assert.deepEqual(
      result.matches.map(match => match.unit.ownerSlug),
      ['a', 'b', 'a'],
    )
    assert.equal(result.matches[0].excerptStartLine, 10)
    assert.ok(progress.includes(3))
    assert.ok(requests.every(url => !url.endsWith('/json.json')))
    const listed = await engine.search(
      { ...query, pattern: '', path: '/a.py' },
      new AbortController().signal,
    )
    assert.equal(listed.matches.length, 1)
    assert.equal(listed.matches[0].ranges.length, 0)
    const json = await engine.search({ ...query, language: 'json' }, new AbortController().signal)
    assert.equal(json.matches.length, 1)
    assert.equal(
      requests.filter(url => url.endsWith('manifest.json')).length,
      1,
      'typing reuses the validated manifest',
    )
    const aborted = new AbortController()
    aborted.abort()
    await assert.rejects(engine.search(query, aborted.signal), { name: 'AbortError' })
    const fresh = new CodeSearchEngine(
      `http://127.0.0.1:${address.port}/static/code-search/manifest.json`,
    )
    await fresh.search({ ...query, language: 'python' }, new AbortController().signal)
    prune = true
    assert.equal(
      (await fresh.search({ ...query, language: 'json' }, new AbortController().signal)).generation,
      'two',
    )
    const expiring = new CodeSearchEngine(
      `http://127.0.0.1:${address.port}/static/code-search/manifest.json`,
      0,
    )
    assert.equal((await expiring.search(query, new AbortController().signal)).generation, 'two')
    generation = 'three'
    assert.equal((await expiring.search(query, new AbortController().signal)).generation, 'three')
    broken = true
    const corrupt = new CodeSearchEngine(
      `http://127.0.0.1:${address.port}/static/code-search/manifest.json`,
    )
    await assert.rejects(
      corrupt.search(query, new AbortController().signal),
      /integrity|length|invalid/i,
    )
  } finally {
    await new Promise<void>((resolve, reject) =>
      server.close(error => (error ? reject(error) : resolve())),
    )
  }
})
