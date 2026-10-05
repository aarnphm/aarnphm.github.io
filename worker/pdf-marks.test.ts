import type { PlatformProxy } from 'wrangler'
import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { after, before, test } from 'node:test'
import { getPlatformProxy } from 'wrangler'
import type { PdfManifest, PdfMark, PdfMarkInput } from '../quartz/util/pdf-marks'
import { handlePdfReaderRequest, type PdfReaderEnv } from './pdf-marks'

interface TestEnv {
  ARENA_READER: D1Database
}

const DOC_A = 'a'.repeat(64)
const DOC_B = 'b'.repeat(64)
const DOC_OLD = 'c'.repeat(64)
const SLUG_A = 'thoughts/pdfs/holt.pdf'
const SLUG_B = 'courses/18.100B/pset 1.pdf'

const manifest: PdfManifest = {
  version: 1,
  documents: {
    [SLUG_A]: {
      doc: DOC_A,
      bytes: 1024,
      title: 'Holt, linear algebra',
      citedBy: [
        { from: 'thoughts/rank', title: 'rank', page: 3, excerpt: 'cleanest kernel picture' },
      ],
    },
    [SLUG_B]: { doc: DOC_B, bytes: 2048, title: 'pset 1', citedBy: [] },
  },
}

const shell =
  '<!DOCTYPE html><html><head><title>PDF_READER_TITLE_SLOT</title>' +
  '<meta property="og:title" content="PDF_READER_TITLE_SLOT"/>' +
  '<link rel="canonical" href="https://aarnphm.xyz/read"/>' +
  '<meta property="og:url" content="https://aarnphm.xyz/read"/></head><body data-slug="read">' +
  '<script type="application/json" id="pdf-reader-data">null</script></body></html>'

const assets = {
  async fetch(input: RequestInfo | URL): Promise<Response> {
    const url = new URL(input instanceof Request ? input.url : input.toString())
    if (url.pathname === '/static/pdf-documents.json') return Response.json(manifest)
    if (url.pathname === '/static/pdf-reader') {
      return new Response(shell, { headers: { 'Content-Type': 'text/html; charset=utf-8' } })
    }
    if (url.pathname === '/read') {
      return new Response('<!DOCTYPE html><html><body>Reading note</body></html>')
    }
    return new Response('missing', { status: 404 })
  },
  connect() {
    throw new Error('not supported')
  },
} as unknown as Fetcher

let proxy: PlatformProxy<TestEnv> | undefined
let directory: string | undefined

function env(): PdfReaderEnv {
  assert.ok(proxy)
  return {
    ARENA_READER: proxy.env.ARENA_READER,
    ASSETS: assets,
    ARENA_READER_DEV: 'true',
    SESSION_SECRET: 'test-secret',
  }
}

before(async () => {
  directory = await mkdtemp(path.join(tmpdir(), 'pdf-marks-'))
  const configPath = path.join(directory, 'wrangler.json')
  await writeFile(
    configPath,
    JSON.stringify({
      name: 'pdf-marks-test',
      compatibility_date: '2025-01-21',
      d1_databases: [
        {
          binding: 'ARENA_READER',
          database_name: 'pdf-marks-test',
          database_id: '00000000-0000-0000-0000-000000000003',
        },
      ],
    }),
  )
  proxy = await getPlatformProxy<TestEnv>({
    configPath,
    persist: false,
    remoteBindings: false,
    envFiles: [],
  })
  for (const file of ['0000_arena_reader.sql', '0001_pdf_marks.sql']) {
    const migration = await readFile(
      new URL(`../migrations/arena-reader/${file}`, import.meta.url),
      'utf8',
    )
    for (const statement of migration.split('--> statement-breakpoint')) {
      await proxy.env.ARENA_READER.prepare(statement).run()
    }
  }
})

after(async () => {
  await proxy?.dispose()
  if (directory) await rm(directory, { recursive: true, force: true })
})

const OWNER_ORIGIN = 'http://localhost'
const VISITOR_ORIGIN = 'https://aarnphm.xyz'

function input(overrides: Partial<PdfMarkInput> = {}): PdfMarkInput {
  return {
    src: SLUG_A,
    doc: DOC_A,
    revision: 0,
    kind: 'mark',
    target: {
      type: 'text',
      page: 3,
      quads: [[72, 700, 300, 700, 72, 688, 300, 688]],
      quote: { exact: 'rank nullity', prefix: 'the ', suffix: ' theorem' },
    },
    body: 'the proof happens here',
    visibility: 'public',
    ...overrides,
  }
}

async function call(
  url: string,
  init: RequestInit & { headers?: Record<string, string> } = {},
): Promise<Response> {
  const response = await handlePdfReaderRequest(new Request(url, init), env())
  assert.ok(response, `expected ${url} to be handled`)
  return response
}

function put(
  id: string,
  body: unknown,
  origin = OWNER_ORIGIN,
  headers: Record<string, string> = {},
) {
  return call(`${origin}/api/pdf/marks/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', Origin: origin, ...headers },
    body: JSON.stringify(body),
  })
}

function remove(id: string, revision: number, origin = OWNER_ORIGIN) {
  return call(`${origin}/api/pdf/marks/${id}`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json', Origin: origin },
    body: JSON.stringify({ revision }),
  })
}

async function list(origin: string, src = `/${SLUG_A}`) {
  const response = await call(`${origin}/api/pdf/marks?src=${encodeURIComponent(src)}`)
  return { response, body: (await response.json()) as Record<string, unknown> }
}

async function rowCount(): Promise<number> {
  const row = await env()
    .ARENA_READER!.prepare('SELECT count(*) AS n FROM pdf_marks')
    .first<{ n: number }>()
  return row?.n ?? 0
}

test('owner creates marks and visitors only read public ones (F1, F9)', async () => {
  const publicMark = await put('aaaaaaaaaa', input())
  assert.equal(publicMark.status, 200)
  assert.equal(publicMark.headers.get('Cache-Control'), 'private, no-store')
  const privateMark = await put('bbbbbbbbbb', input({ visibility: 'private', body: 'draft' }))
  assert.equal(privateMark.status, 200)

  const owner = await list(OWNER_ORIGIN)
  assert.equal(owner.response.headers.get('Cache-Control'), 'private, no-store')
  assert.equal(owner.body.canWrite, true)
  assert.deepEqual((owner.body.marks as PdfMark[]).map(mark => mark.id).sort(), [
    'aaaaaaaaaa',
    'bbbbbbbbbb',
  ])

  const visitor = await list(VISITOR_ORIGIN)
  assert.equal(visitor.response.headers.get('Cache-Control'), 'private, no-store')
  assert.equal(visitor.body.canWrite, false)
  assert.deepEqual(
    (visitor.body.marks as PdfMark[]).map(mark => mark.id),
    ['aaaaaaaaaa'],
  )
  assert.deepEqual(visitor.body.stale, [])
})

test('visitors cannot write or delete (F2)', async () => {
  const before = await rowCount()
  const created = await put('cccccccccc', input(), VISITOR_ORIGIN)
  assert.equal(created.status, 401)
  assert.match(String((await created.json()).signInUrl), /^\/comments\/github\/login\?returnTo=/)
  const deleted = await remove('aaaaaaaaaa', 1, VISITOR_ORIGIN)
  assert.equal(deleted.status, 401)
  assert.equal(await rowCount(), before)
})

test('cross-origin writes are refused (F3)', async () => {
  const before = await rowCount()
  const foreign = await call(`${OWNER_ORIGIN}/api/pdf/marks/dddddddddd`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', Origin: 'https://evil.example' },
    body: JSON.stringify(input()),
  })
  assert.equal(foreign.status, 403)
  const crossSite = await put('dddddddddd', input(), OWNER_ORIGIN, {
    'Sec-Fetch-Site': 'cross-site',
  })
  assert.equal(crossSite.status, 403)
  assert.equal(await rowCount(), before)
})

test('stale revisions conflict and never overwrite (F4)', async () => {
  const first = await put('eeeeeeeeee', input({ body: 'one' }))
  const saved = (await first.json()) as PdfMark
  assert.equal(saved.revision, 1)
  const second = await put('eeeeeeeeee', input({ body: 'two', revision: 1 }))
  assert.equal(((await second.json()) as PdfMark).revision, 2)

  const stale = await put('eeeeeeeeee', input({ body: 'stale', revision: 1 }))
  assert.equal(stale.status, 409)
  const conflict = (await stale.json()) as { current: PdfMark }
  assert.equal(conflict.current.body, 'two')
  assert.equal(conflict.current.revision, 2)

  const duplicate = await put('eeeeeeeeee', input({ body: 'fresh insert' }))
  assert.equal(duplicate.status, 409)
})

test('unknown sources and mismatched documents are refused (F5, F6)', async () => {
  const before = await rowCount()
  const unknown = await put('ffffffffff', input({ src: 'elsewhere/arxiv.pdf' }))
  assert.equal(unknown.status, 404)
  const mismatch = await put('ffffffffff', input({ doc: DOC_B }))
  assert.equal(mismatch.status, 409)
  assert.equal(((await mismatch.json()) as { doc: string }).doc, DOC_A)
  assert.equal(await rowCount(), before)
})

test('malformed payloads are refused (F7)', async () => {
  const before = await rowCount()
  const quad = [72, 700, 300, 700, 72, 688, 300, 688]
  const cases = [
    JSON.stringify(input()).replace('"quads":[[72', '"quads":[[1e999'),
    JSON.stringify(
      input({
        target: {
          type: 'text',
          page: 1,
          quads: Array.from({ length: 65 }, () => quad) as never,
          quote: { exact: 'x', prefix: '', suffix: '' },
        },
      }),
    ),
    JSON.stringify(input({ body: 'x'.repeat(17 * 1024) })),
    JSON.stringify({ ...input(), owner: 'mallory' }),
  ]
  for (const body of cases) {
    const response = await call(`${OWNER_ORIGIN}/api/pdf/marks/gggggggggg`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json', Origin: OWNER_ORIGIN },
      body,
    })
    assert.equal(response.status, 422, body.slice(0, 80))
  }
  const badId = await put('NOT-AN-ID', input())
  assert.equal(badId.status, 404)
  assert.equal(await rowCount(), before)
})

test('deleted marks stay deleted under stale writes (F8)', async () => {
  const created = (await (await put('hhhhhhhhhh', input())).json()) as PdfMark
  const deleted = await remove('hhhhhhhhhh', created.revision)
  assert.equal(deleted.status, 200)
  const stale = await put('hhhhhhhhhh', input({ revision: created.revision, body: 'zombie' }))
  assert.equal(stale.status, 409)
  const owner = await list(OWNER_ORIGIN)
  assert.ok(!(owner.body.marks as PdfMark[]).some(mark => mark.id === 'hhhhhhhhhh'))
  const again = await remove('hhhhhhhhhh', created.revision)
  assert.equal(again.status, 200)
  const missing = await remove('jjjjjjjjjj', 1)
  assert.equal(missing.status, 404)
})

test('owner sees marks from a previous version of the file as stale', async () => {
  const now = Date.now()
  await env()
    .ARENA_READER!.prepare(
      `INSERT INTO pdf_marks (mark_id, owner, doc, src, kind, page, target, body, visibility, revision, created_at, updated_at)
       VALUES ('kkkkkkkkkk', 'aarnphm', ?, ?, 'question', 2, ?, 'old version', 'public', 1, ?, ?)`,
    )
    .bind(DOC_OLD, SLUG_A, JSON.stringify(input().target), now, now)
    .run()
  const owner = await list(OWNER_ORIGIN)
  assert.deepEqual(
    (owner.body.stale as PdfMark[]).map(mark => mark.id),
    ['kkkkkkkkkk'],
  )
  assert.ok(!(owner.body.marks as PdfMark[]).some(mark => mark.id === 'kkkkkkkkkk'))
  const visitor = await list(VISITOR_ORIGIN)
  assert.ok(!(visitor.body.marks as PdfMark[]).some(mark => mark.id === 'kkkkkkkkkk'))

  const reanchored = await put(
    'kkkkkkkkkk',
    input({ revision: 1, kind: 'question', body: 'old version' }),
  )
  assert.equal(reanchored.status, 200)
  assert.equal(((await reanchored.json()) as PdfMark).doc, DOC_A)
})

test('reader markdown twin carries public marks only (F10, F12)', async () => {
  const markdown = await call(`${VISITOR_ORIGIN}/read/thoughts/pdfs/holt`, {
    headers: { Accept: 'text/markdown' },
  })
  assert.equal(markdown.status, 200)
  assert.match(markdown.headers.get('Content-Type') ?? '', /^text\/markdown/)
  const text = await markdown.text()
  assert.match(text, /Holt, linear algebra/)
  assert.match(text, /the proof happens here/)
  assert.doesNotMatch(text, /draft/)
  assert.match(text, /\[rank\]\(\/thoughts\/rank\)/)

  const agent = await call(`${VISITOR_ORIGIN}/read/thoughts/pdfs/holt`, {
    headers: { Accept: '*/*', 'User-Agent': 'Claude-User/1.0' },
  })
  assert.equal(agent.status, 200)
  assert.match(agent.headers.get('Content-Type') ?? '', /^text\/markdown/)
})

test('reader shell uses its dedicated asset and gets the document record injected', async () => {
  const response = await call(`${VISITOR_ORIGIN}/read/courses/18.100B/pset%201`, {
    headers: { Accept: 'text/html' },
  })
  assert.equal(response.status, 200)
  const html = await response.text()
  assert.doesNotMatch(html, /PDF_READER_TITLE_SLOT/)
  assert.match(html, /<title>pset 1<\/title>/)
  const json = /<script type="application\/json" id="pdf-reader-data">(.*?)<\/script>/.exec(html)
  assert.ok(json)
  const data = JSON.parse(json[1])
  assert.equal(data.slug, SLUG_B)
  assert.equal(data.document.doc, DOC_B)
  assert.match(html, /<body data-slug="read\/courses\/18\.100B\/pset 1">/)
  assert.match(html, /href="https:\/\/aarnphm\.xyz\/read\/courses\/18\.100B\/pset%201"/)
  assert.match(html, /og:url" content="https:\/\/aarnphm\.xyz\/read\/courses\/18\.100B\/pset%201"/)

  const index = await call(`${VISITOR_ORIGIN}/read`, { headers: { Accept: 'text/html' } })
  assert.equal(index.status, 200)
  const indexHtml = await index.text()
  assert.match(indexHtml, /"mode":"index"/)
  const indexJson = /<script type="application\/json" id="pdf-reader-data">(.*?)<\/script>/.exec(
    indexHtml,
  )
  assert.ok(indexJson)
  const indexData = JSON.parse(indexJson[1])
  assert.deepEqual(indexData.documents, [
    {
      slug: SLUG_A,
      readPath: '/read/thoughts/pdfs/holt',
      title: 'Holt, linear algebra',
      citations: 1,
    },
    { slug: SLUG_B, readPath: '/read/courses/18.100B/pset 1', title: 'pset 1', citations: 0 },
  ])
  assert.match(indexHtml, /<body data-slug="read">/)
  assert.match(indexHtml, /href="https:\/\/aarnphm\.xyz\/read"/)

  const markdownIndex = await call(`${VISITOR_ORIGIN}/read`, {
    headers: { Accept: 'text/markdown' },
  })
  assert.match(
    await markdownIndex.text(),
    /\[pset 1\]\(\/read\/courses\/18\.100B\/pset%201\) \(0\)/,
  )
})

test('unknown reader paths fall through to the 404 handling (F11)', async () => {
  const response = await handlePdfReaderRequest(
    new Request(`${VISITOR_ORIGIN}/read/not/a/pdf`, { headers: { Accept: 'text/html' } }),
    env(),
  )
  assert.equal(response, null)
})

test('hosted PDF navigations redirect to the reader, fetches do not', async () => {
  const navigate = { 'Sec-Fetch-Mode': 'navigate', 'Sec-Fetch-Dest': 'document' }
  const redirect = await call(`${VISITOR_ORIGIN}/thoughts/pdfs/holt.pdf`, { headers: navigate })
  assert.equal(redirect.status, 301)
  assert.equal(redirect.headers.get('Location'), '/read/thoughts/pdfs/holt')
  assert.equal(redirect.headers.get('Cache-Control'), 'no-store')

  const spaced = await call(`${VISITOR_ORIGIN}/courses/18.100B/pset%201.pdf`, { headers: navigate })
  assert.equal(spaced.headers.get('Location'), '/read/courses/18.100B/pset%201')

  const pass = async (url: string, headers: Record<string, string>) =>
    handlePdfReaderRequest(new Request(url, { headers }), env())
  assert.equal(
    await pass(`${VISITOR_ORIGIN}/thoughts/pdfs/holt.pdf`, {
      'Sec-Fetch-Mode': 'cors',
      'Sec-Fetch-Dest': 'empty',
    }),
    null,
  )
  assert.equal(await pass(`${VISITOR_ORIGIN}/thoughts/pdfs/holt.pdf?raw=1`, navigate), null)
  assert.equal(await pass(`${VISITOR_ORIGIN}/thoughts/pdfs/holt.pdf`, {}), null)
  assert.equal(await pass(`${VISITOR_ORIGIN}/elsewhere/paper.pdf`, navigate), null)
  assert.equal(
    await pass(`${VISITOR_ORIGIN}/thoughts/pdfs/holt.pdf`, {
      'Sec-Fetch-Mode': 'navigate',
      'Sec-Fetch-Dest': 'iframe',
    }),
    null,
  )
})
