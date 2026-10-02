import type { TestHarness } from 'wrangler'
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { after, before, test } from 'node:test'
import { createTestHarness } from 'wrangler'
import { deriveTrainingDocument } from '../quartz/components/triathlon/training/tree'

const origin = 'https://t.aarnphm.xyz'
let server: TestHarness
let evidence: string

// Exercise the full Worker and static assets: host redirects, representation negotiation,
// calendar downloads, shared resources, and HTML links must agree at the request boundary.
before(async () => {
  evidence = await mkdtemp(path.join(tmpdir(), 'garden-triathlon-host-'))
  const assetDirectory = path.join(evidence, 'assets')
  const slugs = [
    'triathlon',
    'triathlon/tools',
    'triathlon/analytics',
    'triathlon/feed',
    'triathlon/training',
    'triathlon/on/2026/08/14',
  ]
  for (const slug of slugs) {
    const relative = (target: string): string =>
      path.posix.relative(path.posix.dirname(slug), target)
    await mkdir(path.dirname(path.join(assetDirectory, slug)), { recursive: true })
    await writeFile(
      path.join(assetDirectory, `${slug}.html`),
      `<!doctype html><html><head>
<link rel="canonical" href="https://aarnphm.xyz/${slug}">
<link rel="stylesheet" href="${relative('static/site.css')}">
</head><body data-slug="${slug}" class="triathlon">
<a class="internal" href="${relative('triathlon/tools')}">tools</a>
<a class="internal" href="${relative('triathlon/analytics')}">analytics</a>
<a class="internal" href="${relative('thoughts')}">thoughts</a>
<a href="https://aarnphm.xyz/">garden</a>
<script src="${relative('static/site.js')}"></script>
</body></html>`,
    )
    await writeFile(path.join(assetDirectory, `${slug}.md`), `# ${slug}\n\nTraining fixture.\n`)
  }
  // Plans originate in /triathlon and are rendered inside its deeper training page.
  // Relative note links must keep that source base through SSR and later plan selection.
  const trainingDocument = deriveTrainingDocument({
    id: 'links',
    meta: 'Link routing',
    distance: 'olympic',
    date: '2026-08-14',
    target: 'finish',
    author: '',
    html: `<h2>References</h2>
<a href="./thoughts/nutrition?view=reader#fuel">relative note</a>
<a href="/thoughts/pdfs/race.pdf#page=2">race PDF</a>
<a href="https://aarnphm.xyz/thoughts/recovery#sleep">apex note</a>
<a href="./triathlon/tools">tools</a>
<a href="https://stream.aarnphm.xyz/on/2026/08/14">stream</a>
<a href="https://support.tridot.com/hc/en-us/sections/201539593-Swim">swim drills</a>
<a href="#fn-1">footnote</a>
<img src="./thoughts/images/fuel.png" alt="fuel">`,
  })
  await writeFile(
    path.join(assetDirectory, 'triathlon/training.html'),
    `<!doctype html><html><head><link rel="canonical" href="https://aarnphm.xyz/triathlon/training"></head><body data-slug="triathlon/training">${trainingDocument.html}</body></html>`,
  )
  await mkdir(path.join(assetDirectory, 'static/triathlon'), { recursive: true })
  await writeFile(path.join(assetDirectory, 'static/site.css'), '.triathlon { display: block }')
  await writeFile(
    path.join(assetDirectory, 'static/site.js'),
    'document.body.dataset.hydrated = "true"',
  )
  await writeFile(
    path.join(assetDirectory, 'static/strava-activity-index.json'),
    JSON.stringify({
      kind: 'strava-activity-index-v1',
      activities: { '19745591953': '2026-08-14' },
    }),
  )
  await writeFile(
    path.join(assetDirectory, 'static/triathlon/data.jsonl'),
    '{"kind":"activity","date":"2026-08-14"}\n',
  )
  await writeFile(
    path.join(assetDirectory, 'triathlon/calendar.ics'),
    'BEGIN:VCALENDAR\r\nVERSION:2.0\r\nEND:VCALENDAR\r\n',
  )
  await writeFile(
    path.join(evidence, 'README.md'),
    '# Triathlon host E2E\n\nCommand: `pnpm test worker/triathlon-host.test.ts`\n\nInputs are saved in `assets/`. The complete Worker runs in Wrangler against these static files. Each request saves its response headers and body here.\n',
  )
  server = createTestHarness({
    root: process.cwd(),
    workers: [
      {
        config: {
          name: 'triathlon-host-test',
          main: path.resolve('worker/index.ts'),
          compatibility_date: '2025-01-21',
          compatibility_flags: ['nodejs_compat', 'global_fetch_strictly_public'],
          rules: [
            {
              type: 'Text',
              globs: ['**/*.txt', 'defuddle/full', '**/purify.js'],
              fallthrough: true,
            },
          ],
          vars: { PUBLIC_BASE_URL: 'https://aarnphm.xyz' },
          assets: { directory: assetDirectory, binding: 'ASSETS', run_worker_first: true },
          kv_namespaces: [{ binding: 'OAUTH_KV', id: 'test-oauth' }],
        },
      },
    ],
  })
  await server.listen()
})

after(async () => {
  await server?.close()
  console.log(`Triathlon host E2E responses: ${evidence}`)
})

async function request(
  pathname: string,
  init?: RequestInit,
  requestOrigin = origin,
): Promise<Response> {
  const response = await server.fetch(`${requestOrigin}${pathname}`, {
    redirect: 'manual',
    ...init,
  })
  const requestHeaders = Object.fromEntries(new Headers(init?.headers))
  const representation = requestHeaders.accept?.replaceAll(/[^a-z0-9]+/gi, '-') ?? 'html'
  const filename = `${new URL(requestOrigin).hostname}-${pathname.replaceAll('/', '_') || 'root'}-${init?.method ?? 'GET'}-${representation}`
  await mkdir(evidence, { recursive: true })
  await writeFile(
    path.join(evidence, `${filename}.headers.json`),
    JSON.stringify(
      {
        url: `${requestOrigin}${pathname}`,
        method: init?.method ?? 'GET',
        requestHeaders,
        status: response.status,
        headers: Object.fromEntries(response.headers),
      },
      null,
      2,
    ),
  )
  await writeFile(path.join(evidence, `${filename}.body`), await response.clone().text())
  return response
}

test('redirects the apex overview to the microsite and preserves legacy subpage URLs', async () => {
  for (const pathname of ['/triathlon', '/triathlon/', '/triathlon.html']) {
    for (const init of [{}, { method: 'HEAD' }, { headers: { Accept: 'text/markdown' } }]) {
      const response = await request(`${pathname}?unit=imperial`, init, 'https://aarnphm.xyz')
      assert.equal(response.status, 308, pathname)
      assert.equal(response.headers.get('Location'), `${origin}/?unit=imperial`)
    }
  }
  for (const requestOrigin of ['https://aarnphm.xyz', 'http://localhost:8080']) {
    const response = await request('/triathlon/training', undefined, requestOrigin)
    assert.equal(response.status, 200, requestOrigin)
  }
  assert.equal((await request('/triathlon', undefined, 'http://localhost:8080')).status, 200)
})

test('keeps training references on their owning domains and resolves their original source base', async () => {
  for (const [requestOrigin, pathname] of [
    [origin, '/training'],
    ['https://aarnphm.xyz', '/triathlon/training'],
  ]) {
    const response = await request(pathname, undefined, requestOrigin)
    assert.equal(response.status, 200)
    const html = await response.text()
    for (const href of [
      'https://aarnphm.xyz/thoughts/nutrition?view=reader#fuel',
      'https://aarnphm.xyz/thoughts/pdfs/race.pdf#page=2',
      'https://aarnphm.xyz/thoughts/recovery#sleep',
      'https://stream.aarnphm.xyz/on/2026/08/14',
      'https://support.tridot.com/hc/en-us/sections/201539593-Swim',
      '#fn-1',
      `${requestOrigin === origin ? origin : requestOrigin}/${requestOrigin === origin ? 'tools' : 'triathlon/tools'}`,
    ])
      assert.ok(html.includes(`href="${href}"`), `${requestOrigin}: ${href}`)
    assert.ok(html.includes('src="https://aarnphm.xyz/thoughts/images/fuel.png"'))
  }
})

test('serves triathlon pages at the microsite root with usable navigation and metadata', async () => {
  for (const [pathname, slug] of [
    ['/', 'triathlon'],
    ['/tools', 'triathlon/tools'],
    ['/analytics', 'triathlon/analytics'],
    ['/feed', 'triathlon/feed'],
    ['/on/2026/08/14', 'triathlon/on/2026/08/14'],
  ]) {
    const response = await request(pathname)
    assert.equal(response.status, 200, pathname)
    assert.match(response.headers.get('Content-Type') ?? '', /^text\/html/)
    const html = await response.text()
    assert.ok(html.includes(`data-slug="${slug}"`), pathname)
    assert.ok(html.includes(`rel="canonical" href="${origin}${pathname}"`), pathname)
    assert.ok(html.includes('href="https://t.aarnphm.xyz/tools"'), pathname)
    assert.ok(html.includes('href="https://t.aarnphm.xyz/analytics"'), pathname)
    assert.ok(html.includes('href="https://aarnphm.xyz/thoughts"'), pathname)
    assert.ok(html.includes('href="https://aarnphm.xyz/"'), pathname)
  }
})

test('canonicalizes prefixed and date shortcut URLs without leaving the microsite', async () => {
  for (const [pathname, target] of [
    ['/triathlon?unit=imperial', '/?unit=imperial'],
    ['/triathlon/tools', '/tools'],
    ['/tools.html', '/tools'],
    ['/tools/', '/tools'],
    ['/2026/08/14?view=run', '/on/2026/08/14?view=run'],
    ['/activities/19745591953', '/on/2026/08/14'],
  ]) {
    const response = await request(pathname)
    assert.equal(response.status, 308, pathname)
    assert.equal(response.headers.get('Location'), `${origin}${target}`)
  }
})

test('negotiates Markdown and preserves the data endpoint representation', async () => {
  for (const pathname of ['/', '/analytics', '/on/2026/08/14']) {
    const response = await request(pathname, { headers: { Accept: 'text/markdown' } })
    assert.equal(response.status, 200, pathname)
    assert.match(response.headers.get('Content-Type') ?? '', /^text\/markdown/)
  }
  assert.equal((await request('/', { method: 'HEAD' })).status, 200)
  const data = await request('/data', { headers: { Accept: 'application/x-ndjson' } })
  assert.equal(data.status, 200)
  assert.match(data.headers.get('Content-Type') ?? '', /^application\/x-ndjson/)
  const htmlData = await request('/data', { headers: { Accept: 'text/html' } })
  assert.equal(htmlData.status, 200)
  assert.match(htmlData.headers.get('Content-Type') ?? '', /^text\/html/)
})

test('serves calendar exports and shared assets through their existing handlers', async () => {
  const calendar = await request('/calendar.ics')
  assert.equal(calendar.status, 200)
  assert.match(await calendar.text(), /BEGIN:VCALENDAR/)
  const asset = await request('/static/strava-activity-index.json')
  assert.equal(asset.status, 200)
  assert.deepEqual(
    await asset.json(),
    JSON.parse(
      await readFile(path.join(evidence, 'assets/static/strava-activity-index.json'), 'utf8'),
    ),
  )
})

test('redirects Garden documents to the apex and keeps missing activity failures explicit', async () => {
  const response = await request('/thoughts?view=reader')
  assert.equal(response.status, 308)
  assert.equal(response.headers.get('Location'), 'https://aarnphm.xyz/thoughts?view=reader')
  assert.equal((await request('/activities/999999999999999')).status, 404)
  assert.equal((await request('/on/1900/01/01')).status, 404)
  assert.equal((await request('/tools', { method: 'POST' })).status, 405)
  const api = await request('/api/unknown')
  assert.equal(api.status, 404)
  assert.match(api.headers.get('Content-Type') ?? '', /^application\/problem\+json/)
})
