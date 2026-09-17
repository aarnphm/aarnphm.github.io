import type { TestHarness } from 'wrangler'
import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { after, before, test } from 'node:test'
import { fileURLToPath } from 'node:url'
import { createTestHarness } from 'wrangler'
import type { ArenaExtractedDocument } from './arena-reader-extraction'
import { curiusPreviewImagePath, parseCuriusPreview } from '../quartz/util/curius-preview'
import { buildCuriusPreview, buildGithubCuriusPreview } from './curius-preview'

const origin = 'https://aarnphm.xyz'
const link = { id: 236997, link: 'https://example.com/article', title: 'Saved title' }
const extracted: ArenaExtractedDocument = {
  title: 'Extracted article',
  readerHtml:
    '<h2 id="curius-preview-236997-details">Details</h2><p>A readable article with enough text to identify the body.</p><img data-arena-image="0" alt="A figure"><img data-arena-image="1" alt="Blocked figure">',
  text: 'A readable article with enough text to identify the body.',
  articleLength: 58,
  hasArticle: true,
  imageUrls: ['https://example.com/figure.png', 'http://127.0.0.1/private.png'],
}

const repository = {
  id: 237000,
  link: 'https://github.com/JasonGross/guarantees-based-mechanistic-interpretability',
  title: 'Saved repository',
}
const repositoryHtml = `<!doctype html><html><head>
  <meta name="octolytics-dimension-repository_nwo" content="JasonGross/guarantees-based-mechanistic-interpretability">
  <meta property="og:image" content="https://opengraph.githubassets.com/revision/JasonGross/guarantees-based-mechanistic-interpretability">
  <meta property="og:image:width" content="1200"><meta property="og:image:height" content="600">
  <meta property="og:description" content="Contribute to JasonGross/guarantees-based-mechanistic-interpretability development by creating an account on GitHub.">
</head><body>
  <nav>Search or jump to... Search code, repositories, users, issues, pull requests... Sign in</nav>
  <article class="markdown-body"><h1>Guarantees-Based Mechanistic Interpretability</h1>
    <p><img src="https://example.com/badge.svg"></p>
    <p>This is the codebase for the <strong>Guarantees-Based Mechanistic Interpretability MARS stream</strong>. Types like &lt;T&gt; &amp; &lt;script&gt; remain text.</p>
    <p>Later documentation does not crowd the repository card.</p>
  </article>
</body></html>`

test('repository previews use the real social card and README introduction without navigation chrome', () => {
  const result = buildGithubCuriusPreview(repository, repository.link, repositoryHtml)
  assert.ok(result.preview.status === 'ready')
  assert.equal(result.preview.title, 'JasonGross/guarantees-based-mechanistic-interpretability')
  assert.equal(result.preview.sourceUrl, repository.link)
  assert.deepEqual(result.imageUrls, [
    'https://opengraph.githubassets.com/revision/JasonGross/guarantees-based-mechanistic-interpretability',
  ])
  assert.match(result.preview.readerHtml, /curius-preview-repository-card/)
  assert.match(result.preview.readerHtml, /width="1200" height="600"/)
  assert.match(
    result.preview.readerHtml,
    /src="\/api\/curius\?query=preview-image&amp;id=237000&amp;image=0&amp;v=\d+"/,
  )
  assert.match(
    result.preview.readerHtml,
    /codebase for the Guarantees-Based Mechanistic Interpretability MARS stream/,
  )
  assert.match(result.preview.readerHtml, /&lt;T&gt; &amp; &lt;script&gt;/)
  assert.doesNotMatch(
    result.preview.readerHtml,
    /Search or jump|Search code|Sign in|Contribute to|Later documentation|badge.svg|<script>/,
  )
  assert.deepEqual(parseCuriusPreview(result.preview), result.preview)
})

test('repository previews validate repository identity and use useful text when a social card is absent', () => {
  for (const html of [
    repositoryHtml.replace('octolytics-dimension-repository_nwo', 'unrelated-metadata'),
    repositoryHtml.replace(
      'content="JasonGross/guarantees-based-mechanistic-interpretability"',
      'content="another/repository"',
    ),
    '<html><body>Sign in to GitHub</body></html>',
  ])
    assert.throws(
      () => buildGithubCuriusPreview(repository, repository.link, html),
      /repository preview is unavailable/,
    )
  assert.throws(
    () => buildGithubCuriusPreview(repository, 'https://github.com/login', repositoryHtml),
    /repository preview is unavailable/,
  )
  const blockedImage = buildGithubCuriusPreview(
    repository,
    repository.link,
    repositoryHtml.replace(
      'https://opengraph.githubassets.com/revision/JasonGross/guarantees-based-mechanistic-interpretability',
      'https://127.0.0.1/private.png',
    ),
  )
  assert.deepEqual(blockedImage.imageUrls, [])
  assert.ok(blockedImage.preview.status === 'ready')
  assert.doesNotMatch(blockedImage.preview.readerHtml, /<img|127\.0\.0\.1/)
  assert.match(blockedImage.preview.readerHtml, /MARS stream/)
  const descriptionOnly = repositoryHtml
    .replace(/<article[\s\S]*?<\/article>/, '')
    .replace(
      'Contribute to JasonGross/guarantees-based-mechanistic-interpretability development by creating an account on GitHub.',
      'Formal guarantees &amp; mechanistic interpretability.',
    )
  const described = buildGithubCuriusPreview(repository, repository.link, descriptionOnly)
  assert.ok(described.preview.status === 'ready')
  assert.match(described.preview.readerHtml, /Formal guarantees &amp; mechanistic interpretability/)
})

test('extracted previews expose only generated image routes and preserve source identity', () => {
  const result = buildCuriusPreview(link, 'https://example.com/redirected', extracted)
  assert.equal(result.preview.status, 'ready')
  if (result.preview.status !== 'ready') return
  assert.equal(result.preview.linkId, link.id)
  assert.equal(result.preview.sourceUrl, link.link)
  assert.equal(result.preview.finalUrl, 'https://example.com/redirected')
  assert.equal(result.preview.title, extracted.title)
  assert.deepEqual(result.imageUrls, ['https://example.com/figure.png', ''])
  assert.match(
    result.preview.readerHtml,
    /src="\/api\/curius\?query=preview-image&amp;id=236997&amp;image=0&amp;v=\d+"/,
  )
  assert.doesNotMatch(
    result.preview.readerHtml,
    /127\.0\.0\.1|data-arena-image|https:\/\/example.com\/figure/,
  )
  assert.match(result.preview.readerHtml, /<img\s+alt="Blocked figure">/)
  assert.deepEqual(parseCuriusPreview(result.preview), result.preview)
  assert.throws(
    () =>
      buildCuriusPreview(link, link.link, {
        ...extracted,
        title: 'Just a moment...',
        text: 'Verify you are human.',
        articleLength: 21,
      }),
    /browser challenge/,
  )
  assert.throws(
    () =>
      buildCuriusPreview(link, link.link, {
        ...extracted,
        readerHtml: null,
        text: '',
        articleLength: 0,
      }),
    /readable article/,
  )
})

let server: TestHarness | undefined
let directory: string | undefined
let imagePath: string

function harness(): TestHarness {
  assert.ok(server)
  return server
}

before(async () => {
  directory = await mkdtemp(path.join(tmpdir(), 'curius-preview-test-'))
  const main = path.join(directory, 'entry.ts')
  const handlerPath = fileURLToPath(new URL('./curius.ts', import.meta.url))
  const cataloguePath = fileURLToPath(new URL('./arena-reader-catalogue.ts', import.meta.url))
  const preview = buildCuriusPreview(link, link.link, extracted)
  assert.equal(preview.preview.status, 'ready')
  if (preview.preview.status !== 'ready') throw new Error('Missing fixture preview')
  imagePath = curiusPreviewImagePath(link.id, 0, preview.preview.fetchedAt)
  await writeFile(
    main,
    `
import handleCurius from ${JSON.stringify(handlerPath)}
import { CURIUS_FEED_CACHE_KEY } from ${JSON.stringify(cataloguePath)}
export default {
  async fetch(request, env) {
    const url = new URL(request.url)
    if (url.pathname === '/__test/seed') {
      await env.ARENA_CONTENT.put(CURIUS_FEED_CACHE_KEY, JSON.stringify({
        schemaVersion: 1, fetchedAt: Date.now(), retryAt: 0,
        links: ${JSON.stringify([link, { id: 236998, link: 'http://127.0.0.1/private', title: 'Private address' }, { id: 236999, link: 'https://example.com/uncached', title: 'Uncached' }])},
      }))
      await env.ARENA_CONTENT.put('arena-reader/private-note-fixture', 'private notes stay private')
      const cache = await caches.open('curius-preview-defuddle-0.19.3-v2')
      await cache.put(new Request(new URL(${JSON.stringify(`/api/curius-preview-cache/v2/${link.id}`)}, request.url)), Response.json(${JSON.stringify(preview)}, { headers: { 'Cache-Control': 'public, max-age=3600' } }))
      await cache.put(new Request(new URL(${JSON.stringify(imagePath)}, request.url)), new Response(new Uint8Array([137, 80, 78, 71]), { headers: { 'Content-Type': 'image/png', 'Cache-Control': 'public, max-age=3600', 'X-Content-Type-Options': 'nosniff', 'Cross-Origin-Resource-Policy': 'same-origin' } }))
      return Response.json({ seeded: true })
    }
    if (url.pathname === '/__test/state') {
      const list = await env.ARENA_CONTENT.list()
      const notes = await env.ARENA_CONTENT.get('arena-reader/private-note-fixture')
      return Response.json({ keys: list.objects.map(item => item.key), notes: await notes.text() })
    }
    return handleCurius(request, env)
  }
}`,
  )
  server = createTestHarness({
    root: directory,
    workers: [
      {
        config: {
          name: 'curius-preview-test',
          main,
          compatibility_date: '2025-01-21',
          compatibility_flags: ['nodejs_compat', 'global_fetch_strictly_public'],
          rules: [{ type: 'Text', globs: ['defuddle/full', '**/purify.js'], fallthrough: true }],
          r2_buckets: [{ binding: 'ARENA_CONTENT', bucket_name: 'curius-preview-test' }],
        },
      },
    ],
  })
  await server.listen()
  assert.equal((await server.fetch(`${origin}/__test/seed`)).status, 200)
})

after(async () => {
  await server?.close()
  if (directory) await rm(directory, { recursive: true, force: true })
})

test('public preview route requires saved link IDs and preserves clear failure responses', async () => {
  for (const id of ['', '0', '-1', '1.5', '9007199254740992', 'https://example.com']) {
    const response = await harness().fetch(
      `${origin}/api/curius?query=preview&id=${encodeURIComponent(id)}`,
    )
    assert.equal(response.status, 400, id)
    assert.equal(response.headers.get('Cache-Control'), 'no-store')
    const body = parseCuriusPreview(await response.json())
    assert.ok(body?.status === 'unavailable' && body.reason === 'invalid-id')
  }
  for (const [id, status, reason] of [
    [999999, 404, 'not-saved'],
    [236998, 400, 'blocked-source'],
    [236999, 503, 'unconfigured'],
  ]) {
    const response = await harness().fetch(`${origin}/api/curius?query=preview&id=${id}`)
    assert.equal(response.status, status)
    const body = parseCuriusPreview(await response.json())
    assert.ok(body?.status === 'unavailable' && body.reason === reason)
  }
  assert.equal(
    (await harness().fetch(`${origin}/api/curius?query=preview&id=${link.id}`, { method: 'POST' }))
      .status,
    405,
  )
})

test('anonymous visitors can read cached article content without Arena notes or sessions', async () => {
  const before = await (await harness().fetch(`${origin}/__test/state`)).json()
  const response = await harness().fetch(
    `${origin}/api/curius?query=preview&id=${link.id}&url=http://127.0.0.1/private`,
  )
  assert.equal(response.status, 200)
  assert.equal(response.headers.get('Cache-Control'), 'public, max-age=300')
  const body = parseCuriusPreview(await response.json())
  assert.ok(body?.status === 'ready')
  assert.equal(body.cached, true)
  assert.equal(body.sourceUrl, link.link)
  assert.match(body.readerHtml, /A readable article/)
  assert.doesNotMatch(
    JSON.stringify(body),
    /private notes|private-note-fixture|session|127\.0\.0\.1/,
  )
  assert.deepEqual(await (await harness().fetch(`${origin}/__test/state`)).json(), before)
})

test('public images are tied to the current saved article extraction', async () => {
  const response = await harness().fetch(`${origin}${imagePath}`)
  assert.equal(response.status, 200)
  assert.equal(response.headers.get('Content-Type'), 'image/png')
  assert.equal(response.headers.get('X-Content-Type-Options'), 'nosniff')
  assert.equal(response.headers.get('Cross-Origin-Resource-Policy'), 'same-origin')
  assert.deepEqual(new Uint8Array(await response.arrayBuffer()), new Uint8Array([137, 80, 78, 71]))
  for (const route of [
    imagePath.replace(/v=\d+/, 'v=1'),
    imagePath.replace('image=0', 'image=1'),
    imagePath.replace('image=0', 'image=300'),
    imagePath.replace(`id=${link.id}`, 'id=999999'),
    `${imagePath}&url=http://127.0.0.1/private`,
  ])
    assert.equal((await harness().fetch(`${origin}${route}`)).status, 404, route)
})
