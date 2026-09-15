import { build } from 'esbuild'
import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { access, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { promisify } from 'node:util'
import type { ArenaReaderArtifactBase } from '../quartz/util/arena-reader'
import type { ArenaExtractedDocument } from './arena-reader-extraction'
import { ARENA_READER_PROFILE } from './arena-reader-cache'
import {
  arenaReaderFailureSignals,
  arenaReaderRelayHeaders,
  buildArenaHtmlArtifact,
} from './arena-reader-render'

const run = promisify(execFile)

test('relay headers preserve content/CORS without forwarding source credentials or executable workers', () => {
  const headers = arenaReaderRelayHeaders(
    new Headers({
      'Content-Type': 'text/html; charset=utf-8',
      'Content-Encoding': 'gzip',
      'Set-Cookie': 'source=private',
      Authorization: 'secret',
      'Access-Control-Allow-Origin': 'https://example.com',
      'Content-Security-Policy': 'script-src https:',
    }),
    true,
  )
  assert.equal(headers['content-type'], 'text/html; charset=utf-8')
  assert.equal(headers['access-control-allow-origin'], 'https://example.com')
  assert.equal(headers['content-encoding'], undefined)
  assert.equal(headers['set-cookie'], undefined)
  assert.equal(headers.authorization, undefined)
  assert.ok(headers['content-security-policy'].includes("worker-src 'none'"))
  assert.ok(headers['content-security-policy'].includes("frame-src 'none'"))
})

test('challenge and login pages are distinct from short articles about browser challenges', () => {
  assert.equal(
    arenaReaderFailureSignals({
      title: 'Just a moment...',
      text: 'Verify you are human to proceed.',
      articleLength: 100,
      hasArticle: false,
    })?.reason,
    'blocked',
  )
  assert.equal(
    arenaReaderFailureSignals({
      title: 'Sign in',
      text: 'Enter your email to continue reading this article.',
      articleLength: 80,
      hasArticle: false,
    })?.reason,
    'requires-login',
  )
  assert.equal(
    arenaReaderFailureSignals({
      title: 'How browser challenges work',
      text: 'Verify you are human is one common challenge. '.repeat(100),
      articleLength: 4400,
      hasArticle: true,
    }),
    null,
  )
  assert.equal(
    arenaReaderFailureSignals({
      title: 'A short note',
      text: 'A short, complete paragraph can still be a useful article to save and read.',
      articleLength: 72,
      hasArticle: true,
    }),
    null,
  )
})

test('unchanged article content keeps its fingerprint across snapshot IDs and resource paths', async () => {
  const base: ArenaReaderArtifactBase = {
    schemaVersion: 1,
    articleId: `article-v1-${'a'.repeat(64)}`,
    snapshotId: crypto.randomUUID(),
    title: 'Stable content',
    sourceUrl: 'https://example.com/article',
    finalUrl: 'https://example.com/article',
    capturedAt: Date.now(),
    profileVersion: ARENA_READER_PROFILE,
    fingerprint: 'a'.repeat(64),
    resources: [],
  }
  const markup =
    '<h2 id="arena-content-details">Details</h2><a href="#arena-content-details">Details</a><img data-arena-image="0">'
  const extracted: ArenaExtractedDocument = {
    title: base.title,
    readerHtml: markup,
    documentHtml: markup,
    text: 'Stable content',
    documentLength: 200,
    articleLength: 200,
    hasArticle: true,
    imageUrls: ['https://example.com/figure.png'],
  }
  const first = await buildArenaHtmlArtifact(base, extracted, new Set())
  const second = await buildArenaHtmlArtifact(
    { ...base, snapshotId: crypto.randomUUID(), capturedAt: Date.now() + 60_000 },
    extracted,
    new Set(),
  )
  assert.equal(first.fingerprint, second.fingerprint)
  assert.equal(first.resources[0].id, second.resources[0].id)
  assert.ok(first.kind === 'html' && second.kind === 'html')
  assert.notEqual(first.readerHtml, second.readerHtml)
  assert.ok(first.readerHtml?.includes(first.snapshotId))
  assert.ok(second.readerHtml?.includes(second.snapshotId))
  const changed = await buildArenaHtmlArtifact(
    base,
    { ...extracted, documentHtml: `${markup}<p>A new paragraph.</p>` },
    new Set(),
  )
  assert.notEqual(first.fingerprint, changed.fingerprint)
})

test('real Chromium extracts articles and strips active content in an isolated document', async t => {
  let chrome: string | null = null
  for (const candidate of [
    process.env.ARENA_TEST_CHROME,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Helium.app/Contents/MacOS/Helium',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/usr/bin/google-chrome',
  ]) {
    if (!candidate) continue
    try {
      await access(candidate)
      chrome = candidate
      break
    } catch {
      /* Try the next installed browser. */
    }
  }
  if (!chrome)
    return t.skip('Chromium is not installed; set ARENA_TEST_CHROME to run browser fixtures.')
  const directory = await mkdtemp(path.join(tmpdir(), 'arena-extraction-'))
  try {
    const [readability, purify, bundle] = await Promise.all([
      readFile(fileURLToPath(import.meta.resolve('@mozilla/readability/Readability.js')), 'utf8'),
      readFile(fileURLToPath(import.meta.resolve('dompurify/dist/purify.js')), 'utf8'),
      build({
        entryPoints: [fileURLToPath(new URL('./arena-reader-extraction.ts', import.meta.url))],
        bundle: true,
        write: false,
        format: 'iife',
        globalName: 'ArenaExtraction',
        platform: 'browser',
        target: 'es2022',
      }),
    ])
    const paragraph =
      'The saved article discusses the design of an interface for reading links. Its paragraphs should survive extraction with their headings, quotations, source links, and footnotes intact. '
    const html = `<html><head><title>Reading links</title><base href="https://attacker.example/"></head><body>
      <nav><a href="/menu">Menu</a></nav><article><h1>Reading links</h1>
      <p>${paragraph.repeat(5)}</p><h2 id="details">Details</h2><p>${paragraph.repeat(5)}</p>
      <p><a href="#details">Local details</a> <a href="/next">Next article</a>
      <a href="javascript:alert(1)">Bad link</a></p>
      <img src="/image.png" srcset="https://attacker.example/x 2x" onerror="alert(1)" style="position:fixed" alt="A figure">
      <table><tbody><tr><th scope="row">Value</th><td>3</td></tr></tbody></table>
      <pre><code>const x = 3</code></pre><blockquote>A quotation.</blockquote>
      <math display="block"><mfrac><mi>a</mi><mi>b</mi></mfrac></math>
      <script>document.body.dataset.executed='yes'; window.Readability=null;</script>
      <iframe src="https://attacker.example/"></iframe><form action="https://attacker.example"><input autofocus></form>
      </article></body></html>`
    const input = JSON.stringify({
      html,
      finalUrl: 'https://example.com/articles/test',
      idPrefix: 'snapshot-',
    }).replaceAll('<', '\\u003c')
    const fixture = path.join(directory, 'fixture.html')
    await writeFile(
      fixture,
      `<!doctype html><html><head><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'"></head><body><script>${readability}\n${purify}\n${bundle.outputFiles[0].text}\nconst result = ArenaExtraction.extractArenaReaderDocument(${input}); document.body.textContent = btoa(unescape(encodeURIComponent(JSON.stringify(result))));</script></body></html>`,
    )
    const result = await run(
      chrome,
      [
        '--headless',
        '--disable-gpu',
        '--no-sandbox',
        '--disable-extensions',
        '--disable-background-networking',
        `--user-data-dir=${path.join(directory, 'chrome')}`,
        '--dump-dom',
        pathToFileURL(fixture).href,
      ],
      { timeout: 30_000, maxBuffer: 4 * 1024 * 1024 },
    )
    const encoded = result.stdout.match(/<body>([A-Za-z\d+/=]+)<\/body>/)?.[1]
    assert.ok(encoded, result.stderr.slice(-1000))
    const output: ArenaExtractedDocument = JSON.parse(
      Buffer.from(encoded, 'base64').toString('utf8'),
    )
    assert.equal(output.title, 'Reading links')
    assert.ok(output.readerHtml?.includes(paragraph.trim()))
    assert.deepEqual(output.imageUrls, ['https://example.com/image.png'])
    for (const markup of [output.readerHtml, output.documentHtml]) {
      assert.ok(markup)
      assert.match(markup, /id="snapshot-details"/)
      assert.match(markup, /href="#snapshot-details"/)
      assert.match(markup, /href="https:\/\/example.com\/next"/)
      assert.match(markup, /data-arena-image="0"/)
      assert.match(markup, /<table>/)
      assert.match(markup, /<code>const x = 3<\/code>/)
      assert.match(markup, /<math/)
      assert.doesNotMatch(
        markup,
        /<script|<iframe|<form|<input|onerror|javascript:|srcset|style=|attacker\.example/,
      )
    }
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
