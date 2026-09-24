import { build } from 'esbuild'
import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { access, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { promisify } from 'node:util'
import { compile } from 'sass-embedded'
import type { ArenaReaderArtifactBase } from '../quartz/util/arena-reader'
import type { ArenaExtractedDocument } from './arena-reader-extraction'
import { ARENA_READER_PROFILE } from './arena-reader-cache'
import {
  arenaReaderFailureSignals,
  arenaReaderRelayHeaders,
  buildArenaHtmlArtifact,
  isArenaSubstackArticle,
  readArenaReaderResource,
  decodeArenaSourceCode,
} from './arena-reader-render'

const run = promisify(execFile)

test('raw source decoding preserves whitespace and rejects binary files', () => {
  const source = '# café\r\n\tprint("<script> & text")\r\n\r\n'
  assert.equal(decodeArenaSourceCode(new TextEncoder().encode(source)), source)
  assert.equal(decodeArenaSourceCode(new Uint8Array()), '')
  assert.throws(() => decodeArenaSourceCode(new Uint8Array([0, 1, 2])), /binary file/)
  assert.throws(() => decodeArenaSourceCode(new Uint8Array([0xff, 0xfe])), /not UTF-8/)
})

test('Substack article captures use their supplied HTML on publication and custom domains', () => {
  const substackHeaders = new Headers({ 'X-Served-By': 'Substack' })
  assert.ok(isArenaSubstackArticle('https://example.substack.com/p/article', new Headers()))
  assert.ok(isArenaSubstackArticle('https://www.astralcodexten.com/p/the-pledge', substackHeaders))
  assert.ok(
    isArenaSubstackArticle(
      'https://publisher.example/p/article/?utm_source=reader',
      substackHeaders,
    ),
  )
  for (const url of [
    'https://example.substack.com/',
    'https://example.substack.com/p/article/comments',
    'https://substack.com/@example/note/c-123',
  ]) {
    assert.equal(isArenaSubstackArticle(url, substackHeaders), false, url)
  }
  for (const url of [
    'https://publisher.example/p/article',
    'https://substack.com.example/p/article',
    'https://notsubstack.com/p/article',
  ]) {
    assert.equal(isArenaSubstackArticle(url, new Headers()), false, url)
  }
})

test('capture allows MathJax-sized scripts while bounding documents and streamed resources', async t => {
  const documentLimit = 2 * 1024 * 1024
  const scriptLimit = 4 * 1024 * 1024
  const bundle = new Uint8Array(2_108_580)
  await t.test('combined math bundles fit the script budget', async () => {
    const body = await readArenaReaderResource(
      new Response(bundle, { headers: { 'Content-Length': String(bundle.byteLength) } }),
      'script',
    )
    assert.equal(body.byteLength, bundle.byteLength)
  })
  await t.test('documents and stylesheets retain their smaller limit', async () => {
    const types: ('document' | 'stylesheet')[] = ['document', 'stylesheet']
    for (const type of types) {
      await assert.rejects(
        readArenaReaderResource(new Response(bundle), type),
        /exceeds the reader size limit/,
      )
    }
  })
  await t.test('oversized scripts are rejected from headers or the streamed body', async () => {
    await assert.rejects(
      readArenaReaderResource(
        new Response('', { headers: { 'Content-Length': String(scriptLimit + 1) } }),
        'script',
      ),
      /exceeds the reader size limit/,
    )
    let cancelled = false
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array(documentLimit))
        controller.enqueue(new Uint8Array(documentLimit + 1))
      },
      cancel() {
        cancelled = true
      },
    })
    await assert.rejects(
      readArenaReaderResource(new Response(body), 'script'),
      /exceeds the reader size limit/,
    )
    assert.ok(cancelled)
  })
})

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
    text: 'Stable content',
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
    { ...extracted, readerHtml: `${markup}<p>A new paragraph.</p>` },
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
  const requests: string[] = []
  const articleParagraph = 'This full article paragraph must appear exactly once in the reader.'
  const postParagraph = 'A short post.'
  const fallbackParagraph = 'The fallback post keeps its text without mounting a Twitter widget.'
  const server = createServer((request, response) => {
    const url = new URL(request.url ?? '/', 'http://localhost')
    const source = url.searchParams.get('url') ?? ''
    requests.push(source)
    response.setHeader('Access-Control-Allow-Origin', '*')
    response.setHeader('Content-Type', 'application/json')
    if (source === 'https://c2.com/wiki/remodel/pages/ArenaAsyncFixture') {
      response.end(
        JSON.stringify({
          text: "The asynchronous article comes from the wiki API.\n\n'''A saved heading'''\n\nIts body is available even when the initial document contains no readable text.",
          date: 'September 15, 2026',
        }),
      )
    } else if (source === 'https://api.fxtwitter.com/arena_fixture/status/101') {
      response.end(
        JSON.stringify({
          tweet: {
            author: { screen_name: 'arena_fixture' },
            text: '',
            article: {
              title: 'An article from X',
              preview_text: articleParagraph,
              content: {
                blocks: [
                  {
                    type: 'unstyled',
                    text: articleParagraph,
                    inlineStyleRanges: [],
                    entityRanges: [],
                  },
                  {
                    type: 'header-two',
                    text: 'A section',
                    inlineStyleRanges: [],
                    entityRanges: [],
                  },
                  {
                    type: 'unstyled',
                    text: 'Read the source.',
                    inlineStyleRanges: [{ offset: 0, length: 4, style: 'Bold' }],
                    entityRanges: [{ offset: 9, length: 6, key: 0 }],
                  },
                ],
                entityMap: [
                  {
                    key: '0',
                    value: { type: 'LINK', data: { url: 'https://example.com/source' } },
                  },
                ],
              },
              cover_media: { media_info: { original_img_url: 'https://pbs.twimg.com/cover.jpg' } },
            },
          },
        }),
      )
    } else if (source === 'https://api.fxtwitter.com/arena_fixture/status/102') {
      response.end(
        JSON.stringify({
          tweet: { author: { screen_name: 'arena_fixture' }, text: postParagraph },
        }),
      )
    } else if (
      source ===
      'https://publish.twitter.com/oembed?url=https%3A%2F%2Fx.com%2Farena_fixture%2Fstatus%2F103&omit_script=true'
    ) {
      response.end(
        JSON.stringify({
          html: `<blockquote class="twitter-tweet"><p>${fallbackParagraph}</p>&mdash; Fixture (@arena_fixture) <a href="https://x.com/arena_fixture/status/103">September 15, 2026</a></blockquote><script src="https://platform.twitter.com/widgets.js"></script>`,
          author_url: 'https://x.com/arena_fixture',
          author_name: 'Fixture',
        }),
      )
    } else {
      response.statusCode = 503
      response.end('{}')
    }
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  assert.ok(address && typeof address !== 'string')
  const fixtureOrigin = `http://127.0.0.1:${address.port}`
  try {
    const [defuddle, purify, bundle] = await Promise.all([
      readFile(fileURLToPath(import.meta.resolve('defuddle/full')), 'utf8'),
      readFile(fileURLToPath(import.meta.resolve('dompurify/dist/purify.js')), 'utf8'),
      build({
        stdin: {
          contents: `export { extractArenaReaderDocument } from './arena-reader-extraction'
            export { serializeArenaReaderDocument } from './arena-reader-capture'
            export { sanitizeReaderHtml } from '../quartz/components/arena-feed/content'`,
          resolveDir: path.dirname(fileURLToPath(import.meta.url)),
          loader: 'ts',
        },
        bundle: true,
        write: false,
        format: 'iife',
        globalName: 'ArenaExtraction',
        platform: 'browser',
        target: 'es2022',
        keepNames: true,
      }),
    ])
    const paragraph =
      'The saved article discusses the design of an interface for reading links. Its paragraphs should survive extraction with their headings, quotations, source links, and footnotes intact. '
    const html = `<html><head><title>Reading links</title><base href="https://attacker.example/"></head><body>
      <nav><a href="/menu">Menu</a></nav><article><h1>Reading links</h1>
      <p>${paragraph.repeat(5)}</p><h2 id="details">Details</h2><p>${paragraph.repeat(5)}</p>
      <p><a href="#details">Local details</a> <a href="/next">Next article</a>
      <a href="#removed">Missing section</a>
      <a href="javascript:alert(1)">Bad link</a></p>
      <img src="/image.png" srcset="https://attacker.example/x 2x" onerror="alert(1)" style="position:fixed" alt="A figure">
      <table><tbody><tr><th scope="row">Value</th><td>3</td></tr></tbody></table>
      <pre><code>const x = 3</code></pre><blockquote>A quotation.</blockquote>
      <math display="block"><mfrac><mi>a</mi><mi>b</mi></mfrac></math>
      <script>document.body.dataset.executed='yes'; window.Defuddle=null;</script>
      <iframe src="https://attacker.example/"></iframe><form action="https://attacker.example"><input autofocus></form>
      </article></body></html>`
    const structuredHtml = `<html><head><title>Site archive</title>
      <script type="application/ld+json">{"@context":"https://schema.org","@type":"Article","headline":"Structured research","author":{"@type":"Person","name":"Research Author"}}</script>
      </head><body><nav>Navigation</nav><article>
      <p>${paragraph.repeat(6)}</p>
      <h2 id="equation">An equation</h2><script type="math/tex; mode=display">\\frac{a}{b}</script>
      <p>Reference <a href="#fn1" role="doc-noteref">1</a>.</p>
      <figure><img src="/placeholder.gif" data-src="/figure.png" width="1000" height="600" alt="Research figure"><figcaption>A research caption.</figcaption></figure>
      <picture><source srcset="/diagram-large.webp 1600w, /diagram-small.webp 800w"><img alt="Responsive diagram"></picture>
      <img src="/tracking.gif" width="1" height="1">
      <pre><code class="language-python">x = 3\n  print(x)</code></pre>
      <div class="callout source-layout" data-callout="note" data-arbitrary="discard"><div class="callout-title"><div class="callout-title-inner">A research note</div></div><div class="callout-content"><p>The callout body preserves its meaning.</p></div></div>
      <section role="doc-endnotes"><ol><li id="fn1"><p>Reference body. <a href="#equation">Back</a></p></li></ol></section>
      <p>${paragraph.repeat(4)}</p>
      <script>window.challenge = 'Verify you are human'; window.Defuddle = null;</script>
      </article><footer>Newsletter signup</footer></body></html>`
    const inputs: {
      html: string
      finalUrl: string
      idPrefix: string
      capture?: boolean
      captureTex?: string[]
    }[] = [
      html,
      structuredHtml,
      '<html><head><title>A short note</title></head><body><article><p>A short, complete paragraph can still be a useful article to save and read.</p></article></body></html>',
      '<html><head><title>Empty shell</title></head><body><script>window.payload = "A large amount of application code is not readable article content.".repeat(100)</script><style>body { color: red; }</style></body></html>',
      `<html><body>${'<span>x</span>'.repeat(30_001)}</body></html>`,
    ].map(html => ({ html, finalUrl: 'https://example.com/articles/test', idPrefix: 'snapshot-' }))
    inputs.push(
      {
        html: '<html><head><title>Wiki shell</title></head><body></body></html>',
        finalUrl: 'https://wiki.c2.com/?ArenaAsyncFixture',
        idPrefix: 'snapshot-',
      },
      {
        html: '<html><head><title>A saved fallback</title></head><body><article><p>This readable article remains available when its asynchronous API cannot return content.</p></article></body></html>',
        finalUrl: 'https://wiki.c2.com/?ArenaUnavailableFixture',
        idPrefix: 'snapshot-',
      },
      {
        html: `<html><head><title>Gustave Flaubert - Wikipedia</title><meta property="og:title" content="Gustave Flaubert - Wikipedia"></head><body>
          <nav>Wikipedia navigation</nav><main><h1 id="firstHeading">Gustave Flaubert</h1>
          <div id="mw-content-text"><div class="mw-parser-output">
          <p>Gustave Flaubert wrote <a href="/wiki/Madame_Bovary">Madame Bovary</a>.<sup class="reference" id="cite_ref-1"><a href="#cite_note-1">[1]</a></sup></p>
          <p>${paragraph.repeat(4)}</p>
          <h2 id="Life">Life</h2><p>${paragraph.repeat(4)}</p>
          <figure><a href="/wiki/File:Flaubert.jpg"><img resource="https://en.wikipedia.org/wiki/File:Flaubert.jpg" src="//upload.wikimedia.org/wikipedia/commons/1/10/Flaubert.jpg" loading="lazy" width="300" height="400" alt="Gustave Flaubert"></a><figcaption>A portrait of Flaubert.</figcaption></figure>
          <h2 id="References">References</h2><ol class="references"><li id="cite_note-1"><span class="mw-cite-backlink"><a href="#cite_ref-1">↑</a></span> <span class="reference-text">A literary biography of Flaubert.</span></li></ol>
          </div></div></main><footer>Wikipedia site footer</footer></body></html>`,
        finalUrl: 'https://en.wikipedia.org/wiki/Gustave_Flaubert',
        idPrefix: 'snapshot-',
      },
    )
    for (const id of ['101', '102', '103']) {
      inputs.push({
        html: '<!doctype html><html><head></head><body></body></html>',
        finalUrl: `https://x.com/arena_fixture/status/${id}`,
        idPrefix: 'snapshot-',
      })
    }
    inputs.push({
      html: `<html><head><title>MathJax research</title></head><body><article>
        <h1>MathJax research</h1><p>${paragraph.repeat(5)}</p>
        <p>The reflection direction is
          <mjx-container class="MathJax" jax="SVG"><svg><path d="M0 0"></path></svg>
            <mjx-assistive-mml display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow data-mjx-texclass="ORD"><mi mathvariant="bold">k</mi></mrow><mo stretchy="false">(</mo><mrow data-mjx-texclass="ORD"><mi mathvariant="bold">X</mi></mrow><mo stretchy="false">)</mo><mo>∈</mo><mi mathvariant="double-struck">R</mi></math></mjx-assistive-mml>
          </mjx-container> and its gate controls the update.</p>
        <h2>The delta operator</h2>
        <mjx-container class="MathJax" jax="SVG" display="true"><svg><path d="M0 0"></path></svg>
          <mjx-assistive-mml display="block"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><msub><mrow data-mjx-texclass="ORD"><mi mathvariant="bold">X</mi></mrow><mrow data-mjx-texclass="ORD"><mi>l</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><munder><mrow data-mjx-texclass="OP"><munder><mrow><mo stretchy="false">(</mo><mrow data-mjx-texclass="ORD"><mi mathvariant="bold">I</mi></mrow><mo>−</mo><msub><mi>β</mi><mi>l</mi></msub><msub><mrow data-mjx-texclass="ORD"><mi mathvariant="bold">k</mi></mrow><mi>l</mi></msub><msubsup><mrow data-mjx-texclass="ORD"><mi mathvariant="bold">k</mi></mrow><mi>l</mi><mi mathvariant="normal">⊤</mi></msubsup><mo stretchy="false">)</mo></mrow><mo>⏟</mo></munder></mrow><mrow data-mjx-texclass="ORD"><mtext>Delta Operator </mtext><mrow data-mjx-texclass="ORD"><mi mathvariant="bold">A</mi></mrow><mo stretchy="false">(</mo><mrow data-mjx-texclass="ORD"><mi mathvariant="bold">X</mi></mrow><mo stretchy="false">)</mo></mrow></munder></math></mjx-assistive-mml>
        </mjx-container>
        <p>${paragraph.repeat(4)}</p>
        <table><thead><tr><th>Regime</th><th>Gate</th></tr></thead><tbody><tr><td>Reflection</td><td>
          <mjx-container class="MathJax" jax="SVG"><svg><path d="M0 0"></path></svg>
            <mjx-assistive-mml display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>β</mi><mo>=</mo><mn>2</mn></math></mjx-assistive-mml>
          </mjx-container></td></tr></tbody></table>
        </article></body></html>`,
      finalUrl: 'https://example.com/mathjax-research',
      idPrefix: 'snapshot-',
      capture: true,
      captureTex: ['\\mathbf{k}(X)\\in\\mathbb{R}', '\\Xb', '\\beta=2'],
    })
    const substackHtml = `<html><head><title>A newsletter article</title>
      <meta property="og:title" content="A newsletter article"></head><body>
      <nav>Newsletter navigation</nav><article><h1>A newsletter article</h1>
      <div class="body markup"><p>${paragraph.repeat(5)}</p>
      <h2 id="a-section">A section</h2><p>${paragraph.repeat(3)}</p>
      <p><a href="#a-section">Read this section</a> and <a href="/p/another-article">another article</a>.</p>
      <figure><img data-src="https://substackcdn.com/image/figure.png" width="1200" height="800" alt="A newsletter figure"><figcaption>A figure caption.</figcaption></figure>
      <p>The final paragraph remains in the saved article.</p></div></article>
      <form><input type="email"><button>Subscribe</button></form>
      <script>document.querySelector('.body.markup').textContent = 'Subscribe to continue'; fetch('/api/v1/track');</script>
      </body></html>`
    for (const origin of ['https://example.substack.com', 'https://publisher.example']) {
      inputs.push({ html: substackHtml, finalUrl: `${origin}/p/article`, idPrefix: 'snapshot-' })
    }
    inputs.push({
      html: `<html><head><title>A long mathematical article</title>
        <script type="application/ld+json">{"@context":"https://schema.org","@type":"Article","headline":"A long mathematical article"}</script>
        <style>mjx-c { display: inline-block; }</style></head><body><article>
        <h1>A long mathematical article</h1><p>${paragraph.repeat(4)}</p>
        ${Array.from(
          { length: 577 },
          (_, index) => `<p>Equation ${index} remains readable.
          <mjx-container class="MathJax" display="${index % 2 === 0}">
            <mjx-math aria-hidden="true">${'<mjx-c class="mjx-c1D465 TEX-I"></mjx-c>'.repeat(110)}</mjx-math>
            <mjx-assistive-mml><math xmlns="http://www.w3.org/1998/Math/MathML"><mi mathvariant="bold">x</mi><mo>=</mo><mn>${index}</mn></math></mjx-assistive-mml>
          </mjx-container></p>`,
        ).join('')}
        <figure><img src="https://publisher.example/large-math#arena-figure-fixture" width="600" height="300" alt="GPU architecture"><figcaption>Generated diagram saved as an image.</figcaption></figure>
        <h2 id="conclusion">Conclusion</h2><p>The final paragraph survives the capture limit.</p>
        </article><script>window.publisherCode = true;</script></body></html>`,
      finalUrl: 'https://publisher.example/large-math',
      idPrefix: 'snapshot-',
      capture: true,
    })
    const input = JSON.stringify(inputs).replaceAll('<', '\\u003c')
    const fixture = path.join(directory, 'fixture.html')
    const css = compile(
      fileURLToPath(new URL('../quartz/components/styles/arena-feed.scss', import.meta.url)),
    ).css
    await writeFile(
      fixture,
      `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src ${fixtureOrigin}"><style>${css}</style></head><body><script>${defuddle}\n${purify}\n${bundle.outputFiles[0].text}\n
      globalThis.arenaReaderFetch = async url => {
        const response = await fetch(${JSON.stringify(fixtureOrigin)} + '/extractor?url=' + encodeURIComponent(url));
        return { body: await response.text(), status: response.status, contentType: response.headers.get('Content-Type'), url };
      };
      const serialized = document.createElement('script');
      serialized.textContent = 'globalThis.extractSerialized = ' + ArenaExtraction.extractArenaReaderDocument.toString();
      document.head.appendChild(serialized);
      const captureScript = document.createElement('script');
      captureScript.textContent = 'globalThis.captureSerialized = ' + ArenaExtraction.serializeArenaReaderDocument.toString();
      document.head.appendChild(captureScript);
      Promise.all(${input}.map(async input => {
        try {
          let capture = null;
          if (input.capture) {
            const doc = new DOMParser().parseFromString(input.html, 'text/html');
            if (input.captureTex) {
              globalThis.MathJax = { startup: { document: { math: [...doc.querySelectorAll('mjx-container')].map((typesetRoot, index) => ({
                math: input.captureTex[index], inputJax: { name: 'TeX' }, typesetRoot,
              })) } } };
            }
            const originalBytes = new TextEncoder().encode(input.html).byteLength;
            input.html = captureSerialized(2 * 1024 * 1024, doc);
            delete globalThis.MathJax;
            capture = {
              originalBytes, bytes: new TextEncoder().encode(input.html).byteLength,
              math: doc.querySelectorAll('math').length,
              display: doc.querySelectorAll('math[display="block"]').length,
              scripts: [...doc.scripts].map(script => script.type),
              bounded: captureSerialized(64, new DOMParser().parseFromString('<p>' + 'x'.repeat(100) + '</p>', 'text/html')).length,
            };
          }
          const result = await extractSerialized(input);
          const displayHtml = ArenaExtraction.sanitizeReaderHtml(result.readerHtml ?? '');
          const article = document.createElement('div');
          article.className = 'arena-reader-prose';
          article.style.width = '320px';
          article.innerHTML = displayHtml;
          document.body.appendChild(article);
          const equation = article.querySelector('math[display="block"]');
          const mathLayout = equation ? {
            display: getComputedStyle(equation).display,
            width: equation.getBoundingClientRect().width,
            articleWidth: article.getBoundingClientRect().width,
            overflowX: getComputedStyle(equation).overflowX,
          } : null;
          article.remove();
          return { ...result, displayHtml, mathLayout, capture };
        }
        catch (error) { return { error: error.message } }
      })).then(results => { document.body.textContent = btoa(unescape(encodeURIComponent(JSON.stringify(results)))); });</script></body></html>`,
    )
    const result = await run(
      chrome,
      [
        '--headless',
        '--disable-gpu',
        '--no-sandbox',
        '--disable-extensions',
        '--disable-background-networking',
        '--virtual-time-budget=12000',
        `--user-data-dir=${path.join(directory, 'chrome')}`,
        '--dump-dom',
        pathToFileURL(fixture).href,
      ],
      { timeout: 30_000, maxBuffer: 4 * 1024 * 1024 },
    )
    const encoded = result.stdout.match(/<body>([A-Za-z\d+/=]+)<\/body>/)?.[1]
    assert.ok(encoded, result.stderr.slice(-1000))
    const outputs: (
      | (ArenaExtractedDocument & {
          displayHtml: string
          capture: {
            originalBytes: number
            bytes: number
            math: number
            display: number
            scripts: string[]
            bounded: number
          } | null
          mathLayout: {
            display: string
            width: number
            articleWidth: number
            overflowX: string
          } | null
        })
      | { error: string }
    )[] = JSON.parse(Buffer.from(encoded, 'base64').toString('utf8'))
    const [
      output,
      structured,
      short,
      empty,
      oversized,
      asynchronous,
      fallback,
      wikipedia,
      xArticle,
      xPost,
      xFallback,
      mathJax,
      substack,
      customSubstack,
      largeMath,
    ] = outputs
    assert.ok(output && !('error' in output))
    assert.equal(output.title, 'Reading links')
    assert.ok(output.readerHtml?.includes(paragraph.trim()))
    assert.doesNotMatch(output.readerHtml ?? '', /Menu/)
    assert.deepEqual(output.imageUrls, ['https://example.com/image.png'])
    const markup = output.readerHtml
    assert.ok(markup)
    assert.match(markup, /id="snapshot-details"/)
    assert.match(markup, /href="#snapshot-details"/)
    assert.match(markup, /href="https:\/\/example.com\/next"/)
    assert.match(markup, /href="https:\/\/example.com\/articles\/test#removed"/)
    assert.match(markup, /data-arena-image="0"/)
    assert.match(markup, /<table>/)
    assert.match(markup, /<code>const x = 3<\/code>/)
    assert.match(markup, /<math/)
    assert.doesNotMatch(
      markup,
      /<script|<iframe|<form|<input|onerror|javascript:|srcset|style=|attacker\.example/,
    )
    await t.test('Substack HTML reaches Defuddle without running publisher scripts', () => {
      for (const { article, origin } of [
        { article: substack, origin: 'https://example.substack.com' },
        { article: customSubstack, origin: 'https://publisher.example' },
      ]) {
        assert.ok(article && !('error' in article))
        assert.equal(article.title, 'A newsletter article')
        assert.ok(article.readerHtml?.includes(paragraph.trim()))
        assert.match(article.displayHtml, /The final paragraph remains in the saved article/)
        assert.match(article.displayHtml, /id="snapshot-a-section"/)
        assert.match(article.displayHtml, /href="#snapshot-a-section"/)
        assert.ok(article.displayHtml.includes(`href="${origin}/p/another-article"`))
        assert.match(article.displayHtml, /A figure caption/)
        assert.deepEqual(article.imageUrls, ['https://substackcdn.com/image/figure.png'])
        assert.doesNotMatch(article.displayHtml, /Newsletter navigation|Subscribe|<script|<form/)
        assert.equal(arenaReaderFailureSignals(article), null)
      }
      assert.equal(
        requests.some(url => url.includes('/api/v1/track')),
        false,
      )
    })
    await t.test('Defuddle extracts one full X article with formatting and saved images', () => {
      assert.ok(xArticle && !('error' in xArticle))
      assert.equal(xArticle.title, 'An article from X')
      assert.ok(xArticle.readerHtml)
      assert.equal(xArticle.text.split(articleParagraph).length - 1, 1)
      assert.match(xArticle.displayHtml, /<strong>Read<\/strong>/)
      assert.match(xArticle.displayHtml, /href="https:\/\/example.com\/source"/)
      assert.match(xArticle.displayHtml, /<h2[^>]*>A section<\/h2>/)
      assert.deepEqual(xArticle.imageUrls, ['https://pbs.twimg.com/cover.jpg'])
      assert.doesNotMatch(xArticle.displayHtml, /twitter-tweet|widgets\.js|<script|<iframe/)
      assert.equal(arenaReaderFailureSignals(xArticle), null)
      assert.equal(
        requests.filter(url => url === 'https://api.fxtwitter.com/arena_fixture/status/101').length,
        1,
      )
    })
    await t.test('a short X post remains readable and renders once', () => {
      assert.ok(xPost && !('error' in xPost))
      assert.equal(xPost.text, postParagraph)
      assert.equal(arenaReaderFailureSignals(xPost, 1), null)
      assert.doesNotMatch(xPost.displayHtml, /twitter-tweet|widgets\.js|<iframe/)
    })
    await t.test('Defuddle uses the oEmbed text fallback without installing its widget', () => {
      assert.ok(xFallback && !('error' in xFallback))
      assert.equal(xFallback.text, fallbackParagraph)
      assert.doesNotMatch(xFallback.displayHtml, /twitter-tweet|widgets\.js|<script|<iframe/)
      assert.equal(arenaReaderFailureSignals(xFallback), null)
      assert.ok(requests.includes('https://api.fxtwitter.com/arena_fixture/status/103'))
    })
    await t.test(
      'Wikipedia uses its article extractor and retains links, images, and citations',
      () => {
        assert.ok(wikipedia && !('error' in wikipedia))
        assert.equal(wikipedia.title, 'Gustave Flaubert')
        assert.ok(wikipedia.readerHtml)
        assert.match(wikipedia.readerHtml, /Gustave Flaubert wrote/)
        assert.match(wikipedia.readerHtml, /href="https:\/\/en.wikipedia.org\/wiki\/Madame_Bovary"/)
        assert.match(wikipedia.readerHtml, /id="snapshot-Life"/)
        assert.match(wikipedia.readerHtml, /A literary biography of Flaubert/)
        assert.match(wikipedia.readerHtml, /href="#snapshot-fn:1"/)
        assert.match(wikipedia.readerHtml, /id="snapshot-fn:1"/)
        assert.deepEqual(wikipedia.imageUrls, [
          'https://upload.wikimedia.org/wikipedia/commons/1/10/Flaubert.jpg',
        ])
        assert.doesNotMatch(wikipedia.readerHtml, /Wikipedia navigation|Wikipedia site footer/)
        assert.equal(arenaReaderFailureSignals(wikipedia), null)
      },
    )
    await t.test('Defuddle preserves structured titles, equations, figures, and footnotes', () => {
      assert.ok(structured && !('error' in structured))
      assert.equal(structured.title, 'Structured research')
      assert.ok(structured.readerHtml)
      assert.match(structured.readerHtml, /<math[\s>]/)
      assert.match(structured.readerHtml, /<mfrac>/)
      assert.match(structured.readerHtml, /data-latex="\\frac\{a\}\{b\}"/)
      assert.match(structured.readerHtml, /<code data-lang="python">x = 3\n  print\(x\)<\/code>/)
      assert.match(structured.readerHtml, /class="arena-callout"/)
      assert.match(structured.readerHtml, /data-callout="note"/)
      assert.match(structured.readerHtml, /class="arena-callout-title-inner"/)
      assert.match(structured.readerHtml, /The callout body preserves its meaning/)
      assert.doesNotMatch(structured.readerHtml, /source-layout|data-arbitrary|Newsletter signup/)
      assert.match(structured.readerHtml, /Reference body/)
      assert.match(structured.readerHtml, /href="#snapshot-fn:1"/)
      assert.match(structured.readerHtml, /id="snapshot-fn:1"/)
      assert.match(structured.readerHtml, /href="#snapshot-fnref:1"/)
      assert.match(structured.readerHtml, /id="snapshot-fnref:1"/)
      assert.ok(structured.imageUrls.includes('https://example.com/figure.png'))
      assert.ok(structured.imageUrls.includes('https://example.com/diagram-large.webp'))
      assert.ok(!structured.imageUrls.includes('https://example.com/placeholder.gif'))
      assert.ok(!structured.imageUrls.includes('https://example.com/tracking.gif'))
      for (const markup of [structured.readerHtml, structured.displayHtml]) {
        assert.doesNotMatch(markup, /<script|<style|onerror|javascript:|srcset=/)
      }
      assert.match(structured.readerHtml, /data-arena-image="0"/)
      assert.doesNotMatch(structured.displayHtml, /data-arena-image=/)
      assert.doesNotMatch(structured.text, /Verify you are human|window\.challenge/)
      assert.equal(arenaReaderFailureSignals(structured), null)
      assert.match(structured.displayHtml, /class="arena-callout"/)
      assert.match(structured.displayHtml, /data-callout="note"/)
      assert.match(structured.displayHtml, /data-lang="python"/)
      assert.match(structured.displayHtml, /class="katex-display"/)
      assert.match(structured.displayHtml, /href="#snapshot-fn:1"/)
    })
    await t.test(
      'MathJax TeX renders with KaTeX while unsupported macros retain captured MathML',
      () => {
        assert.ok(mathJax && !('error' in mathJax))
        assert.ok(mathJax.readerHtml)
        for (const markup of [mathJax.readerHtml, mathJax.displayHtml]) {
          assert.equal((markup.match(/<math[\s>]/g) ?? []).length, 3)
          assert.match(markup, /<math[^>]*display="block"/)
          assert.match(markup, /<munder>/)
          assert.match(markup, /<msubsup>/)
          assert.match(markup, /<mtext>Delta Operator <\/mtext>/)
          assert.doesNotMatch(markup, /mjx-container|mjx-assistive-mml|<svg|<script|\\kb/)
        }
        assert.ok(mathJax.readerHtml.includes('data-latex="\\mathbf{k}(X)\\in\\mathbb{R}"'))
        assert.ok(mathJax.readerHtml.includes('data-latex="\\Xb"'))
        assert.equal((mathJax.displayHtml.match(/class="katex"/g) ?? []).length, 2)
        assert.match(mathJax.displayHtml, /class="katex-mathml"/)
        assert.match(mathJax.displayHtml, /<td>\s*<span class="katex"/)
        assert.match(mathJax.readerHtml, /<mi mathvariant="bold">k<\/mi>/)
        assert.match(mathJax.readerHtml, /<mi mathvariant="bold">X<\/mi>/)
        assert.match(mathJax.displayHtml, /<mi mathvariant="normal">𝐗<\/mi>/)
        assert.ok(mathJax.mathLayout)
        assert.equal(mathJax.mathLayout.display, 'block math')
        assert.ok(mathJax.mathLayout.width <= mathJax.mathLayout.articleWidth)
        assert.equal(mathJax.mathLayout.overflowX, 'auto')
      },
    )
    await t.test(
      'short articles remain readable and executable payloads do not count as content',
      () => {
        assert.ok(short && !('error' in short))
        assert.match(short.readerHtml ?? '', /A short, complete paragraph/)
        assert.equal(arenaReaderFailureSignals(short), null)
        assert.ok(empty && !('error' in empty))
        assert.equal(empty.text, '')
        assert.equal(empty.articleLength, 0)
        assert.equal(arenaReaderFailureSignals(empty)?.reason, 'empty')
        assert.equal(arenaReaderFailureSignals(empty, 1)?.reason, 'empty')
      },
    )
    await t.test('documents remain bounded before Defuddle runs', () => {
      assert.ok(oversized && 'error' in oversized)
      assert.match(oversized.error, /reader element limit/)
    })
    await t.test(
      'rendered math is compacted before size limits without truncating the article',
      () => {
        assert.ok(largeMath && !('error' in largeMath))
        assert.ok(largeMath.capture)
        assert.ok(largeMath.capture.originalBytes > 2 * 1024 * 1024)
        assert.ok(largeMath.capture.bytes < 2 * 1024 * 1024)
        assert.equal(largeMath.capture.math, 577)
        assert.equal(largeMath.capture.display, 289)
        assert.deepEqual(largeMath.capture.scripts, ['application/ld+json'])
        assert.equal(largeMath.capture.bounded, 65)
        for (const markup of [largeMath.readerHtml ?? '', largeMath.displayHtml]) {
          assert.equal((markup.match(/<math\b/g) ?? []).length, 577)
          assert.match(markup, /The final paragraph survives the capture limit/)
          assert.match(markup, /Generated diagram saved as an image/)
          assert.doesNotMatch(markup, /mjx-container|mjx-c |<script|<style/)
        }
        assert.deepEqual(largeMath.imageUrls, [
          'https://publisher.example/large-math#arena-figure-fixture',
        ])
      },
    )
    await t.test(
      'URL extraction uses the real asynchronous wiki extractor with sync fallback',
      () => {
        assert.ok(asynchronous && !('error' in asynchronous))
        assert.equal(asynchronous.title, 'Arena Async Fixture')
        assert.match(
          asynchronous.readerHtml ?? '',
          /The asynchronous article comes from the wiki API/,
        )
        assert.ok(requests.includes('https://c2.com/wiki/remodel/pages/ArenaAsyncFixture'))
        assert.ok(fallback && !('error' in fallback))
        assert.match(fallback.readerHtml ?? '', /This readable article remains available/)
        assert.ok(requests.includes('https://c2.com/wiki/remodel/pages/ArenaUnavailableFixture'))
      },
    )
  } finally {
    server.closeAllConnections()
    await new Promise<void>((resolve, reject) =>
      server.close(error => (error ? reject(error) : resolve())),
    )
    await rm(directory, { recursive: true, force: true })
  }
})
