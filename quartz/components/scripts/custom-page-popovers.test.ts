import { connect } from '@cloudflare/puppeteer'
import { build } from 'esbuild'
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { access, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import { compile } from 'sass-embedded'
import { notebookPreviewFixture } from './custom-page-popovers-fixtures'

const artifacts = 'docs/verification/custom-page-popovers/artifacts'

test('custom page and Curius link popovers use the shared browser lifecycle', async t => {
  let chrome: string | undefined
  for (const candidate of [
    process.env.ARENA_TEST_CHROME,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Helium.app/Contents/MacOS/Helium',
    '/usr/bin/chromium',
  ]) {
    if (!candidate) continue
    try {
      await access(candidate)
      chrome = candidate
      break
    } catch {
      // Try the next installed browser.
    }
  }
  if (!chrome) return t.skip('Chromium is required for the popover interaction fixture.')

  const stylesheet = compile('quartz/components/styles/popover.scss').css
  const bundle = await build({
    stdin: {
      resolveDir: path.dirname(fileURLToPath(import.meta.url)),
      loader: 'ts',
      contents: `
        import './popover.inline'
        import { createLinkEl, createTitle, _SENTINEL } from './curius'
        const cleanups = []
        window.addCleanup = fn => cleanups.push(fn)
        window.notifyNav = () => {}
        document.dispatchEvent(new CustomEvent('nav', { detail: { url: 'fixture' } }))
        const passed = []
        function check(ok, label) {
          if (!ok) throw Error(label)
          passed.push(label)
        }
        async function waitFor(fn) {
          const start = Date.now()
          while (!fn()) {
            if (Date.now() - start > 2500) throw Error('Popover did not settle')
            await new Promise(r => setTimeout(r, 20))
          }
        }
        function enter(a) {
          const rect = a.getBoundingClientRect()
          a.dispatchEvent(new MouseEvent('mouseover', {
            bubbles: true, clientX: rect.left + 10, clientY: rect.top + 10,
          }))
        }
        function escape() {
          document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))
        }
        async function preview(a, kind) {
          escape()
          enter(a)
          await waitFor(() => document.querySelector('.active-popover .page-preview-card'))
          const card = document.querySelector('.active-popover .page-preview-card')
          check(card.dataset.kind === kind, kind + ': selected the correct preview')
          check(!card.querySelector('input, button, iframe, canvas, [id]'), kind + ': no runtime shell or duplicate IDs')
          check(card.querySelector('.page-preview-title').textContent.length > 0, kind + ': has a title')
          return card
        }
        void (async () => {
          try {
            const row = createLinkEl({
              ..._SENTINEL, id: 101, title: 'Saved <T> & title', link: 'https://example.org/article',
              snippet: '<p>A saved <strong>excerpt</strong> &amp; context.</p>',
              createdDate: '2026-10-20T12:00:00Z', highlights: [{ highlight: 'Existing highlight' }],
              topics: [{ topic: 'public topic', public: true }, { topic: 'private topic', public: false }],
            })
            document.querySelector('#curius-fragments').append(row)
            const a = row.querySelector('.curius-item-link > a')
            const card = await preview(a, 'curius-link')
            check(card.textContent.includes('Saved <T> & title'), 'literal title survives')
            check(card.textContent.includes('A saved excerpt & context.'), 'saved excerpt is plain text')
            check(card.textContent.includes('public topic') && !card.textContent.includes('private topic'), 'only public topics appear')
            check(card.textContent.includes('1 highlight'), 'highlight count appears')
            check(card.querySelector('a').href === a.href, 'source link retains its destination')
            let requested = false
            document.addEventListener('curius:preview', () => { requested = true }, { once: true })
            a.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))
            check(requested, 'click still opens the Curius reader')
            check(!document.querySelector('.active-popover'), 'reader activation dismisses the hover card')
            const { topics, highlights, ...trailLink } = _SENTINEL
            const trailRow = createTitle({ Link: { ...trailLink, title: 'Trail link', link: 'https://example.org/trail' } })
            document.querySelector('#curius-fragments').append(trailRow)
            const trailCard = await preview(trailRow.querySelector('a'), 'curius-link')
            check(trailCard.textContent.includes('Trail link'), 'trail links without topics or highlights still render')
            await preview(document.querySelector('#curius-page'), 'curius')
            check(document.querySelectorAll('.active-popover .page-preview-list li').length === 3, 'Curius page shows three saved links')
            check(!document.querySelector('.active-popover a[href^="javascript:"]'), 'unsafe feed destinations are dropped')
            await preview(document.querySelector('#curius-offline'), 'curius')
            check(document.querySelectorAll('.active-popover .page-preview-list li').length === 0, 'feed failure leaves an honest page summary')
            check(document.querySelector('.active-popover').textContent.includes('Saved links'), 'feed failure retains the page description')
            for (const kind of ['arena', 'flashcards', 'masonry', 'letter', 'notebook']) {
              await preview(document.querySelector('#' + kind), kind)
              check(document.querySelectorAll('.active-popover .page-preview-list li').length <= 4, kind + ': bounded rows')
            }
            check(document.querySelector('.active-popover').textContent.includes('Functional parsing'), 'notebook shows its section headings')
            check(!document.querySelector('.active-popover').textContent.includes('Generated output'), 'notebook excludes headings inside code output')
            check(!document.querySelector('.active-popover script, .active-popover [data-notebook-runtime]'), 'notebook runtime is not copied into the preview')
            const notebookCard = document.querySelector('.active-popover')
            check(notebookCard.textContent.includes('runtime: python, javascript'), 'notebook lists each executable runtime')
            check(notebookCard.textContent.includes('6 blocks'), 'notebook counts source cells including blank and raw cells')
            check(notebookCard.textContent.includes('2 code · 3 text · 1 raw'), 'notebook shows its block breakdown')
            check(notebookCard.querySelector('.page-preview-description').textContent === 'Build parsers from small functions.', 'notebook uses introductory prose when no description is authored')
            const anchor = document.querySelector('#notebook')
            for (const [expected, x, y] of [
              ['bottom', innerWidth / 2, 30],
              ['top', innerWidth / 2, innerHeight - 30],
              ['right', 30, innerHeight / 2],
              ['left', innerWidth - 30, innerHeight / 2],
            ]) {
              anchor.style.cssText = 'position:fixed;transform:translate(-50%,-50%);left:' + x + 'px;top:' + y + 'px'
              await preview(anchor, 'notebook')
              await waitFor(() => document.querySelector('.active-popover')?.dataset.placement === expected)
              const popup = document.querySelector('.active-popover')
              const rect = popup.getBoundingClientRect()
              check(popup.dataset.placement === expected, 'available space selects ' + expected)
              check(rect.left >= 7 && rect.top >= 7 && rect.right <= innerWidth - 7 && rect.bottom <= innerHeight - 7, expected + ': preview fits inside viewport gutter')
              const arrow = getComputedStyle(popup.querySelector('.popover-arrow'), '::before')
              const inner = getComputedStyle(popup.querySelector('.popover-inner'))
              check(arrow.getPropertyValue('border-' + expected + '-color') === inner.borderTopColor, expected + ': arrow color matches the frame')
            }
            anchor.style.cssText = ''
            const described = await preview(document.querySelector('#notebook-described'), 'notebook')
            check(described.querySelector('.page-preview-description').textContent === 'An authored overview.', 'notebook prefers its authored description')
            const staticCard = await preview(document.querySelector('#notebook-static'), 'notebook')
            check(staticCard.textContent.includes('runtime: static'), 'static notebook does not imply an executable runtime')
            check(staticCard.textContent.includes('6 blocks'), 'static notebook retains source block counts')
            escape()
            check(!document.querySelector('.active-popover'), 'Escape dismisses without moving focus')
            enter(document.querySelector('#ordinary'))
            await waitFor(() => document.querySelector('.active-popover article'))
            check(document.querySelector('.active-popover').textContent.includes('Ordinary note body'), 'ordinary notes retain the article preview')
            escape()
            enter(document.querySelector('#stream-entry'))
            await waitFor(() => document.querySelector('.active-popover .stream-entry'))
            check(document.querySelectorAll('.active-popover .stream-entry').length === 1, 'stream entry links retain the isolated entry')
            escape()
            document.querySelector('#optout').dispatchEvent(new MouseEvent('mouseover', { bubbles: true }))
            await new Promise(r => setTimeout(r, 50))
            check(!document.querySelector('.active-popover'), 'explicit popover opt-out survives')
            for (const fn of cleanups) fn()
            check(document.querySelectorAll('.popover').length === 0, 'navigation removes cached popovers')
            document.body.textContent = btoa(JSON.stringify({ passed }))
          } catch (error) {
            document.body.textContent = btoa(JSON.stringify({ passed, error: String(error.stack ?? error) }))
          }
          document.body.dataset.fixtureDone = 'true'
        })()
      `,
    },
    bundle: true,
    write: false,
    platform: 'browser',
    target: 'es2022',
  })
  const requests: string[] = []
  let unavailableFeed = false
  const pages: Record<string, string> = {
    '/curius-offline': '<body data-slug="curius"><div id="curius-fragments"></div>',
    '/curius': '<body data-slug="curius"><input id="curius-bar"><div id="curius-fragments"></div>',
    '/arena':
      '<body data-slug="arena"><h1>are.na</h1><article class="arena-index"><div class="arena-channel-row-header"><h2><a href="/arena/one">One channel</a></h2></div><input></article>',
    '/flashcards':
      '<body data-slug="flashcards" data-flashcards="true"><h1>Review deck</h1><article><button>Grade</button><h2>Topic</h2></article>',
    '/masonry':
      '<body data-slug="photos"><h1>Photos</h1><article class="masonry-container"><button>Chaos mode</button><img alt="A landscape"></article>',
    '/letter':
      '<body data-slug="letter" data-layout="letter"><h1>A letter</h1><article><h2>First section</h2><p>Words of the letter.</p></article>',
    '/notebook': await notebookPreviewFixture(true),
    '/notebook-described': await notebookPreviewFixture(true, 'An authored overview.'),
    '/notebook-static': await notebookPreviewFixture(false),
    '/ordinary':
      '<body data-slug="ordinary" data-layout="default"><article class="popover-hint">Ordinary note body</article>',
    '/stream':
      '<body data-slug="stream"><article class="stream popover-hint"><ol class="stream-feed"><li class="stream-entry" id="one">Entry one</li><li class="stream-entry" id="two">Entry two</li></ol></article>',
  }
  const server = createServer((request, response) => {
    const url = new URL(request.url ?? '/', 'http://localhost')
    requests.push(url.pathname + url.search)
    if (url.pathname === '/fixture.js') {
      response.setHeader('Content-Type', 'application/javascript')
      response.end(bundle.outputFiles[0].text)
    } else if (url.pathname === '/api/curius') {
      response.setHeader('Content-Type', 'application/json')
      if (unavailableFeed) {
        response.writeHead(503)
        response.end('{}')
        return
      }
      response.end(
        JSON.stringify({
          links: [
            { link: 'javascript:alert(1)', title: 'Invalid destination' },
            ...[1, 2, 3, 4].map(id => ({
              id,
              link: 'https://example.org/' + id,
              title: 'Saved link ' + id,
            })),
          ],
        }),
      )
    } else if (pages[url.pathname]) {
      if (url.pathname === '/curius-offline') unavailableFeed = true
      response.setHeader('Content-Type', 'text/html')
      response.end(
        '<!doctype html><title>Fixture page</title><meta name="description" content="' +
          (url.pathname.startsWith('/notebook') ? '' : 'A page description.') +
          '">' +
          pages[url.pathname] +
          '</body>',
      )
    } else {
      if (url.pathname === '/') unavailableFeed = false
      response.setHeader('Content-Type', 'text/html')
      response.end(`<!doctype html><style>:root{--light:#fffcf0;--gray:#b7b5ac;--darkgray:#6f6e69;--dark:#100f0f;--z-popover:100;--radius-none:0;--text-ui:14px;--text-meta:12px}${stylesheet}</style><body data-slug="fixture">
        <ul id="curius-fragments"></ul>
        <a id="curius-offline" class="internal" href="/curius-offline">offline curius</a>
        ${['curius', 'arena', 'flashcards', 'masonry', 'letter', 'notebook', 'ordinary'].map(slug => `<a id="${slug === 'curius' ? 'curius-page' : slug}" class="internal" href="/${slug}">${slug}</a>`).join(' ')}
        <a id="stream-entry" class="internal" href="/stream#one">stream entry</a>
        <a id="notebook-described" class="internal" href="/notebook-described">described notebook</a>
        <a id="notebook-static" class="internal" href="/notebook-static">static notebook</a>
        <a id="optout" class="internal" data-no-popover href="/curius">opt out</a>
        <script src="/fixture.js"></script></body>`)
    }
  })
  const directory = await mkdtemp(path.join(tmpdir(), 'custom-popovers-'))
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  assert.ok(address && typeof address !== 'string')
  const browserProcess = spawn(
    chrome,
    [
      '--headless',
      '--no-first-run',
      '--no-default-browser-check',
      '--disable-gpu',
      '--no-sandbox',
      '--disable-extensions',
      '--disable-background-networking',
      '--remote-debugging-port=0',
      `--user-data-dir=${directory}`,
      'about:blank',
    ],
    { stdio: ['ignore', 'ignore', 'pipe'] },
  )
  let browser: Awaited<ReturnType<typeof connect>> | undefined
  try {
    const endpoint = await new Promise<string>((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Chromium did not start')), 10_000)
      let stderr = ''
      browserProcess.stderr.on('data', (chunk: Buffer) => {
        stderr += chunk.toString()
        const endpoint = stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/)?.[1]
        if (endpoint) {
          clearTimeout(timeout)
          resolve(endpoint)
        }
      })
      browserProcess.once('error', error => {
        clearTimeout(timeout)
        reject(error)
      })
    })
    browser = await connect({
      browserWSEndpoint: endpoint,
      defaultViewport: { width: 1280, height: 900 },
    })
    const page = await browser.newPage()
    await page.goto(`http://127.0.0.1:${address.port}/`, { waitUntil: 'domcontentloaded' })
    await page.waitForFunction(() => document.body.dataset.fixtureDone === 'true', {
      timeout: 20_000,
    })
    await mkdir(artifacts, { recursive: true })
    await writeFile(`${artifacts}/fixture.html`, await page.content())
    const encoded = await page.evaluate(() => document.body.textContent)
    assert.ok(encoded)
    const output: { passed: string[]; error?: string } = JSON.parse(
      Buffer.from(encoded, 'base64').toString('utf8'),
    )
    await writeFile(`${artifacts}/fixture.json`, JSON.stringify({ ...output, requests }, null, 2))
    assert.equal(output.error, undefined, output.error)
    assert.ok(output.passed.length >= 30)
    assert.ok(
      !requests.some(url => url.includes('query=preview')),
      'hover never extracts an article',
    )
  } finally {
    await browser?.disconnect()
    browserProcess.kill('SIGKILL')
    await new Promise<void>(resolve => server.close(() => resolve()))
    await rm(directory, { recursive: true, force: true })
  }
})
