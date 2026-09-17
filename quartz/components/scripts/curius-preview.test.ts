import { build } from 'esbuild'
import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { access, mkdtemp, rm } from 'node:fs/promises'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'
import { curiusPreviewImagePath, type CuriusPreviewResponse } from '../../util/curius-preview'

const run = promisify(execFile)

test('Curius previews preserve navigation, identify saved text, and render video and repository previews', async t => {
  let chrome: string | undefined
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
      // Try the next installed browser.
    }
  }
  if (!chrome) return t.skip('Chromium is required for the Curius interaction fixture.')

  const bundle = await build({
    stdin: {
      contents: `
        import { setupCuriusPreview } from './curius-preview'
        import { createLinkEl, _SENTINEL } from './curius'
        const cleanups = []
        window.addCleanup = callback => cleanups.push(callback)
        document.dispatchEvent(new CustomEvent('nav', { detail: { url: 'curius' } }))
        setupCuriusPreview()
        const passed = []
        const check = (condition, message) => {
          if (!condition) throw new Error(message)
          passed.push(message)
        }
        const query = selector => {
          const element = document.querySelector(selector)
          if (!element) throw new Error('Missing fixture element ' + selector)
          return element
        }
        const click = (element, options = {}) => {
          const event = new MouseEvent('click', { bubbles: true, cancelable: true, ...options })
          element.dispatchEvent(event)
          return event
        }
        const escape = () => document.dispatchEvent(new KeyboardEvent('keydown', {
          key: 'Escape', bubbles: true, cancelable: true,
        }))
        const checkNative = (element, options, message) => {
          let preserved = false
          element.addEventListener('click', event => {
            preserved = !event.defaultPrevented
            event.preventDefault()
          }, { once: true })
          click(element, options)
          check(preserved, message)
        }
        const waitFor = async predicate => {
          const deadline = Date.now() + 1500
          while (!predicate()) {
            if (Date.now() > deadline) throw new Error('Timed out waiting for preview state')
            await new Promise(resolve => setTimeout(resolve, 20))
          }
        }
        void (async () => {
        try {
          const firstLink = {
            ..._SENTINEL, id: 101, title: 'First saved link',
            link: location.origin + '/original/first', snippet: 'First saved paragraph',
            topics: [{ topic: 'A topic', slug: 'topic', public: true }],
            highlights: [{ id: 1, highlight: 'A saved quotation' }],
          }
          const secondLink = {
            ..._SENTINEL, id: 102, title: 'Second saved link',
            link: location.origin + '/original/second', snippet: 'Second saved paragraph',
          }
          const firstRow = createLinkEl(firstLink)
          const secondRow = createLinkEl(secondLink)
          query('#curius-fragments').append(firstRow, secondRow)
          const first = query('#curius-item-101 .curius-item-link > a')
          const second = query('#curius-item-102 .curius-item-link > a')
          const panel = query('#curius-preview')
          const toggle = query('#curius-preview-toggle')
          const original = query('#curius-preview-original')
          const source = query('#curius-preview-source')
          const close = query('#curius-preview-close')
          check(!document.querySelector('.curius-preview-trigger'), 'rows have no preview buttons')
          check(!first.hasAttribute('title'), 'title anchors have no preview hover tooltip')
          check(!first.hasAttribute('aria-expanded'), 'open state belongs to the toolbar toggle')

          toggle.click()
          check(!panel.hidden && source.textContent === firstLink.title, 'toggle opens the first saved entry')
          check(toggle.getAttribute('aria-expanded') === 'true', 'toggle announces its open state')
          check(toggle.getAttribute('aria-label') === 'Close preview', 'toggle label describes closing')
          check(firstRow.classList.contains('preview-active'), 'preview selects the complete saved row')
          check(!original.hidden && original.href === firstLink.link, 'source action targets the selected original')
          checkNative(original, {}, 'source action keeps native navigation')
          checkNative(source, {}, 'source title keeps native navigation')
          toggle.click()
          check(panel.hidden && original.hidden && !original.hasAttribute('href'), 'closing clears the original action')
          check(toggle.getAttribute('aria-expanded') === 'false', 'toggle announces its closed state')
          check(document.activeElement === toggle, 'closing from the toggle keeps toolbar focus')

          click(first)
          click(second)
          check(source.textContent === secondLink.title, 'clicking a second entry replaces the preview')
          check(original.href === secondLink.link, 'the original action follows the latest selected entry')
          check(secondRow.classList.contains('preview-active') && !firstRow.classList.contains('preview-active'), 'selection follows the current entry')
          escape()
          check(panel.hidden && document.activeElement === second, 'Escape closes and restores the invoking link')
          toggle.click()
          check(source.textContent === secondLink.title, 'toggle reopens the last selected entry')
          close.click()
          check(document.activeElement === toggle, 'close button restores a toolbar invocation')

          click(firstRow)
          check(!panel.hidden && source.textContent === firstLink.title, 'primary row clicks select the row article')
          close.click()
          click(query('#curius-highlights-101'))
          check(panel.hidden, 'highlight interactions do not select the containing row')
          checkNative(query('.curius-item-tags a'), { shiftKey: true }, 'topic links retain modifier navigation')
          const fullTextLink = {
            ..._SENTINEL, id: 103, title: 'Stored article',
            link: location.origin + '/original/full', snippet: 'A short excerpt',
            metadata: { full_text: 'First complete paragraph.\\n\\nSecond complete paragraph. <T> <random_label>' },
          }
          const fullTextRow = createLinkEl(fullTextLink)
          query('#curius-fragments').append(fullTextRow)
          click(fullTextRow)
          check(query('.curius-preview-summary').title === 'Article text saved by Curius', 'full_text provenance is available on hover')
          check(!document.querySelector('.curius-preview-summary h3'), 'saved content has no extra heading')
          check(query('.curius-preview-summary').textContent.includes('Second complete paragraph.'), 'the complete saved article is available while refreshing')
          check(query('.curius-preview-summary').textContent.includes('<T> <random_label>'), 'plain saved text preserves literal technical tokens')
          check(document.querySelectorAll('.curius-preview-summary > p').length === 2, 'saved article paragraph breaks are preserved')
          await waitFor(() => panel.getAttribute('aria-busy') === 'false')
          check(query('.curius-preview-summary').textContent.includes('Second complete paragraph.'), 'failed refresh retains the saved article')
          check(!query('#curius-preview-retry').disabled, 'failed refresh remains retryable')
          const videoLink = {
            ..._SENTINEL, id: 104, title: 'A YouTube video',
            link: 'https://youtu.be/1J2iN6I0gCQ?t=1m20s', snippet: 'Video description',
            highlights: [{ id: 2, highlight: 'A video note' }],
          }
          const videoRow = createLinkEl(videoLink)
          query('#curius-fragments').append(videoRow)
          click(videoRow)
          const frame = query('.curius-preview-video')
          check(frame.src === 'https://www.youtube-nocookie.com/embed/1J2iN6I0gCQ?start=80', 'YouTube preview preserves the video and start time')
          check(frame.title === videoLink.title && frame.allowFullscreen, 'video frame has a title and fullscreen support')
          check(frame.referrerPolicy === 'strict-origin-when-cross-origin', 'video frame supplies the player origin referrer')
          check(frame.sandbox.contains('allow-scripts') && frame.sandbox.contains('allow-presentation'), 'video frame grants the player required capabilities')
          check(panel.getAttribute('aria-busy') === 'false' && !query('#curius-preview-retry').disabled, 'YouTube opens without waiting for article extraction')
          check(original.href === videoLink.link, 'video original action targets the selected video')
          check(!document.querySelector('.curius-preview-summary') && query('.curius-preview-highlights').textContent.includes('A video note'), 'video preview keeps highlights without duplicating its description')
          if (matchMedia('(max-width: 800px)').matches) {
            toggle.focus()
            check(document.activeElement === close, 'mobile focus remains within the panel after leaving a frame')
          }
          close.click()
          check(!document.querySelector('.curius-preview-video'), 'closing removes the player to stop playback')
          const repositoryLink = {
            ..._SENTINEL, id: 105, title: 'owner/repository',
            link: 'https://github.com/owner/repository',
            snippet: 'Search or jump to... Sign in',
            metadata: { full_text: 'Search code, repositories, users, issues, pull requests...' },
            highlights: [{ id: 3, highlight: 'A repository note' }],
          }
          const repositoryRow = createLinkEl(repositoryLink)
          query('#curius-fragments').append(repositoryRow)
          click(repositoryRow)
          check(!query('#curius-preview-content').textContent.includes('Search'), 'repository loading never displays saved GitHub navigation')
          await waitFor(() => panel.getAttribute('aria-busy') === 'false')
          const card = query('.curius-preview-repository-card')
          check(card.getAttribute('src').startsWith('/api/curius?query=preview-image&id=105&image=0&v='), 'repository card uses the validated Curius image route')
          check(card.alt === 'owner/repository repository card', 'repository card has an accessible description')
          check(query('.curius-preview-article').textContent.includes('A useful README introduction.'), 'repository preview contains README text')
          check(!query('#curius-preview-content').textContent.includes('Search'), 'repository success excludes saved navigation text')
          check(query('.curius-preview-highlights').textContent.includes('A repository note'), 'repository previews preserve saved highlights')
          check(original.href === repositoryLink.link, 'repository original action follows the selected repository')
          const unavailableRepositoryRow = createLinkEl({ ...repositoryLink, id: 106, link: 'https://github.com/owner/unavailable' })
          query('#curius-fragments').append(unavailableRepositoryRow)
          click(unavailableRepositoryRow)
          await waitFor(() => panel.getAttribute('aria-busy') === 'false')
          check(!document.querySelector('.curius-preview-summary') && !query('#curius-preview-content').textContent.includes('Search'), 'repository failure never falls back to navigation chrome')
          check(!query('#curius-preview-retry').disabled, 'unavailable repository remains retryable')
          check(query('.curius-preview-highlights').textContent.includes('A repository note'), 'repository failure preserves saved highlights')
          click(first)
          await waitFor(() => panel.getAttribute('aria-busy') === 'false')
          check(query('.curius-preview-summary').title === 'Excerpt saved by Curius', 'snippet provenance is distinct and available on hover')
          check(query('.curius-preview-summary').textContent === firstLink.snippet, 'excerpt content appears without a status description')
          close.click()
          for (const modifier of ['ctrlKey', 'metaKey', 'altKey']) {
            checkNative(first, { [modifier]: true }, modifier + ' keeps native article navigation')
            check(panel.hidden, modifier + ' leaves the panel closed')
          }
          checkNative(first, { button: 1 }, 'middle clicks retain native article navigation')
          check(panel.hidden, 'middle clicks leave the panel closed')
          click(first, { shiftKey: true })
          check(panel.hidden, 'Shift-click opens the original without opening the panel')
          second.dispatchEvent(new KeyboardEvent('keydown', {
            key: 'Enter', shiftKey: true, bubbles: true, cancelable: true,
          }))
          check(panel.hidden, 'Shift-Enter opens the original without opening the panel')
          click(secondRow, { shiftKey: true })
          check(panel.hidden, 'Shift-click on a row also opens its original')
          first.dispatchEvent(new KeyboardEvent('keydown', {
            key: 'Enter', bubbles: true, cancelable: true,
          }))
          check(!panel.hidden && source.textContent === firstLink.title, 'Enter opens the focused article preview')
          check(panel.getAttribute('aria-modal') === String(matchMedia('(max-width: 800px)').matches), 'panel modality follows the viewport')
          if (matchMedia('(max-width: 800px)').matches) {
            const focusable = Array.from(panel.querySelectorAll('a[href], button:not([disabled]), summary, [tabindex="0"]'))
              .filter(element => element.getClientRects().length > 0)
            const last = focusable.at(-1)
            last.focus()
            last.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true }))
            check(document.activeElement === close, 'mobile Tab cycles into the first panel control')
          }
          close.click()
          for (const cleanup of cleanups) cleanup()
          toggle.click()
          check(panel.hidden && !document.body.classList.contains('curius-preview-open'), 'navigation cleanup closes and detaches the preview')
          setTimeout(() => {
            document.body.textContent = btoa(JSON.stringify({ passed }))
          }, 300)
        } catch (error) {
          document.body.textContent = btoa(JSON.stringify({ passed, error: String(error) }))
        }
        })()
      `,
      resolveDir: path.dirname(fileURLToPath(import.meta.url)),
      loader: 'ts',
    },
    bundle: true,
    write: false,
    format: 'iife',
    platform: 'browser',
    target: 'es2022',
  })
  const script = bundle.outputFiles[0].text
  const originalRequests: string[] = []
  const previewRequests: string[] = []
  const repositoryPreview: CuriusPreviewResponse = {
    status: 'ready',
    linkId: 105,
    title: 'owner/repository',
    sourceUrl: 'https://github.com/owner/repository',
    finalUrl: 'https://github.com/owner/repository',
    fetchedAt: 1,
    cached: false,
    readerHtml: `<a href="https://github.com/owner/repository"><img class="curius-preview-repository-card" src="${curiusPreviewImagePath(105, 0, 1)}" alt="owner/repository repository card"></a><p>A useful README introduction.</p>`,
  }
  const server = createServer((request, response) => {
    const requestUrl = new URL(request.url ?? '/', 'http://localhost')
    const requestPath = requestUrl.pathname
    if (requestPath === '/fixture.js') {
      response.setHeader('Content-Type', 'application/javascript')
      response.end(script)
    } else if (requestPath.startsWith('/original/')) {
      originalRequests.push(requestPath)
      response.setHeader('Content-Type', 'text/html')
      response.end('<!doctype html><title>Original source</title><p>Original source</p>')
    } else if (requestPath === '/api/curius') {
      if (requestUrl.searchParams.get('query') === 'preview-image') {
        response.writeHead(200, { 'Content-Type': 'image/png' })
        response.end(
          Buffer.from(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=',
            'base64',
          ),
        )
        return
      }
      previewRequests.push(requestUrl.searchParams.get('id') ?? '')
      if (requestUrl.searchParams.get('id') === '105') {
        response.writeHead(200, { 'Content-Type': 'application/json' })
        response.end(JSON.stringify(repositoryPreview))
        return
      }
      response.writeHead(503, { 'Content-Type': 'application/json' })
      response.end('{}')
    } else {
      response.setHeader('Content-Type', 'text/html')
      response.setHeader('Content-Security-Policy', "frame-src 'none'")
      response.end(`<!doctype html><html><body>
        <button id="curius-preview-toggle" aria-expanded="false">Open preview</button>
        <ul id="curius-fragments"></ul>
        <div id="highlight-modal"><ul id="highlight-modal-list"></ul></div>
        <button id="curius-preview-backdrop" hidden></button>
        <aside id="curius-preview" tabindex="-1" hidden>
          <button id="curius-preview-close">Close preview</button>
          <a id="curius-preview-source" target="_blank" rel="noopener noreferrer"></a>
          <a id="curius-preview-original" target="_blank" rel="noopener noreferrer" hidden>Open original</a>
          <button id="curius-preview-retry">Retry preview</button>
          <div id="curius-preview-content"></div>
        </aside>
        <script src="/fixture.js"></script>
      </body></html>`)
    }
  })
  const directory = await mkdtemp(path.join(tmpdir(), 'curius-interaction-'))
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  assert.ok(address && typeof address !== 'string')
  try {
    for (const width of [1200, 600]) {
      const result = await run(
        chrome,
        [
          '--headless',
          '--disable-gpu',
          '--no-sandbox',
          '--disable-extensions',
          '--disable-background-networking',
          '--disable-popup-blocking',
          '--virtual-time-budget=5000',
          `--window-size=${width},900`,
          `--user-data-dir=${path.join(directory, String(width))}`,
          '--dump-dom',
          `http://127.0.0.1:${address.port}/`,
        ],
        { timeout: 30_000, maxBuffer: 2 * 1024 * 1024 },
      )
      const encoded = result.stdout.match(/<body[^>]*>([A-Za-z\d+/=]+)<\/body>/)?.[1]
      assert.ok(encoded, result.stdout.slice(-1500) + result.stderr.slice(-1000))
      const output: { passed: string[]; error?: string } = JSON.parse(
        Buffer.from(encoded, 'base64').toString('utf8'),
      )
      assert.equal(output.error, undefined, `${width}px: ${output.error}`)
      assert.ok(output.passed.length >= 47, `${width}px passed ${output.passed.length} checks`)
    }
    assert.ok(
      originalRequests.includes('/original/first'),
      'Shift-click requests the first original',
    )
    assert.ok(
      originalRequests.includes('/original/second'),
      'Shift-Enter requests the second original',
    )
    assert.ok(previewRequests.includes('103'), 'saved article attempts to refresh its preview')
    assert.ok(!previewRequests.includes('104'), 'YouTube skips article extraction')
    assert.ok(previewRequests.includes('105'), 'repository preview requests the structured card')
  } finally {
    await new Promise<void>((resolve, reject) =>
      server.close(error => (error ? reject(error) : resolve())),
    )
    await rm(directory, { recursive: true, force: true })
  }
})
