import { connect } from '@cloudflare/puppeteer'
import { build } from 'esbuild'
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { access, mkdtemp, rm } from 'node:fs/promises'
import { createServer, ServerResponse } from 'node:http'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import type { Link, Trail } from '../../types/curius'

const trail: Trail = {
  id: 1,
  trailName: 'Fixture trail',
  ownerId: 1,
  description: 'Saved articles',
  colorHex: '',
  emojiUnicode: '',
  flipped: null,
  hash: 'fixture-trail',
  slug: 'fixture-trail',
  createdDate: '2026-09-01T00:00:00Z',
  users: [],
}

function savedLink(id: number, trails: Trail[] = []): Link {
  return {
    id,
    link: `https://example.org/article/${id}`,
    title: `Saved article ${id}`,
    favorite: false,
    snippet: `Saved text ${id}`,
    toRead: false,
    createdBy: 1,
    metadata: { full_text: '', author: '', page_type: '' },
    lastCrawled: null,
    createdDate: '2026-09-01T00:00:00Z',
    modifiedDate: '2026-09-01T00:00:00Z',
    trails,
    comments: [],
    mentions: [],
    topics: [],
    highlights: [],
  }
}

test('Curius loads sections independently and appends pages within the feed scrollport', async t => {
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
  if (!chrome) return t.skip('Chromium is required for the Curius feed fixture.')

  const bundle = await build({
    stdin: {
      contents: `
        import './curius.inline'
        import './curius-friends.inline'
        const cleanups = []
        const messages = []
        const passed = []
        window.addCleanup = callback => cleanups.push(callback)
        document.addEventListener('toast', event => messages.push(event.detail.message))
        const check = (condition, message) => {
          if (!condition) throw new Error(message)
          passed.push(message)
        }
        const query = selector => {
          const element = document.querySelector(selector)
          if (!element) throw new Error('Missing fixture element ' + selector)
          return element
        }
        const waitFor = (predicate, label) => new Promise((resolve, reject) => {
          const observer = new MutationObserver(checkState)
          const timeout = setTimeout(() => {
            dispose()
            reject(new Error('Timed out: ' + label))
          }, 3000)
          function dispose() {
            clearTimeout(timeout)
            observer.disconnect()
            document.removeEventListener('toast', checkState)
          }
          function checkState() {
            if (!predicate()) return
            dispose()
            resolve()
          }
          observer.observe(document.documentElement, { childList: true, subtree: true, attributes: true })
          document.addEventListener('toast', checkState)
          checkState()
        })
        const wait = delay => new Promise(resolve => setTimeout(resolve, delay))
        const release = section => fetch('/release?section=' + section)
        const requests = () => fetch('/requests').then(response => response.json())
        const feed = query('#curius-fragments')
        const rows = () => feed.querySelectorAll(':scope > .curius-item')
        const previous = query('#curius-prev')
        const next = query('#curius-next')
        const profile = query('.curius-profile')
        const friends = query('.curius-friends')
        const trails = query('.curius-trail')
        const scenario = new URL(location.href).searchParams.get('scenario')
        document.dispatchEvent(new CustomEvent('nav', { detail: { url: 'curius' } }))

        void (async () => {
          try {
            check(friends.getClientRects().length > 0 && trails.getClientRects().length > 0, 'sidebar shells exist during initial loading')
            check(profile.hidden, 'profile waits for both sidebar sections')
            await waitFor(() => feed.getAttribute('aria-busy') !== 'true', 'initial links')
            check(messages[0] === 'Récupération des liens curius…', 'initial loading uses the toast event')
            if (scenario === 'filtered') {
              check(rows().length === 0 && !feed.querySelector('.curius-list-status'), 'a filtered initial page avoids a premature empty-state message')
              check(!next.disabled, 'a filtered initial page can advance to later raw pages')
              await waitFor(() => feed.getAttribute('aria-busy') === 'true', 'automatic filtered-page continuation')
              check(!feed.querySelector('.curius-list-status'), 'filtered-page continuation keeps the feed free of transient empty text')
              await release('page')
              await waitFor(() => next.disabled && feed.querySelector('.curius-list-status'), 'filtered feed actual end')
              check(query('.curius-list-status').getAttribute('role') === 'status', 'an entirely filtered feed announces its actual empty result')
              check(rows().length === 0 && !feed.querySelector('.curius-load-more'), 'an entirely filtered feed stops at the actual end')
              check((await requests()).pages['1'] === 1, 'an entirely filtered feed advances the raw page once')
              check(!profile.hidden, 'filtered links do not delay settled sidebar sections')
            } else if (scenario === 'empty' || scenario === 'error') {
              await waitFor(() => !profile.hidden, 'empty sidebar settlement')
              check(friends.getAttribute('aria-busy') === 'false' && trails.getAttribute('aria-busy') === 'false', 'empty or failed sections settle independently')
              check(!query('#curius-friends-status').hidden && !query('#curius-trails-status').hidden, 'empty or failed sidebar messages stay visible')
              check(query('.curius-list-status').getAttribute('role') === 'status', 'empty or failed feed has an accessible status')
              check(rows().length === 0 && previous.disabled && next.disabled, 'empty or failed feed disables navigation')
              check(!feed.querySelector('.curius-load-more'), 'empty or failed feed has no loading sentinel')
              check(query('#see-more-friends').hidden, 'empty or failed friends hide expansion')
              if (scenario === 'error') {
                check(query('.curius-list-status').textContent.includes('Impossible'), 'failed initial links show the error state')
                check(messages.some(message => message.includes('Impossible')), 'failed initial links dispatch an error toast')
              } else {
                check(query('.curius-list-status').textContent.includes('Aucun'), 'empty links show the empty state')
              }
            } else {
              check(rows().length === 20, 'initial feed excludes trail members and duplicate IDs')
              check(previous.disabled && !next.disabled, 'initial navigation exposes only the next page')
              if (scenario === 'cleanup') {
                next.click()
                let counts = await requests()
                while (!counts.pages['1']) {
                  await wait(20)
                  counts = await requests()
                }
                const messageCount = messages.length
                for (const cleanup of cleanups.splice(0)) cleanup()
                document.dispatchEvent(new CustomEvent('nav', { detail: { url: 'other' } }))
                await release('page')
                await wait(100)
                check(rows().length === 20, 'navigation cleanup prevents an in-flight page from rendering')
                check(window.curiusState === undefined, 'navigation cleanup clears page state')
                check(messages.length === messageCount, 'an aborted page emits no completion or error toast')
                next.click()
                feed.scrollTop = feed.scrollHeight
                await wait(100)
                counts = await requests()
                check(counts.pages['1'] === 1, 'cleanup removes button and scroll loading listeners')
                check(counts.abortedPages === 1, 'cleanup aborts the in-flight HTTP request')
              } else {
                check(friends.getAttribute('aria-busy') === 'true' && trails.getAttribute('aria-busy') === 'true', 'feed renders before pending sidebar requests')
                await release('trails')
                await waitFor(() => trails.getAttribute('aria-busy') === 'false', 'trails')
                check(profile.hidden, 'one completed sidebar section keeps the profile hidden')
                const moreTrail = query('.trail-ul > li.see-more')
                check(moreTrail === moreTrail.parentElement.lastElementChild, 'trail continuation is the final tree list item')
                check(moreTrail.querySelector('a').textContent === 'Voir de plus →', 'trail continuation retains its native source link')
                await release('following')
                await waitFor(() => !profile.hidden, 'friends and profile')
                const moreFriends = query('#see-more-friends')
                const friendList = query('#friends-list')
                check(friendList.querySelectorAll('.active').length === 4, 'friends start with four visible entries')
                check(!friendList.querySelector('strong, b, [style*="font-weight"]'), 'friend names contain no bold rendering')
                check(!friendList.classList.contains('overflow'), 'friends avoid the global trailing spacer utility')
                moreFriends.click()
                check(friendList.querySelectorAll('.active').length === 6 && moreFriends.getAttribute('aria-expanded') === 'true', 'friends expand through the native control')
                friendList.scrollTop = friendList.scrollHeight
                moreFriends.click()
                check(friendList.scrollTop === 0 && moreFriends.getAttribute('aria-expanded') === 'false', 'collapsing friends returns their scroll position to the top')
                const initialRequests = await requests()
                check(initialRequests.pendingSearch, 'search indexing does not block the feed or sidebar')
                await release('searchLinks')

                const firstRow = rows()[0]
                feed.scrollTop = feed.scrollHeight
                const originalScrollTop = feed.scrollTop
                await waitFor(() => rows().length === 40, 'scroll-triggered appended page')
                check(firstRow === rows()[0] && firstRow.isConnected, 'appending keeps the existing feed rows')
                check(Math.abs(feed.scrollTop - originalScrollTop) <= 1, 'automatic append preserves feed scroll offset')
                check(window.scrollY === 0, 'feed scrolling leaves the document at its current position')
                check(new Set(Array.from(rows(), row => row.id)).size === 40, 'appended pages deduplicate saved link IDs')
                check(feed.lastElementChild.classList.contains('curius-load-more'), 'the near-end sentinel stays last')
                check(messages.includes('Page 3 chargée.'), 'append completion reports through the toast event')
                next.click()
                await waitFor(() => !previous.disabled, 'next loaded boundary')
                check(window.curiusState.currentPage === 2, 'Next moves to the next loaded raw-page boundary')
                previous.click()
                await waitFor(() => feed.scrollTop === 0, 'previous boundary')
                check(previous.disabled, 'Previous moves back to the first loaded page')
                next.click()
                await waitFor(() => window.curiusState.currentPage === 2, 'return to loaded boundary')
                const loadedRequests = await requests()
                check(loadedRequests.pages['1'] === 1 && loadedRequests.pages['2'] === 1, 'filtered pages advance the cursor once without refetching loaded boundaries')

                feed.scrollTop = feed.scrollHeight
                await waitFor(() => messages.some(message => message.includes('Réessayez')), 'failed append')
                await wait(200)
                check((await requests()).pages['3'] === 1, 'a failed automatic append pauses repeated requests')
                check(!next.disabled && rows().length === 40, 'failed append preserves rows and offers a manual retry')
                next.click()
                await waitFor(() => next.disabled && !feed.querySelector('.curius-load-more'), 'actual end of feed')
                check(rows().length === 40 && !previous.disabled, 'actual end retains loaded pages and backward navigation')
                check(messages.includes("Pas d'autres liens pour le moment."), 'actual end reports through the toast event')
                const finalRequests = await requests()
                check(finalRequests.pages['3'] === 2, 'Next retries only the failed raw page')
              }
            }
            for (const cleanup of cleanups.splice(0)) cleanup()
            document.body.textContent = btoa(JSON.stringify({ passed }))
            document.body.dataset.fixtureDone = 'true'
          } catch (error) {
            const state = { requests: await requests(), messages, rows: rows().length,
              scrollTop: feed.scrollTop, scrollHeight: feed.scrollHeight, clientHeight: feed.clientHeight }
            document.body.textContent = btoa(JSON.stringify({ passed, error: String(error) + ' ' + JSON.stringify(state) }))
            document.body.dataset.fixtureDone = 'true'
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

  const firstPage = Array.from({ length: 20 }, (_, index) => savedLink(index + 1))
  const secondPage = Array.from({ length: 20 }, (_, index) => savedLink(index + 21))
  const following = Array.from({ length: 6 }, (_, index) => ({
    user: {
      id: index + 1,
      firstName: `Friend ${index + 1}`,
      lastName: 'Fixture',
      userLink: `friend-${index + 1}`,
      lastOnline: '2026-09-01T00:00:00Z',
    },
    link: savedLink(index + 1),
  }))
  let scenario = ''
  let pages: Record<string, number> = {}
  let abortedPages = 0
  let pending = new Map<string, ServerResponse>()
  let released = new Set<string>()
  const send = (response: ServerResponse, data: unknown, status = 200) => {
    response.writeHead(status, { 'Content-Type': 'application/json' })
    response.end(JSON.stringify(data))
  }
  const sectionResponse = (section: string, response: ServerResponse) => {
    if (section === 'following') send(response, { following })
    else if (section === 'page')
      send(response, {
        links: scenario === 'filtered' ? [] : secondPage,
        page: 1,
        hasMore: scenario !== 'filtered',
      })
    else send(response, section === 'trails' ? { trails: [] } : { links: [] })
  }

  const server = createServer((request, response) => {
    const url = new URL(request.url ?? '/', 'http://localhost')
    if (url.pathname === '/fixture.js') {
      response.setHeader('Content-Type', 'application/javascript')
      response.end(bundle.outputFiles[0].text)
    } else if (url.pathname === '/release') {
      const section = url.searchParams.get('section') ?? ''
      released.add(section)
      const waiting = pending.get(section)
      if (waiting) {
        pending.delete(section)
        sectionResponse(section, waiting)
      }
      send(response, {})
    } else if (url.pathname === '/requests') {
      send(response, { pages, abortedPages, pendingSearch: pending.has('searchLinks') })
    } else if (url.pathname === '/api/curius') {
      const section = url.searchParams.get('query') ?? ''
      if (section === 'links') {
        const page = url.searchParams.get('page') ?? '0'
        pages[page] = (pages[page] ?? 0) + 1
        if (scenario === 'error') send(response, {}, 503)
        else if (scenario === 'empty') send(response, { links: [], hasMore: false })
        else if (page === '0' && scenario === 'filtered') {
          send(response, { links: [savedLink(100, [trail])], page: 0, hasMore: true })
        } else if (page === '0') {
          send(response, {
            links: [...firstPage, firstPage[0], savedLink(100, [trail])],
            page: 0,
            hasMore: true,
          })
        } else if (scenario === 'cleanup' || scenario === 'filtered') {
          pending.set('page', response)
          response.on('close', () => {
            if (!response.writableEnded) abortedPages += 1
          })
        } else if (page === '1') {
          send(response, { links: [savedLink(101, [trail]), firstPage[0]], page: 1, hasMore: true })
        } else if (page === '2') {
          send(response, {
            links: [firstPage[0], ...secondPage, secondPage[0]],
            page: 2,
            hasMore: true,
          })
        } else if (page === '3' && pages[page] === 1) send(response, {}, 503)
        else send(response, { links: [], page: Number(page), hasMore: false })
      } else if (scenario === 'scroll' && !released.has(section)) {
        pending.set(section, response)
      } else if (scenario === 'error' && section !== 'searchLinks') send(response, {}, 503)
      else if (scenario === 'empty' || scenario === 'cleanup' || scenario === 'filtered') {
        send(
          response,
          section === 'following'
            ? { following: [] }
            : section === 'trails'
              ? { trails: [] }
              : { links: [] },
        )
      } else sectionResponse(section, response)
    } else {
      response.setHeader('Content-Type', 'text/html')
      response.end(`<!doctype html><html><head><style>
        #curius-fragments { height: 240px; width: 600px; overflow-y: auto; margin: 0; padding: 0; }
        #curius-fragments > .curius-item { box-sizing: border-box; height: 60px; overflow: hidden; }
        .curius-load-more { height: 1px; padding: 0; margin: 0; list-style: none; }
        #friends-list { max-height: 120px; overflow-y: auto; }
        .friend-li { height: 40px; }
        .friend-li:not(.active) { display: none; }
      </style></head><body>
        <input id="curius-bar" type="search"><div id="curius-search-container"></div>
        <nav id="curius-pagination"><button id="curius-prev" disabled>Previous</button><button id="curius-next" disabled>Next</button></nav>
        <ul id="curius-fragments" aria-busy="true"></ul>
        <div class="curius-friends" aria-busy="true">
          <p id="curius-friends-status" role="status">Chargement des amis…</p>
          <ul id="friends-list" class="section-ul"></ul>
          <button id="see-more-friends" aria-expanded="false" hidden><span id="more">de plus</span><svg></svg></button>
        </div>
        <div class="curius-trail" aria-busy="true" data-limits="4" data-num-trails="3" data-locale="en-US">
          <p id="curius-trails-status" role="status">Chargement des sentiers…</p><ul id="trail-list"></ul>
        </div>
        <a class="curius-profile" hidden href="https://curius.app/aaron-pham">curius.app/aaron-pham</a>
        <script src="/fixture.js"></script>
      </body></html>`)
    }
  })
  const directory = await mkdtemp(path.join(tmpdir(), 'curius-feed-'))
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  assert.ok(address && typeof address !== 'string')
  const browserProcess = spawn(
    chrome,
    [
      '--headless',
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
      defaultViewport: { width: 1200, height: 900 },
      protocolTimeout: 10_000,
    })
    for (const currentScenario of ['scroll', 'cleanup', 'filtered', 'empty', 'error']) {
      scenario = currentScenario
      pages = {}
      abortedPages = 0
      pending = new Map()
      released = new Set()
      const page = await browser.newPage()
      await page.goto(`http://127.0.0.1:${address.port}/?scenario=${currentScenario}`, {
        waitUntil: 'domcontentloaded',
      })
      await page.waitForFunction(() => document.body.dataset.fixtureDone === 'true', {
        timeout: 8000,
      })
      const encoded = await page.evaluate(() => document.body.textContent)
      assert.ok(encoded)
      const output: { passed: string[]; error?: string } = JSON.parse(
        Buffer.from(encoded, 'base64').toString('utf8'),
      )
      assert.equal(
        output.error,
        undefined,
        `${currentScenario}: ${output.error}; passed: ${output.passed.join(', ')}`,
      )
      assert.ok(
        output.passed.length >= 10,
        `${currentScenario} passed ${output.passed.length} checks`,
      )
      assert.equal(pages['0'], 1, `${currentScenario}: one initial links request`)
      await page.close()
    }
  } finally {
    await browser?.disconnect()
    browserProcess.kill('SIGKILL')
    for (const response of pending.values()) response.destroy()
    await new Promise<void>((resolve, reject) =>
      server.close(error => (error ? reject(error) : resolve())),
    )
    await rm(directory, { recursive: true, force: true })
  }
})
