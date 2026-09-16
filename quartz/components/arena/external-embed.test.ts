import { build } from 'esbuild'
import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { access, mkdtemp, rm } from 'node:fs/promises'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

test('item previews mount on demand, follow redirects, retry failures and cancel on close', async t => {
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
      continue
    }
  }
  if (!chrome) return t.skip('Set ARENA_TEST_CHROME to run the Chromium preview fixture.')

  const directory = await mkdtemp(path.join(tmpdir(), 'arena-preview-'))
  const requests: string[] = []
  const attempts = new Map<string, number>()
  const report = Promise.withResolvers<string>()
  const bundle = await build({
    entryPoints: [fileURLToPath(new URL('./external-embed.ts', import.meta.url))],
    bundle: true,
    write: false,
    format: 'iife',
    globalName: 'ArenaPreview',
    platform: 'browser',
  })
  const server = createServer((request, response) => {
    const url = new URL(request.url ?? '/', `http://${request.headers.host}`)
    requests.push(url.pathname)
    if (url.pathname === '/result') {
      let body = ''
      request.setEncoding('utf8')
      request.on('data', chunk => {
        body += chunk
      })
      request.on('end', () => {
        response.end('ok')
        report.resolve(body)
      })
    } else if (url.pathname === '/api/arena-embed/capability') {
      const source = new URL(url.searchParams.get('url') ?? url.origin)
      const attempt = (attempts.get(source.pathname) ?? 0) + 1
      attempts.set(source.pathname, attempt)
      response.setHeader('Content-Type', 'application/json')
      if (source.pathname === '/retry' && attempt === 1) {
        response.writeHead(503).end('{}')
      } else {
        response.end(JSON.stringify({ mode: 'iframe', finalUrl: `${url.origin}/document` }))
      }
    } else if (url.pathname === '/document' || url.pathname === '/api/arena-embed/html') {
      response.setHeader('Content-Type', 'text/html')
      response.end(
        '<!doctype html><title>Saved article</title><p>The complete article is visible.</p>',
      )
    } else if (url.pathname === '/api/arena-embed/capture') {
      response.setHeader('Content-Type', 'image/svg+xml')
      response.end(
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="100" height="100" fill="black"/></svg>',
      )
    } else {
      response.setHeader('Content-Type', 'text/html')
      response.end(`<!doctype html><body><script>${bundle.outputFiles[0].text}
        const controller = ArenaPreview.createArenaExternalEmbeds();
        const root = document.createElement('main');
        document.body.append(root);
        function makeHost(source, mode = 'auto') {
          const host = document.createElement('div');
          host.className = 'arena-modal-external-host';
          host.dataset.arenaUrl = location.origin + source;
          host.dataset.arenaEmbedMode = mode;
          host.dataset.arenaTitle = 'Saved article';
          root.append(host);
          return host;
        }
        function waitFor(host, status) {
          return new Promise((resolve, reject) => {
            if (host.dataset.arenaEmbedStatus === status) return resolve();
            const timeout = setTimeout(() => reject(new Error('Timed out: ' + host.dataset.arenaUrl + ' ' + status + ' actual=' + host.dataset.arenaEmbedStatus)), 8000);
            const observer = new MutationObserver(() => {
              if (host.dataset.arenaEmbedStatus !== status) return;
              clearTimeout(timeout);
              observer.disconnect();
              resolve();
            });
            observer.observe(host, { attributes: true, childList: true, subtree: true });
          });
        }
        function close() { controller.cleanup(); root.replaceChildren(); }
        (async () => {
          const result = {};
          const first = makeHost('/original');
          result.framesBeforeOpen = root.querySelectorAll('iframe').length;
          controller.mount(root);
          controller.mount(root);
          await waitFor(first, 'loaded');
          const frame = first.querySelector('iframe');
          result.direct = {
            count: root.querySelectorAll('iframe').length,
            src: new URL(frame.src).pathname,
            loading: frame.loading,
            body: frame.contentDocument.body.textContent,
            title: frame.title,
            busy: first.hasAttribute('aria-busy'),
            placeholder: !!first.querySelector('.arena-embed-loading'),
          };
          close();
          result.framesAfterClose = root.querySelectorAll('iframe').length;
          const retry = makeHost('/retry');
          controller.mount(root);
          await waitFor(retry, 'error');
          result.failed = { frames: retry.querySelectorAll('iframe').length, original: retry.querySelector('a').href };
          retry.querySelector('button').click();
          await waitFor(retry, 'loaded');
          result.retried = !!retry.querySelector('iframe');
          close();
          const fetched = makeHost('/fetched', 'fetch');
          controller.mount(root);
          await waitFor(fetched, 'loaded');
          result.fetched = { sandbox: fetched.querySelector('iframe').getAttribute('sandbox'), src: new URL(fetched.querySelector('iframe').src).pathname };
          close();
          const captured = makeHost('/captured', 'capture');
          controller.mount(root);
          await waitFor(captured, 'loaded');
          result.captured = captured.querySelector('img').naturalWidth;
          close();
          const cancelled = makeHost('/cancelled');
          controller.mount(root);
          close();
          const reopened = makeHost('/original');
          controller.mount(root);
          await waitFor(reopened, 'loaded');
          result.reopened = reopened.querySelector('iframe').contentDocument.body.textContent;
          result.cancelled = cancelled.querySelectorAll('iframe').length;
          close();
          await fetch('/result', { method: 'POST', body: JSON.stringify(result) });
        })().catch(error => fetch('/result', { method: 'POST', body: JSON.stringify({error: error.message}) }));
      </script></body>`)
    }
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  assert.ok(address && typeof address !== 'string')
  const browser = execFile(
    chrome,
    [
      '--headless',
      '--disable-gpu',
      '--disable-extensions',
      '--disable-background-networking',
      `--user-data-dir=${path.join(directory, 'chrome')}`,
      `http://127.0.0.1:${address.port}/`,
    ],
    { maxBuffer: 2 * 1024 * 1024 },
    error => {
      if (error && !browser.killed) report.reject(error)
    },
  )
  const timeout = setTimeout(() => report.reject(new Error('Preview fixture timed out')), 30_000)
  try {
    const output: {
      error?: string
      framesBeforeOpen: number
      framesAfterClose: number
      direct: {
        count: number
        src: string
        loading: string
        body: string
        title: string
        busy: boolean
        placeholder: boolean
      }
      failed: { frames: number; original: string }
      retried: boolean
      fetched: { sandbox: string; src: string }
      captured: number
      cancelled: number
      reopened: string
    } = JSON.parse(await report.promise)
    assert.equal(output.error, undefined, JSON.stringify({ output, requests }))
    assert.equal(output.framesBeforeOpen, 0)
    assert.deepEqual(output.direct, {
      count: 1,
      src: '/document',
      loading: 'eager',
      body: 'The complete article is visible.',
      title: 'Embedded block: Saved article',
      busy: false,
      placeholder: false,
    })
    assert.equal(output.framesAfterClose, 0)
    assert.equal(output.failed.frames, 0)
    assert.equal(new URL(output.failed.original).pathname, '/retry')
    assert.equal(output.retried, true)
    assert.equal(attempts.get('/retry'), 2)
    assert.equal(attempts.get('/original'), 1)
    assert.deepEqual(output.fetched, { sandbox: '', src: '/api/arena-embed/html' })
    assert.equal(output.captured, 100)
    assert.equal(output.cancelled, 0)
    assert.equal(output.reopened, 'The complete article is visible.')
    assert.ok(requests.includes('/api/arena-embed/capture'))
  } finally {
    clearTimeout(timeout)
    if (browser.exitCode === null && browser.signalCode === null) {
      await new Promise<void>(resolve => {
        browser.once('close', () => resolve())
        browser.kill()
      })
    }
    server.closeAllConnections()
    await new Promise<void>(resolve => server.close(() => resolve()))
    await rm(directory, { recursive: true, force: true })
  }
})
