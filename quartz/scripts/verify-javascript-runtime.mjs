import { build } from 'esbuild'
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { once } from 'node:events'
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises'
import http from 'node:http'
import path from 'node:path'
import { listenForOAuthCallback } from './oauth-callback.ts'

const root = path.resolve('docs/verification/code-scanning-remediation')
const artifacts = path.join(root, 'artifacts/runtime')
await mkdir(artifacts, { recursive: true })
const profile = await mkdtemp(path.join(artifacts, 'chrome-'))
const watcher = process.env.GARDEN_TEST_ORIGIN ?? 'http://127.0.0.1:8080'
const browser = process.env.ARENA_TEST_CHROME ?? '/Applications/Helium.app/Contents/MacOS/Helium'
await build({
  stdin: {
    contents: `
    export { createBrowserJavaScriptKernel } from './quartz/runtime/javascript/browser-kernel'
    export { notebookCellActions, notebookCellControls, notebookCellFrameOpen,
      notebookCellRuntimeOutput, notebookSourceEditor } from './quartz/util/notebook/cell-html'
    export { notebookRuntimeLocalSourceKey } from './quartz/runtime/notebook/types'`,
    resolveDir: process.cwd(),
    loader: 'ts',
  },
  outfile: path.join(artifacts, 'fixture-exports.js'),
  bundle: true,
  format: 'esm',
  platform: 'browser',
  target: 'es2022',
})
const delivered = []
let result
let resolveResult
const completed = new Promise(resolve => {
  resolveResult = resolve
})
const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url ?? '/', 'http://localhost')
    if (url.pathname.startsWith('/static/scripts/')) {
      const upstream = await fetch(watcher + url.pathname)
      const bytes = Buffer.from(await upstream.arrayBuffer())
      response.writeHead(upstream.status, { 'Content-Type': 'application/javascript' })
      response.end(bytes)
      delivered.push({
        url: watcher + url.pathname,
        status: upstream.status,
        sha256: createHash('sha256').update(bytes).digest('hex'),
      })
      return
    }
    if (url.pathname === '/fixture-exports.js' || url.pathname === '/runtime-browser.mjs') {
      response.setHeader('Content-Type', 'application/javascript')
      response.end(
        await readFile(
          url.pathname.endsWith('.mjs')
            ? path.resolve('quartz/runtime/javascript/boundary-fixture.mjs')
            : path.join(artifacts, 'fixture-exports.js'),
        ),
      )
    } else if (url.pathname === '/private/probe') {
      response.end(
        request.headers.cookie?.includes('runtimeFixture=only-test-data')
          ? 'fixture-cookie-present'
          : 'fixture-cookie-absent',
      )
    } else if (url.pathname === '/fixtures/sibling.txt') {
      response.end('sibling-fixture')
    } else if (url.pathname === '/result') {
      const chunks = []
      for await (const chunk of request) chunks.push(chunk)
      result = JSON.parse(Buffer.concat(chunks).toString())
      resolveResult(result)
      response.end('saved')
    } else if (url.pathname === '/fixtures/notebook.html') {
      response.setHeader('Content-Type', 'text/html')
      response.setHeader(
        'Set-Cookie',
        'runtimeFixture=only-test-data; HttpOnly; SameSite=Strict; Path=/',
      )
      response.end(
        '<!doctype html><html><body><script type="module" src="/runtime-browser.mjs"></script></body></html>',
      )
    } else {
      response.writeHead(404).end('not found')
    }
  } catch (error) {
    response.writeHead(500).end(String(error))
  }
})
server.listen(0, '127.0.0.1')
await once(server, 'listening')
const address = server.address()
assert(address && typeof address !== 'string')
const url = `http://127.0.0.1:${address.port}/fixtures/notebook.html?code=URL-injection-must-not-run`
const args = [
  '--headless',
  '--disable-gpu',
  '--disable-extensions',
  '--disable-background-networking',
  '--remote-debugging-port=0',
  `--user-data-dir=${profile}`,
  'about:blank',
]
const chrome = spawn(browser, args, { stdio: ['ignore', 'ignore', 'pipe'] })
let browserLog = ''
chrome.stderr.on('data', chunk => {
  browserLog += chunk.toString()
})
let socket
let deadline
try {
  let debugPort
  for (let attempt = 0; attempt < 100; attempt++) {
    try {
      debugPort = (await readFile(path.join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]
      break
    } catch {
      await new Promise(resolve => setTimeout(resolve, 100))
    }
  }
  assert(debugPort, 'Chromium did not expose its test debugging endpoint')
  const page = await (
    await fetch(`http://127.0.0.1:${debugPort}/json/new?about:blank`, { method: 'PUT' })
  ).json()
  socket = new WebSocket(page.webSocketDebuggerUrl)
  await once(socket, 'open')
  let sequence = 0
  const pending = new Map()
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data)
    const request = pending.get(message.id)
    if (!request) return
    pending.delete(message.id)
    if (message.error) request.reject(new Error(JSON.stringify(message.error)))
    else request.resolve(message.result)
  })
  const send = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const id = ++sequence
      pending.set(id, { resolve, reject })
      socket.send(JSON.stringify({ id, method, params }))
    })
  await send('Page.navigate', { url })
  await Promise.race([
    completed,
    new Promise((_, reject) => {
      deadline = setTimeout(
        () => reject(new Error('Runtime browser did not finish within 35 seconds')),
        35_000,
      )
    }),
  ])
  const dom = await send('Runtime.evaluate', {
    expression: 'document.documentElement.outerHTML',
    returnByValue: true,
  })
  await writeFile(path.join(artifacts, 'browser.html'), dom.result.value)
  const screenshot = await send('Page.captureScreenshot', { format: 'png' })
  await writeFile(path.join(artifacts, 'browser.png'), Buffer.from(screenshot.data, 'base64'))
  const oauth = []
  for (const provider of ['Strava', 'Oura']) {
    const callback = listenForOAuthCallback({ provider, redirectUri: 'http://localhost:0/' })
    const grant = callback.grant.catch(error => error.message)
    await once(callback.server, 'listening')
    const callbackAddress = callback.server.address()
    assert(callbackAddress && typeof callbackAddress !== 'string')
    const query = new URLSearchParams({ error: '<img src=x onerror=alert(1)>' })
    const callbackUrl = `http://127.0.0.1:${callbackAddress.port}/?${query}`
    await send('Page.navigate', { url: callbackUrl })
    let documentResult
    for (let attempt = 0; attempt < 100; attempt++) {
      documentResult = await send('Runtime.evaluate', {
        expression: `JSON.stringify({
        url: location.href, contentType: document.contentType, body: document.body?.textContent,
        elements: document.querySelectorAll('img, script, iframe').length })`,
        returnByValue: true,
      })
      const value = JSON.parse(documentResult.result.value)
      if (value.url === callbackUrl && value.body) {
        oauth.push({ provider, ...value, error: await grant })
        break
      }
      await new Promise(resolve => setTimeout(resolve, 20))
    }
    callback.server.close()
  }
  assert.equal(oauth.length, 2)
  for (const response of oauth) {
    assert.equal(response.contentType, 'text/plain')
    assert.equal(response.elements, 0)
    assert.equal(response.body.includes('<img'), false)
    assert.equal(response.error, '<img src=x onerror=alert(1)>')
  }
  await writeFile(path.join(artifacts, 'oauth-browser.json'), JSON.stringify(oauth, null, 2))
  await writeFile(path.join(artifacts, 'browser.log'), browserLog)
  await writeFile(
    path.join(artifacts, 'result.json'),
    JSON.stringify(
      {
        result,
        delivered,
        command: 'node quartz/scripts/verify-javascript-runtime.mjs',
        browser,
        args,
      },
      null,
      2,
    ),
  )
  assert(result?.passed, JSON.stringify(result ?? { error: 'No browser result' }, null, 2))
  console.log(JSON.stringify({ passed: true, checks: result.checks.length, artifacts }, null, 2))
} finally {
  clearTimeout(deadline)
  socket?.close()
  chrome.kill('SIGTERM')
  server.closeAllConnections()
  server.close()
}
