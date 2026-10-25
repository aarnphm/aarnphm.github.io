import { spawn } from 'node:child_process'
import { readFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import path from 'node:path'
import { createInterface } from 'node:readline'
import { fileURLToPath } from 'node:url'
import { parseArgs } from 'node:util'
import { WebSocket, WebSocketServer } from 'ws'
import { isRecord } from '../../quartz/util/type-guards'
import { enqueuePdfs } from './queue-api'

const directory = path.dirname(fileURLToPath(import.meta.url))
const repository = path.resolve(directory, '../..')
const { values } = parseArgs({
  options: {
    root: { type: 'string', default: repository },
    port: { type: 'string', default: '8091' },
    interval: { type: 'string', default: '3' },
  },
})
const port = Number(values.port)
const interval = Number(values.interval)
if (
  !Number.isInteger(port) ||
  port < 0 ||
  port > 65535 ||
  !Number.isFinite(interval) ||
  interval < 0.1
)
  throw new Error('Use a valid local port and an interval of at least 0.1 seconds')
const root = path.resolve(values.root)
let snapshot: string | undefined
let fault: string | undefined
let stopped = false
const sockets = new WebSocketServer({ noServer: true, maxPayload: 1024 })
const responsive = new WeakSet<WebSocket>()
const send = (payload: string) => {
  for (const client of sockets.clients) {
    if (client.readyState !== WebSocket.OPEN) continue
    if (client.bufferedAmount > 2 * 1024 * 1024) client.terminate()
    else client.send(payload)
  }
}
const assets = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/index.html', ['index.html', 'text/html; charset=utf-8']],
  ['/app.js', ['app.js', 'text/javascript; charset=utf-8']],
  ['/style.css', ['style.css', 'text/css; charset=utf-8']],
  ['/title.ttf', ['../../quartz/static/SpaceGrotesk-VariableFont_wght.ttf', 'font/ttf']],
])
const server = createServer(async (request, response) => {
  const url = new URL(request.url ?? '/', 'http://127.0.0.1')
  if (url.pathname === '/api/queue' && request.method === 'POST') {
    const address = server.address()
    if (!address || typeof address === 'string') {
      response.writeHead(503).end('server starting')
      return
    }
    await enqueuePdfs(request, response, root, repository, address.port)
    return
  }
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    response.writeHead(405, { Allow: 'GET, HEAD' }).end('unsupported method')
    return
  }
  response.setHeader('Cache-Control', 'no-store')
  response.setHeader('X-Content-Type-Options', 'nosniff')
  response.setHeader(
    'Content-Security-Policy',
    "default-src 'self'; connect-src 'self' ws://127.0.0.1:* ws://localhost:*; style-src 'self'; script-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'none'",
  )
  if (url.pathname === '/api/status') {
    response.writeHead(snapshot ? 200 : 503, { 'Content-Type': 'application/json' })
    response.end(
      request.method === 'HEAD'
        ? undefined
        : (snapshot ?? JSON.stringify({ type: 'error', message: fault ?? 'Reading the queue' })),
    )
    return
  }
  const asset = assets.get(url.pathname)
  if (!asset) {
    response.writeHead(404).end('not found')
    return
  }
  try {
    const bytes = await readFile(path.resolve(directory, asset[0]))
    response.writeHead(200, { 'Content-Type': asset[1] })
    response.end(request.method === 'HEAD' ? undefined : bytes)
  } catch (error) {
    response.writeHead(500).end('could not read harness asset')
    console.error(error)
  }
})

server.on('upgrade', (request, socket, head) => {
  const address = server.address()
  if (!address || typeof address === 'string') return socket.destroy()
  const hosts = new Set([`127.0.0.1:${address.port}`, `localhost:${address.port}`])
  const origin = request.headers.origin
  if (
    !hosts.has(request.headers.host ?? '') ||
    (origin !== undefined && ![...hosts].some(host => origin === `http://${host}`))
  ) {
    socket.end('HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n')
    return
  }
  if (request.url !== '/events') {
    socket.end('HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n')
    return
  }
  sockets.handleUpgrade(request, socket, head, client => sockets.emit('connection', client))
})
sockets.on('connection', client => {
  responsive.add(client)
  client.on('pong', () => responsive.add(client))
  client.on('error', () => client.terminate())
  if (snapshot) client.send(snapshot)
  if (fault) client.send(JSON.stringify({ type: 'error', message: fault }))
})

const observer = spawn(
  path.join(repository, '.venv/bin/python'),
  ['-u', path.join(directory, 'snapshot.py'), '--root', root, '--interval', String(interval)],
  { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] },
)
const lines = createInterface({ input: observer.stdout })
lines.on('line', line => {
  let frame: unknown
  try {
    frame = JSON.parse(line)
  } catch {
    return
  }
  if (!isRecord(frame)) return
  if (frame.type === 'snapshot') {
    snapshot = line
    fault = undefined
  } else if (frame.type === 'error') {
    fault = typeof frame.message === 'string' ? frame.message : 'Observer failed'
  } else return
  send(line)
})
observer.stderr.on('data', (bytes: Buffer) => process.stderr.write(bytes))
const failed = (message: string) => {
  if (stopped) return
  fault = message
  send(JSON.stringify({ type: 'error', message }))
  console.error(message)
}
observer.on('error', error => failed(`Observer could not start: ${error.message}`))
observer.on('exit', (code, signal) =>
  failed(`Observer stopped (${signal ?? code}); restart the harness`),
)
const heartbeat = setInterval(() => {
  for (const client of sockets.clients) {
    if (!responsive.has(client)) client.terminate()
    else {
      responsive.delete(client)
      client.ping()
    }
  }
}, 15_000)
const stop = () => {
  if (stopped) return
  stopped = true
  clearInterval(heartbeat)
  observer.kill('SIGTERM')
  lines.close()
  for (const client of sockets.clients) client.close(1001, 'harness stopped')
  sockets.close()
  server.close()
  const force = setTimeout(() => {
    for (const client of sockets.clients) client.terminate()
    observer.kill('SIGKILL')
  }, 1500)
  force.unref()
}
process.on('SIGTERM', stop)
process.on('SIGINT', stop)
server.on('error', error => {
  console.error(error)
  stop()
  process.exitCode = 1
})
server.listen(port, '127.0.0.1', () => {
  const address = server.address()
  if (address && typeof address !== 'string')
    console.log(`embedding harness · http://127.0.0.1:${address.port}`)
})
