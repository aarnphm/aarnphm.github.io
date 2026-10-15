// Serves a built harness directory without caching. Usage: node serve.mjs [dir=site] [port=7373]
import { readFile } from 'node:fs/promises'
import http from 'node:http'
import path from 'node:path'

const root = path.resolve(process.argv[2] ?? 'site')
const port = Number(process.argv[3] ?? 7373)
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' }
http
  .createServer(async (request, response) => {
    const url = new URL(request.url ?? '/', 'http://localhost')
    const file = path.join(root, decodeURIComponent(url.pathname.replace(/\/$/, '/index.html')))
    try {
      const body = await readFile(file)
      const type = types[path.extname(file)] ?? 'application/octet-stream'
      response.writeHead(200, { 'content-type': type, 'cache-control': 'no-store' })
      response.end(body)
    } catch {
      response.writeHead(404)
      response.end()
    }
  })
  .listen(port, () => console.log(`serving ${root} on ${port}`))
