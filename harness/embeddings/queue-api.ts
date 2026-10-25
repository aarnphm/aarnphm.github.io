import type { IncomingMessage, ServerResponse } from 'node:http'
import { execFile } from 'node:child_process'
import path from 'node:path'
import { promisify } from 'node:util'
import { isRecord } from '../../quartz/util/type-guards'

const execute = promisify(execFile)

export async function enqueuePdfs(
  request: IncomingMessage,
  response: ServerResponse,
  root: string,
  repository: string,
  port: number,
) {
  response.setHeader('Cache-Control', 'no-store')
  const hosts = new Set([`127.0.0.1:${port}`, `localhost:${port}`])
  const origin = request.headers.origin
  if (
    !hosts.has(request.headers.host ?? '') ||
    request.headers['x-garden-queue'] !== '1' ||
    (origin !== undefined && ![...hosts].some(host => origin === `http://${host}`))
  ) {
    response.writeHead(403).end('same-origin queue client required')
    return
  }
  if (!request.headers['content-type']?.startsWith('application/json')) {
    response.writeHead(415).end('JSON required')
    return
  }
  try {
    const chunks: Buffer[] = []
    let length = 0
    for await (const chunk of request) {
      if (!Buffer.isBuffer(chunk)) throw new Error('Invalid request body')
      length += chunk.length
      if (length > 16 * 1024) {
        response.writeHead(413).end('queue request too large')
        return
      }
      chunks.push(chunk)
    }
    const body: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'))
    if (
      !isRecord(body) ||
      Object.keys(body).some(key => key !== 'slugs') ||
      (body.slugs !== undefined &&
        (!Array.isArray(body.slugs) ||
          body.slugs.length > 128 ||
          !body.slugs.every(
            slug => typeof slug === 'string' && slug.length > 0 && slug.length <= 512,
          )))
    ) {
      response.writeHead(400).end('Use an optional array of public PDF slugs')
      return
    }
    const slugs: string[] = Array.isArray(body.slugs)
      ? body.slugs.filter((slug): slug is string => typeof slug === 'string')
      : []
    const { stdout } = await execute(
      path.join(repository, '.venv/bin/python'),
      [
        path.join(repository, 'quartz/scripts/pdf-search/queue-client.py'),
        'enqueue',
        '--root',
        root,
        ...slugs.flatMap(slug => ['--slug', slug]),
      ],
      { cwd: root, timeout: 120_000, maxBuffer: 128 * 1024 },
    )
    const report: unknown = JSON.parse(stdout)
    response.writeHead(202, { 'Content-Type': 'application/json' }).end(JSON.stringify(report))
  } catch (error) {
    if (error instanceof SyntaxError) {
      response.writeHead(400).end('Invalid JSON')
      return
    }
    const output = isRecord(error) && typeof error.stdout === 'string' ? error.stdout : ''
    let message = 'Could not enqueue PDFs; inspect the harness log'
    if (output) {
      try {
        const report: unknown = JSON.parse(output)
        if (isRecord(report) && typeof report.error === 'string') message = report.error
      } catch {
        console.error(output)
      }
    }
    console.error(error)
    response
      .writeHead(output ? 400 : 500, { 'Content-Type': 'application/json' })
      .end(JSON.stringify({ error: message }))
  }
}
