export type ProblemDetailsOptions = {
  status: number
  title: string
  detail: string
  code: string
  resolution: string
  type?: string
  headers?: HeadersInit
}

export function problemResponse(request: Request, options: ProblemDetailsOptions): Response {
  const headers = new Headers(options.headers)
  headers.set('Cache-Control', 'no-store')
  headers.set('Content-Type', 'application/problem+json; charset=utf-8')
  headers.set('X-Content-Type-Options', 'nosniff')
  const body = JSON.stringify({
    type: options.type ?? 'about:blank',
    title: options.title,
    status: options.status,
    detail: options.detail,
    instance: new URL(request.url).pathname,
    code: options.code,
    resolution: options.resolution,
  })
  return new Response(request.method === 'HEAD' ? null : body, { status: options.status, headers })
}
