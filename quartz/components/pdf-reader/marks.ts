import {
  parsePdfMark,
  type PdfMark,
  type PdfMarkInput,
  type PdfMarksResponse,
} from '../../util/pdf-marks'

export class MarkApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly body: Record<string, unknown>,
  ) {
    super(message)
  }

  /** Network failures and server outages are worth retrying; 4xx answers are not. */
  get retryable(): boolean {
    return this.status === 0 || this.status === 429 || this.status >= 500
  }
}

async function request(path: string, init?: RequestInit): Promise<unknown> {
  let response: Response
  try {
    response = await fetch(path, { credentials: 'same-origin', cache: 'no-store', ...init })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new MarkApiError(0, 'offline', 'The network is unavailable.', {})
  }
  const body: unknown = await response.json().catch(() => null)
  if (response.ok) return body
  const record = typeof body === 'object' && body !== null ? (body as Record<string, unknown>) : {}
  throw new MarkApiError(
    response.status,
    typeof record.error === 'string' ? record.error : 'error',
    typeof record.message === 'string' ? record.message : `Request failed (${response.status}).`,
    record,
  )
}

function asMark(value: unknown): PdfMark {
  const mark = parsePdfMark(value)
  if (!mark)
    throw new MarkApiError(502, 'invalid-response', 'The server sent a malformed mark.', {})
  return mark
}

export async function fetchMarks(src: string, signal: AbortSignal): Promise<PdfMarksResponse> {
  const body = await request(`/api/pdf/marks?src=${encodeURIComponent(src)}`, { signal })
  const record = body as Record<string, unknown>
  return {
    src: String(record.src ?? src),
    doc: String(record.doc ?? ''),
    canWrite: record.canWrite === true,
    signInUrl: typeof record.signInUrl === 'string' ? record.signInUrl : '/comments/github/login',
    marks: Array.isArray(record.marks) ? record.marks.map(parsePdfMark).filter(isMark) : [],
    stale: Array.isArray(record.stale) ? record.stale.map(parsePdfMark).filter(isMark) : [],
  }
}

function isMark(value: PdfMark | null): value is PdfMark {
  return value !== null
}

export async function putMark(id: string, input: PdfMarkInput): Promise<PdfMark> {
  return asMark(
    await request(`/api/pdf/marks/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    }),
  )
}

export async function deleteMark(id: string, revision: number): Promise<void> {
  await request(`/api/pdf/marks/${id}`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ revision }),
    // A deletion queued during unload still has to reach the server.
    keepalive: true,
  })
}

export function conflictMark(error: MarkApiError): PdfMark | null {
  return error.body.current ? parsePdfMark(error.body.current) : null
}
