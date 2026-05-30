export const QUARTZ_DEV_EVENT_PREFIX = '__QUARTZ_DEV_EVENT__'

export type QuartzDevEvent =
  | { type: 'build:start'; epoch: string; reason: 'initial' | 'source' | 'content' }
  | { type: 'public:remove:start'; epoch: string }
  | { type: 'build:ready'; epoch: string; files: number; elapsedMs: number }
  | { type: 'build:error'; epoch: string; message: string }

export type SplitDevEventLinesResult = { lines: string[]; rest: string }

export function emitQuartzDevEvent(event: QuartzDevEvent): void {
  console.log(`${QUARTZ_DEV_EVENT_PREFIX}${JSON.stringify(event)}`)
}

export function parseQuartzDevEvent(line: string): QuartzDevEvent | undefined {
  const start = line.indexOf(QUARTZ_DEV_EVENT_PREFIX)
  if (start === -1) return undefined

  const raw = line.slice(start + QUARTZ_DEV_EVENT_PREFIX.length)
  try {
    const parsed: unknown = JSON.parse(raw)
    return isQuartzDevEvent(parsed) ? parsed : undefined
  } catch {
    return undefined
  }
}

export function splitDevEventLines(buffer: string, chunk: string): SplitDevEventLinesResult {
  const lines = `${buffer}${chunk}`.split(/\r?\n/)
  return { lines: lines.slice(0, -1), rest: lines.at(-1) ?? '' }
}

function isQuartzDevEvent(value: unknown): value is QuartzDevEvent {
  if (!isRecord(value) || typeof value.type !== 'string' || typeof value.epoch !== 'string') {
    return false
  }

  switch (value.type) {
    case 'build:start':
      return value.reason === 'initial' || value.reason === 'source' || value.reason === 'content'
    case 'public:remove:start':
      return true
    case 'build:ready':
      return typeof value.files === 'number' && typeof value.elapsedMs === 'number'
    case 'build:error':
      return typeof value.message === 'string'
    default:
      return false
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}
